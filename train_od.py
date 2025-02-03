import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from PIL import Image
import logging
import os
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from utils import visualize_ground_truth, visualize_predictions_od
import json
from utils import resize_boxes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cache_dir = "./hf_assets"
device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize model and processor
model_id = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=False)



def load_coco_subset(train_ann_path, val_ann_path, images_dir):
    """Load COCO subset annotations."""
    def load_split(ann_path, img_dir_sfx):
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Create lookup dictionaries
        images = {img['id']: img['file_name'] for img in data['images']}
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Group annotations by image
        annotations = []  # Change to list of dicts instead of dict
        image_anns = {}
        
        # First group annotations by image
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_anns:
                image_anns[img_id] = {
                    'image_path': os.path.join(images_dir, img_dir_sfx, images[img_id]),
                    'boxes': [],
                    'categories': []
                }
            x, y, w, h = ann['bbox']
            image_anns[img_id]['boxes'].append([x, y, x + w, y + h])
            image_anns[img_id]['categories'].append(categories[ann['category_id']])
        
        # Convert to list
        annotations = list(image_anns.values())
        return annotations

    train_data = load_split(train_ann_path, "train2017")
    val_data = load_split(val_ann_path, "val2017")
    return train_data, val_data



def get_location_token(coord, scale_factor):
    """Convert coordinate to location token index using PaLI-GEMMA scaling.
    
    Args:
        coord: Unnormalized coordinate value
        scale_factor: Image dimension (width or height) to scale against
    """
    # Scale to 0-1024 range as per PaLI-GEMMA spec
    bin_idx = min(int((coord / scale_factor) * 1024), 1023)
    return f"<loc{bin_idx:04d}>"

def box_to_location_tokens(box, width, height):
    """Convert box coordinates [xmin, ymin, xmax, ymax] to location tokens."""
    return [
        get_location_token(box[0], width),   # xmin
        get_location_token(box[1], height),  # ymin
        get_location_token(box[2], width),   # xmax
        get_location_token(box[3], height)   # ymax
    ]


class DetectionDataset(Dataset):
    """Pure PyTorch Dataset for object detection."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        return {
            'image': image,
            'boxes': torch.tensor(item['boxes']),  # Convert to tensor
            'labels': item['categories'] 
        }

def collate_fn(examples):
    """Collate function that handles both training and visualization."""
    # Collect basic data
    images = [ex['image'] for ex in examples]
    boxes = [ex['boxes'] for ex in examples]
    labels = [ex['labels'] for ex in examples]

    orig_sizes = [img.size[::-1] for img in images]  # (h, w)
    target_size = (224, 224)

    texts = []       # prompt (prefix) sent to the model
    text_labels = [] # target outputs for the model

    for boxes_per_img, categories, orig_size in zip(boxes, labels, orig_sizes):
        boxes_resized = resize_boxes(boxes_per_img.tolist(), orig_size, target_size)
        unique_cats = sorted(set(categories))
        prefix = "detect " + " ; ".join(str(cat) for cat in unique_cats)

        detections = []
        for box, cat in zip(boxes_resized, categories):
            loc_tokens = box_to_location_tokens(box, target_size[1], target_size[0])
            detections.append("".join(loc_tokens) + " " + str(cat))
        
        # Now, use prefix as the instruction and detections as the target
        texts.append(f"<image>{prefix}")
        text_labels.append(" ; ".join(detections))

    tokens = processor(
        text=texts,
        images=images,
        suffix=text_labels,
        return_tensors="pt",
        padding="longest"
    )
    return tokens.to(torch.bfloat16).to(device)

def train_paligemma(
    train_dataset=None,
    val_dataset=None,
    num_epochs=5,
    batch_size=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    checkpoint_dir="checkpoints",
    fp16=True,
    visualize_every_n_epochs=1
):
    """Train PaLI-GEMMA model for object detection task."""
    

    train_dataset = DetectionDataset(train_dataset)
    val_dataset = DetectionDataset(val_dataset) if val_dataset else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        visualize_ground_truth(val_dataset, output_dir="output/gt_visualizations", epoch=0)
    

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        cache_dir=cache_dir,
        local_files_only=False,
        torch_dtype=torch.bfloat16 if fp16 else torch.float32
    ).to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=warmup_steps
    )
    
    # Training optimizations
    model.enable_input_require_grads()
    model.config.use_memory_efficient_attention = True
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            torch.cuda.empty_cache()
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            total_train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({
                'loss': total_train_loss / train_steps,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if global_step % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
        
        # Validation
        if val_dataset:
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    total_val_loss += outputs.loss.item()
                    val_steps += 1
            
            avg_val_loss = total_val_loss / val_steps
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
                processor.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
        
        if (epoch) % visualize_every_n_epochs == 0 and val_dataset:
            visualize_predictions_od(model, processor, val_loader, output_dir="output/pred_visualizations", epoch=epoch + 1)
    
    return model, processor

def inference(model, processor, image_path):
    """Run inference on a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text="<image>detect", images=image, return_tensors="pt")
    inputs = inputs.to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        temperature=0.7
    )
    
    return processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load dataset
    train_ann_path = "/home/asad/dev/GLIP/DATASET/coco/annotations/instances_train2017_subset.json"
    val_ann_path = "/home/asad/dev/GLIP/DATASET/coco/annotations/instances_val2017_subset.json"
    images_dir = "/home/asad/dev/GLIP/DATASET/coco/"  

    # Load custom COCO subset
    train_data, val_data = load_coco_subset(train_ann_path, val_ann_path, images_dir)

    
    # Train model
    model, processor = train_paligemma(
        train_dataset=train_data,
        val_dataset=val_data,
        num_epochs=5,
        batch_size=2,
        learning_rate=2e-5,
        checkpoint_dir="paligemma_detection_checkpoints"
    )