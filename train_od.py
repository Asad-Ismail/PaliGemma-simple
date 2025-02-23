import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.optim import AdamW
from PIL import Image
import logging
import os
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from utils import visualize_predictions_od
from utils import resize_boxes
from datasets import load_dataset
import io
import base64
import wandb  # for logging
from torch.nn.utils import clip_grad_norm_
from utils import get_parameter_statistics, compute_grad_norm
from transformers import get_linear_schedule_with_warmup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cache_dir = "./hf_assets"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and processor
model_id = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=False)


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
    # This is how paligemma is trained y coordinates are before x weired but lets do it same way so model does not have to relearn the mapping
    return [
        get_location_token(box[1], height),  # ymin
        get_location_token(box[0], width),   # xmin
        get_location_token(box[3], height),  # ymax
        get_location_token(box[2], width),   # xmax
    ]


class DetectionDataset(Dataset):
    """Pure PyTorch Dataset for object detection."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image'].convert('RGB')
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
    print(tokens.keys())
    print(tokens['attention_mask'])
    return tokens.to(torch.bfloat16).to(device)

def train_paligemma(
    train_dataset=None,
    val_dataset=None,
    num_epochs=5,
    batch_size=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    warmup_steps=1000,
    max_grad_norm=1.0,
    checkpoint_dir="checkpoints",
    fp16=True,
    visualize_every_n_steps=1000
):
    """Train PaLI-GEMMA model for object detection task."""

    wandb.init(project="paligemma-od", config={
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "batch_size": batch_size
    })
    

    train_dataset = DetectionDataset(train_dataset)
    val_dataset = DetectionDataset(val_dataset) if val_dataset else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        #pin_memory=True,
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

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        cache_dir=cache_dir,
        local_files_only=False,
        torch_dtype=torch.bfloat16 if fp16 else torch.float32
    ).to(device)
    
    # Set use_cache False before enabling gradient checkpointing
    model.config.use_cache = False
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False,
        lora_alpha=32,
        bias="none"
    )

    #model = get_peft_model(model, lora_config)
    
    # Enable input gradients before gradient checkpointing
    model.enable_input_require_grads()
    model.config.use_memory_efficient_attention = True
    # Enable gradient checkpointing after setting up LoRA and input gradients
    #model.gradient_checkpointing_enable()
    #model.print_trainable_parameters()
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    num_training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    global_step = 0
    best_val_loss = float('inf')

    # Create visualization directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_grad_norms = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Clear gradients at the start
            optimizer.zero_grad()
            
            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward passs
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            epoch_grad_norms.append(grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Scheduler step
            scheduler.step()
            
            # Log training metrics
            if global_step % 100 == 0:
                param_stats = get_parameter_statistics(model)
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    **{f"params/{k}/mean": v['mean'] for k, v in param_stats.items()},
                    **{f"params/{k}/std": v['std'] for k, v in param_stats.items()},
                }, step=global_step)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if global_step % visualize_every_n_steps == 0:
                print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
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
                            break
                    
                    avg_val_loss = total_val_loss / val_steps
                    logger.info(f"Validation loss: {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
                        processor.save_pretrained(os.path.join(checkpoint_dir, "best_model"))

                    # Generate and save visualizations
                    viz_subdir = os.path.join(viz_dir, f"step_{global_step}")
                    os.makedirs(viz_subdir, exist_ok=True)
                    visualize_predictions_od(model, processor, val_dataset, output_dir=viz_subdir, epoch=epoch + 1)
                    model.train()
            global_step += 1
            
    wandb.finish()
    return model, processor



def convert_to_training_format(example):
    return {
        'image': Image.open(io.BytesIO(base64.b64decode(example['image']))),
        'boxes': example['boxes'],
        'categories': example['categories']
    }


if __name__ == "__main__":

    dataset = load_dataset("AsadIsmail/coco-cats-dogs-small-Detection",cache_dir=cache_dir)
    train_data = [convert_to_training_format(example) for example in dataset['train']]
    val_data = [convert_to_training_format(example) for example in dataset['validation']]

    model, processor = train_paligemma(
        train_dataset=train_data,
        val_dataset=val_data,
        num_epochs=500,
        batch_size=2,
        learning_rate=2e-5,
        checkpoint_dir="paligemma_detection_checkpoints"
    )