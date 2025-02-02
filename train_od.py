import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from datasets import load_dataset
from PIL import Image
import logging
import os
from tqdm import tqdm
from peft import get_peft_model, LoraConfig

cache_dir = "./hf_assets"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and processor
model_id = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=False)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def collate_fn(examples):
    """Collate function following PaLI-GEMMA's exact format for detection."""
    images = [example["image"].convert("RGB") for example in examples]
    
    # Create detection prompt texts with location tokens
    texts = []
    labels = []
    for example in examples:
        boxes = example["boxes"]
        categories = example["labels"]
        width, height = example["image"].size
        
        # Get unique categories for the prefix
        unique_categories = sorted(set(categories))
        prefix = "detect " + " ; ".join(str(cat) for cat in unique_categories)
        
        # Create location tokens and labels for each object
        detection_text = []
        for box, category in zip(boxes, categories):
            loc_tokens = box_to_location_tokens(box, width, height)
            detection_text.append("".join(loc_tokens) + " " + str(category))
        
        # Combine with proper format
        full_text = f"{prefix} {' ; '.join(detection_text)}"
        texts.append(f"<image>{full_text}")
        labels.append(full_text)

    # Process inputs
    tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16 if device == "cuda" else torch.float32)
    return tokens

class DetectionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert annotations to boxes and labels
        boxes = []
        labels = []
        for ann in item['annotations']:
            # bbox format: [x, y, width, height] -> convert to [x1, y1, x2, y2]
            bbox = ann['bbox']
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])
            labels.append(ann['category_id'])
        
        return {
            "image": item["image"],
            "boxes": boxes,
            "labels": labels
        }

def load_balloon_dataset():
    """Load and prepare the balloon dataset."""
    ds = load_dataset('frgfm/balloon', split="train")
    
    # Split into train and validation
    ds = ds.train_test_split(test_size=0.2)
    return ds["train"], ds["test"]

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
    fp16=True
):
    """Train PaLI-GEMMA model for object detection task."""
    
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
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    
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
    train_ds, val_ds = load_balloon_dataset()
    
    # Train model
    model, processor = train_paligemma(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=5,
        batch_size=2,
        learning_rate=2e-5,
        checkpoint_dir="paligemma_detection_checkpoints"
    )