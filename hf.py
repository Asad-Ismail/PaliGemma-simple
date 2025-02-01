import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from datasets import load_dataset
from PIL import Image
import requests
import logging
from inference import generate_answer
import os
from tqdm import tqdm
from fsspec.exceptions import FSTimeoutError
import  aiohttp

cache_dir="./hf_assets"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and processor
model_id = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    texts = [f"<image>{text}" for text in texts]
    tokens = processor(text=texts, images=images, suffix=labels,return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

class VQADataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item['image'], Image.Image):
            image = item['image']
        else:
            from io import BytesIO
            image = Image.open(BytesIO(item['image']))
        
        return {
            "image": image,
            "question": item["question"],
            "multiple_choice_answer": item["answer"]
        }


def load_vqa_dataset():
    """Load and prepare a small subset of the VQA dataset."""
    ds = load_dataset('HuggingFaceM4/VQAv2', 
                     split="train",
                     cache_dir=cache_dir, 
                     download_mode="reuse_dataset_if_exists",  
                     trust_remote_code=True,
                     # Required for large datasets
                     storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
    
    # Remove unnecessary columns
    cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
    ds = ds.remove_columns(cols_remove)
    
    # Split into train and validation
    ds = ds.train_test_split(test_size=0.1)
    return ds["train"], ds["test"]

def train_paligemma(
    model_id="google/paligemma-3b-mix-224",
    train_dataset=None,
    val_dataset=None,
    num_epochs=2,
    batch_size=16,
    learning_rate=2e-5,
    weight_decay=1e-6,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    checkpoint_dir="checkpoints",
    device="cuda",
    fp16=True
):
    """Train PaLI-GEMMA model for VQA task."""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model (processor is now global)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if fp16 else torch.float32
    ).to(device)
    
    # Freeze vision tower, unfreeze projector
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True
    
    # Create dataset objects
    train_dataset = VQADataset(train_dataset)
    val_dataset = VQADataset(val_dataset) if val_dataset else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=warmup_steps
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            total_train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_train_loss / train_steps,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Save checkpoint periodically
            if global_step % 1000 == 0:
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
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
                processor.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
    
    return model, processor


if __name__ == "__main__":
    train = True
    if train:
        # Load dataset
        train_ds, val_ds = load_vqa_dataset()
        
        # Train model
        model, processor = train_paligemma(
            train_dataset=train_ds,
            val_dataset=val_ds,
            num_epochs=2,
            batch_size=16,
            learning_rate=2e-5,
            checkpoint_dir="paligemma_checkpoints"
        )
    else:
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device="cpu"
        model_id =  "google/paligemma-3b-mix-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir,
        local_files_only=False,torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32 ).to(device)
        processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir,
        local_files_only=False)
        
        # Single image example
        prompt = "What is on the flower?"
        image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        
        answer = generate_answer(model, processor, raw_image, prompt, device)
        print(f"Single image answer: {answer}")

        '''
        # Image pair example
        model_id = "google/paligemma-3b-ft-nlvr2-448"  # checkpoint tuned for multiple images
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        prompt = "answer en Which of the two pictures shows a snowman, first or second?"
        stop_sign_image = Image.open(
            requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
        )
        snow_image = Image.open(
            requests.get(
                "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", 
                stream=True
            ).raw
        )
        answer = generate_answer(model, processor, [snow_image, stop_sign_image], prompt)
        print(f"Dual image answer: {answer}")
    
        '''


    