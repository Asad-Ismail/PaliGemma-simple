from datasets import load_dataset 
from transformers import PaliGemmaProcessor
from transformers import TrainingArguments

cache_dir="./hf_dataset"
ds = load_dataset('HuggingFaceM4/VQAv2', split="train")
    #cache_dir=cache_dir, 
    #download_mode="reuse_dataset_if_exists",  # or force_redownload
    #trust_remote_code=True)

cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"] 
ds = ds.remove_columns(cols_remove)
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
val_ds = ds["test"]

# Loop through the dataset
#for idx, sample in enumerate(train_ds):
#    print(f"Index: {idx}")
#    print(f"Sample: {sample}")
#    break


model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

args=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

    
trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args
        )
trainer.train()