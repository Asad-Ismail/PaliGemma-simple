import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

def generate_answer(model, processor, image, question, device="cuda"):
    """Generate answer for a single image and question."""
    if isinstance(image, list):
        image = list(image)
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    #tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    #print("\nToken by token:")
    #for i, token in enumerate(tokens):
    #     print(f"Position {i}: {token}")

    #inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        #num_beams=5,
        #early_stopping=True
    )
    # Only return the generated part of the response
    answer = processor.decode(outputs[0], skip_special_tokens=True)[len(question):]
    return answer


device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir="./hf_assets"

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