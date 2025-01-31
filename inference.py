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
