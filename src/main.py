from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", token="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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf",token ="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")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
a = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(a)