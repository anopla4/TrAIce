import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7B, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token = "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"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_auth_token = "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")

"""Now load the QLoRA adapter from the appropriate checkpoint directory, i.e. the best performing model checkpoint:"""

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "llama2-7b-journal-finetune/checkpoint-50")

"""and run your inference!

Let's try the same `eval_prompt` and thus `model_input` as above, and see if the new finetuned model performs better.
"""

eval_prompt = pd.read_json('pcapnew.jsonl', lines=True)
eval_prompt_input = f"### Question: {eval_prompt['input'][0]}\n ### Answer: "
model_input = tokenizer(eval_prompt_input, return_tensors="pt").to(device)


ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=2000)[0], skip_special_tokens=True))