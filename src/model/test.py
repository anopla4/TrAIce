import torch
from configparser import ConfigParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from load_data import prepare_prompt
import sys
from pathlib import Path

# configuration parameters
c = ConfigParser()
c.read("model/.config")
hf_token = c.get("DEFAULT", "token")
base_model_id = c.get("DEFAULT", "model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments: <model_path> <pcap_file_path>")
        return
    
    model_path = sys.argv[1]
    pcap_path = sys.argv[2]

    print("Loading model...")
    # base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map={"": 0},
        quantization_config=bnb_config,
        token=hf_token)
    
    max_length = 1700
    tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                        add_eos_token=True, 
                                        add_bos_token=True,
                                        trust_remote_code=True,
                                        truncation=True, 
                                        max_length=max_length,
                                        token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding="max_length"
    tokenizer.padding_side="left"
    # Load the QLoRA adapter from the appropriate checkpoint directory
    ft_model = PeftModel.from_pretrained(model, model_path)

    # run inference
    eval_prompt_input = prepare_prompt(Path(pcap_path)) # build prompt with pcap file content
    model_input = tokenizer(eval_prompt_input,
                            return_tensors="pt",
                            truncation=True, 
                            max_length=max_length,
                            padding="max_length").to(device) # load model

    # generate response
    ft_model.eval()
    with torch.no_grad():
        print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=max_length, pad_token_id=2)[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()