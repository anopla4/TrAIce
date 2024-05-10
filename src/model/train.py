import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from accelerate import Accelerator, FullyShardedDataParallelPlugin, init_empty_weights, infer_auto_device_map
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
import torch
import wandb
from datetime import datetime
from pathlib import Path
from configparser import ConfigParser
from utils import print_trainable_parameters, generate_and_tokenize_prompt, generate_and_tokenize_prompt2, plot_data_lengths
# paths
base_path = "/data/arguellesa/traice/"
data_train_path = 'train_data.json'
data_test_path = 'validation_data.json'

# configuration parameters
c = ConfigParser()
c.read("model/.config")
hf_token = c.get("DEFAULT", "token")
base_model_id = c.get("DEFAULT", "model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")
# load datasets
train_dataset = load_dataset('json', data_files=base_path+data_train_path, split='train')
eval_dataset = load_dataset('json', data_files=base_path+data_test_path, split='train')

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

# accelerator
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

print("Tokenizing...")
# tokenization
tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True,
                                          token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
# tokenized_train_dataset = train_dataset.map(lambda x : generate_and_tokenize_prompt(tokenizer, x))
# tokenized_val_dataset = eval_dataset.map(lambda x : generate_and_tokenize_prompt(tokenizer, x))

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

max_length = 1700
tokenized_train_dataset = train_dataset.map(lambda x : generate_and_tokenize_prompt2(tokenizer, x, max_length=max_length))
tokenized_val_dataset = eval_dataset.map(lambda x : generate_and_tokenize_prompt2(tokenizer, x, max_length=max_length))

# print(tokenized_train_dataset[0]['input_ids'])
# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

# test model before fine-tuning
# eval_prompt = """\n\nNo.\tTime\tSource\tDestination\tProtocol\tLength\tInfo\n
# 1\t0\t192.168.0.2\t142.251.32.174\tTCP\t74\t37436  >  443 [SYN] Seq=0 Win=64240 Len=0 MSS=1460 SACK_PERM=1 TSval=1393734230 TSecr=0 WS=128
# \n2\t0.040585\t142.251.32.174\t192.168.0.2\tTCP\t58\t443  >  37436 [SYN, ACK] Seq=0 Ack=1 Win=65535 Len=0 MSS=1460
# \n3\t0.042195\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=1 Ack=1 Win=64240 Len=0
# \n4\t0.048234\t192.168.0.2\t142.251.32.174\tTLSv1.3\t444\tClient Hello
# \n5\t0.048712\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=1 Ack=391 Win=65535 Len=0
# \n6\t0.090352\t142.251.32.174\t192.168.0.2\tTLSv1.3\t1484\tServer Hello, Change Cipher Spec
# \n7\t0.091339\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=1431 Win=62920 Len=0
# \n8\t0.09594\t142.251.32.174\t192.168.0.2\tTCP\t4434\t443  >  37436 [ACK] Seq=1431 Ack=391 Win=65535 Len=4380 [TCP segment of a reassembled PDU]
# \n9\t0.096204\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=5811 Win=61320 Len=0
# \n10\t0.096295\t142.251.32.174\t192.168.0.2\tTLSv1.3\t884\tApplication Data
# \n11\t0.096504\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=6641 Win=62780 Len=0
# \n12\t0.10038\t192.168.0.2\t142.251.32.174\tTLSv1.3\t134\tChange Cipher Spec, Application Data
# \n13\t0.100832\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=6641 Ack=471 Win=65535 Len=0
# \n14\t0.101101\t192.168.0.2\t142.251.32.174\tTLSv1.3\t213\tApplication Data
# \n15\t0.101512\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=6641 Ack=630 Win=65535 Len=0
# \n16\t0.150147\t142.251.32.174\t192.168.0.2\tTLSv1.3\t1345\tApplication Data, Application Data
# \n17\t0.150373\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=630 Ack=7932 Win=62780 Len=0
# \n18\t0.199491\t192.168.0.2\t142.251.45.36\tTCP\t74\t48312  >  443 [SYN] Seq=0 Win=64240 Len=0 MSS=1460 SACK_PERM=1 TSv
# \n\nExplain the table, covering login attempts, brute force attacks, certificate issues, and other network activities. Include causes, detection methods, and mitigation strategies for each scenario. Aim for clarity and specificity, suitable for both technical and non-technical audiences."""

# print("Testing before training...")
# model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=max_length, pad_token_id=2).to(device)[0], skip_special_tokens=True))

print("Setting up lora...")
# set up lora
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
# print(model)

# LoRA configuration and model preparation
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
# print_trainable_parameters(model)

print("Applying accelerator...")
# apply the accelerator
# you can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)
# print(model)


# weights and biases setup
wandb.login()
wandb_project = "traice"
os.environ["WANDB_PROJECT"] = wandb_project

print("Configuring training...")
# training configuration
if torch.cuda.device_count() > 1: # If more than 1 GPU
    print("More than one GPU...")
    model.is_parallelizable = True
    model.model_parallel = True

project = "traice"
base_model_name = "llama2-13b"
run_name = f"{base_model_name}-{project}"
output_dir = f"{base_path}{run_name}"
tokenizer.pad_token = tokenizer.eos_token

# training execution
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        # bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()