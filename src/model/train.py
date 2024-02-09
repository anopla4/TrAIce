from accelerate import Accelerator, FullyShardedDataParallelPlugin, init_empty_weights, infer_auto_device_map
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
import torch
import wandb
import os
from datetime import datetime

# Initialize FullyShardedDataParallelPlugin
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Load datasets
train_dataset = load_dataset('json', data_files='pcapnew.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='pcapeval.jsonl', split='train')


# Formatting function for prompts
def formatting_func(example):
    return f"### Question: {example['input']}\n ### Answer: {example['output']}"


# Initialize tokenizer and model
hf_token = "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"
base_model_id = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left", add_eos_token=True, add_bos_token=True,
                                          use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoModelForCausalLM.from_pretrained(base_model_id)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.tie_weights()

device_map = infer_auto_device_map(model, max_memory={0: "30GiB", 1: "30GiB"},
                                   no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer',
                                                            'LlamaDecoderLayer'])

offload = "offload"
model = AutoModelForCausalLM.from_pretrained(base_model_id, use_auth_token=hf_token, device_map=device_map,
                                             offload_folder=offload, offload_state_dict=True)


# Tokenization and dataset preparation
def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# LoRA configuration and model preparation
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    bias="none", lora_dropout=0.05, task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model)

# Weights & Biases setup
wandb.login()
wandb_project = "journal-finetune"
os.environ["WANDB_PROJECT"] = wandb_project

# Training configuration
project = "journal-finetune"
base_model_name = "llama2-13b"
run_name = f"{base_model_name}-{project}"
output_dir = f"./{run_name}"

max_length = 512


def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(formatting_func(prompt), truncation=True, max_length=max_length, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result


# Training execution
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        max_steps=50, learning_rate=2.5e-5, bf16=False, optim="paged_adamw_8bit",
        logging_dir="./logs", save_strategy="steps", save_steps=10,
        evaluation_strategy="steps", eval_steps=5, do_eval=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()