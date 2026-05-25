import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SFT  = "outputs/sft-1.5b/final"
OUTPUT_DIR = "outputs/dpo-3b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, MODEL_SFT)
model = model.merge_and_unload()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_dataset = load_dataset("json", data_files="data/dpo_train.jsonl", split="train")
eval_dataset  = load_dataset("json", data_files="data/dpo_eval.jsonl",  split="train")

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    max_length=1024,
    
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
print("DPO训练完成，保存至", OUTPUT_DIR + "/final")
