import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SFT  = "outputs/sft-1.5b/final"
MODEL_DPO  = "outputs/dpo-3b/final"
EVAL_FILE  = "data/sft_eval.jsonl"
N_SAMPLES  = 500

def load_jsonl(path, n=None):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n: break
            data.append(json.loads(line))
    return data

def extract_answer(text):
    for c in ["A","B","C","D"]:
        if f"correct answer is {c}" in text or f"answer is {c}" in text:
            return c
    return None

def get_label(msg):
    for m in msg["messages"]:
        if m["role"] == "assistant":
            for c in ["A","B","C","D"]:
                if f"correct answer is {c}" in m["content"]:
                    return c
    return None

def evaluate(model, tokenizer, data, label):
    correct = 0
    total = 0
    for sample in tqdm(data, desc=label):
        gold = get_label(sample)
        if gold is None:
            continue
        msgs = sample["messages"][:2]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred is None:
            continue
        total += 1
        if pred == gold:
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"\n{label}: {acc:.4f} ({correct}/{total})")
    return acc

eval_data = load_jsonl(EVAL_FILE, N_SAMPLES)
mcq_data = [s for s in eval_data if get_label(s) is not None]

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

print("=== Loading SFT ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_SFT)
sft_acc = evaluate(model, tokenizer, mcq_data, "SFT")

print("\n=== Loading DPO ===")
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_DPO)
dpo_acc = evaluate(model, tokenizer, mcq_data, "DPO")

print(f"\n{'='*40}")
print(f"SFT accuracy: {sft_acc:.4f}")
print(f"DPO accuracy: {dpo_acc:.4f}")
print(f"Delta:        {dpo_acc - sft_acc:+.4f}")
