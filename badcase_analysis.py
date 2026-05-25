import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SFT  = "outputs/sft-1.5b/final"
EVAL_FILE  = "data/sft_eval.jsonl"
N_SAMPLES  = 3000
OUTPUT     = "reports/badcases.jsonl"

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
                    return c, m["content"]
    return None, None

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_SFT)

data = load_jsonl(EVAL_FILE, N_SAMPLES)
badcases = []

for sample in tqdm(data):
    gold, gold_content = get_label(sample)
    if gold is None:
        continue
    msgs = sample["messages"][:2]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(response)
    if pred != gold:
        badcases.append({
            "question": sample["messages"][1]["content"],
            "gold": gold,
            "gold_explanation": gold_content,
            "pred": pred,
            "pred_explanation": response,
        })

os.makedirs("reports", exist_ok=True)
with open(OUTPUT, "w") as f:
    for b in badcases:
        f.write(json.dumps(b) + "\n")

print(f"\n共 {len(badcases)} 个 badcase，保存至 {OUTPUT}")
