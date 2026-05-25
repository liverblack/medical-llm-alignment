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
N_SAMPLES  = 200

def load_jsonl(path, n=None):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n: break
            data.append(json.loads(line))
    return data

def get_label(msg):
    for m in msg["messages"]:
        if m["role"] == "assistant":
            for c in ["A","B","C","D"]:
                if f"correct answer is {c}" in m["content"]:
                    return c
    return None

def has_repetition(text, threshold=3):
    sentences = text.split(".")
    seen = {}
    for s in sentences:
        s = s.strip()[:50]
        if len(s) < 10:
            continue
        seen[s] = seen.get(s, 0) + 1
        if seen[s] >= threshold:
            return True
    return False

def evaluate_alignment(model, tokenizer, data, label):
    format_ok = 0
    refusal = 0
    total_len = 0
    repetition = 0
    total = 0

    for sample in tqdm(data, desc=label):
        if get_label(sample) is None:
            continue
        msgs = sample["messages"][:2]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        total += 1
        if "correct answer is" in response:
            format_ok += 1
        if any(w in response.lower() for w in ["i'm not sure", "i cannot", "uncertain", "consult"]):
            refusal += 1
        total_len += len(response)
        if has_repetition(response):
            repetition += 1

    print(f"\n=== {label} ===")
    print(f"格式遵循率:  {format_ok/total:.2%} ({format_ok}/{total})")
    print(f"拒答/谨慎率: {refusal/total:.2%} ({refusal}/{total})")
    print(f"平均回答长度: {total_len/total:.0f} 字符")
    print(f"重复输出率:  {repetition/total:.2%} ({repetition}/{total})")
    return {
        "format": format_ok/total,
        "refusal": refusal/total,
        "avg_len": total_len/total,
        "repetition": repetition/total
    }

eval_data = load_jsonl(EVAL_FILE, N_SAMPLES)
mcq_data = [s for s in eval_data if get_label(s) is not None]

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)

print("Loading SFT...")
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_SFT)
sft_result = evaluate_alignment(model, tokenizer, mcq_data, "SFT")

print("\nLoading DPO...")
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_DPO)
dpo_result = evaluate_alignment(model, tokenizer, mcq_data, "DPO")

os.makedirs("reports", exist_ok=True)
with open("reports/dpo_vs_sft.md", "w") as f:
    f.write("# SFT vs DPO Alignment Evaluation\n\n")
    f.write("| Metric | SFT | DPO | Delta |\n|--------|-----|-----|-------|\n")
    for k in sft_result:
        s, d = sft_result[k], dpo_result[k]
        delta = d - s
        f.write(f"| {k} | {s:.4f} | {d:.4f} | {delta:+.4f} |\n")
print("\n结果已保存至 reports/dpo_vs_sft.md")
