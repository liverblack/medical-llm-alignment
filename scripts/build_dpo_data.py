import json
import os
import random

os.makedirs("data", exist_ok=True)

SYSTEM = "You are a helpful and accurate medical assistant. Answer medical questions carefully and provide clear explanations. Always respond in the same language as the user's question."

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

badcases = load_jsonl("reports/badcases.jsonl")

dpo_data = []

for b in badcases:
    if not b["pred"] or b["pred"] == b["gold"]:
        continue
    if not b["gold_explanation"] or len(b["gold_explanation"].strip()) < 20:
        continue
    if not b["pred_explanation"] or len(b["pred_explanation"].strip()) < 20:
        continue

    prompt = b["question"]
    chosen = b["gold_explanation"]
    rejected = b["pred_explanation"]

    if chosen == rejected:
        continue

    dpo_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "system": SYSTEM,
    })

random.shuffle(dpo_data)

split = int(len(dpo_data) * 0.9)
train_data = dpo_data[:split]
eval_data = dpo_data[split:]

with open("data/dpo_train.jsonl", "w") as f:
    for s in train_data:
        f.write(json.dumps(s) + "\n")

with open("data/dpo_eval.jsonl", "w") as f:
    for s in eval_data:
        f.write(json.dumps(s) + "\n")

print(f"DPO train: {len(train_data)} 条")
print(f"DPO eval:  {len(eval_data)} 条")

print("\n=== 样本预览 ===")
sample = train_data[0]
print(f"\n[prompt]\n{sample['prompt'][:200]}")
print(f"\n[chosen]\n{sample['chosen'][:200]}")
print(f"\n[rejected]\n{sample['rejected'][:200]}")
