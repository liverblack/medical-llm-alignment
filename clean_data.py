from datasets import load_dataset
import json

# ===== MedMCQA 清洗 =====
print("loading MedMCQA...")
medmcqa = load_dataset("openlifescienceai/medmcqa")
train = medmcqa["train"]

def is_valid_medmcqa(sample):
    # 选项不能为空
    for opt in ["opa", "opb", "opc", "opd"]:
        if not sample[opt] or len(sample[opt].strip()) == 0:
            return False
    # 问题不能为空
    if not sample["question"] or len(sample["question"].strip()) == 0:
        return False
    # 答案索引必须是 0-3
    if sample["cop"] not in [0, 1, 2, 3]:
        return False
    # exp 不能为空（我们需要它构造 SFT）
    if not sample["exp"] or len(sample["exp"].strip()) < 10:
        return False
    return True

cleaned = [s for s in train if is_valid_medmcqa(s)]
print(f"MedMCQA: {len(train)} -> {len(cleaned)} (过滤掉 {len(train)-len(cleaned)} 条)")

# ===== PubMedQA 清洗 =====
print("loading PubMedQA...")
pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
pqa = pubmedqa["train"]

def is_valid_pubmedqa(sample):
    if not sample["question"] or len(sample["question"].strip()) == 0:
        return False
    if not sample["long_answer"] or len(sample["long_answer"].strip()) < 20:
        return False
    if sample["final_decision"] not in ["yes", "no", "maybe"]:
        return False
    return True

cleaned_pqa = [s for s in pqa if is_valid_pubmedqa(s)]
print(f"PubMedQA: {len(pqa)} -> {len(cleaned_pqa)} (过滤掉 {len(pqa)-len(cleaned_pqa)} 条)")

# 保存清洗后的数据
import os
os.makedirs("data", exist_ok=True)

with open("data/medmcqa_cleaned.jsonl", "w") as f:
    for s in cleaned:
        f.write(json.dumps(s) + "\n")

with open("data/pubmedqa_cleaned.jsonl", "w") as f:
    for s in cleaned_pqa:
        f.write(json.dumps(s) + "\n")

print("\n保存完成：")
print(f"  data/medmcqa_cleaned.jsonl: {len(cleaned)} 条")
print(f"  data/pubmedqa_cleaned.jsonl: {len(cleaned_pqa)} 条")