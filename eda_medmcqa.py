from datasets import load_dataset
from collections import Counter

print("loading MedMCQA...")
dataset = load_dataset("openlifescienceai/medmcqa")
train = dataset["train"]

print(dataset)
print("\n=== 第一条样本 ===")
sample = train[0]
for k, v in sample.items():
    print(f"\n[{k}]")
    print(v)

print("\n科目分布（前10）：")
subjects = [s["subject_name"] for s in train]
for subject, count in Counter(subjects).most_common(10):
    print(f"  {subject}: {count}")