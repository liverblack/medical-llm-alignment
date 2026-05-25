from datasets import load_dataset

print("loading PubMedQA...")
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

print(dataset)
print("\n=== 第一条样本 ===")
sample = dataset["train"][0]
for k, v in sample.items():
    print(f"\n[{k}]")
    print(v)