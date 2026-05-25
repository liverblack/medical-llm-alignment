from datasets import load_dataset
from collections import Counter

dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
train = dataset["train"]

decisions = [s["final_decision"] for s in train]
print("标签分布：", Counter(decisions))

q_lengths = [len(s["question"]) for s in train]
a_lengths = [len(s["long_answer"]) for s in train]

print(f"问题长度：avg={sum(q_lengths)//len(q_lengths)}, max={max(q_lengths)}, min={min(q_lengths)}")
print(f"回答长度：avg={sum(a_lengths)//len(a_lengths)}, max={max(a_lengths)}, min={min(a_lengths)}")