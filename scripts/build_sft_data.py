import json
import os
import random

os.makedirs("data", exist_ok=True)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

medmcqa = load_jsonl("data/medmcqa_cleaned.jsonl")
pubmedqa = load_jsonl("data/pubmedqa_cleaned.jsonl")

options_map = {0: "A", 1: "B", 2: "C", 3: "D"}

def format_medmcqa(s):
    question = s["question"]
    choices = f"A. {s['opa']}\nB. {s['opb']}\nC. {s['opc']}\nD. {s['opd']}"
    answer = options_map[s["cop"]]
    explanation = s["exp"]
    prompt = f"Question: {question}\n\n{choices}\n\nPlease select the correct answer and explain your reasoning."
    response = f"The correct answer is {answer}.\n\nExplanation: {explanation}"
    return {"messages": [
        {"role": "system", "content": "You are a helpful and accurate medical assistant. Answer medical questions carefully and provide clear explanations."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]}

def format_pubmedqa(s):
    question = s["question"]
    context = " ".join(s["context"]["contexts"])
    answer = s["final_decision"]
    explanation = s["long_answer"]
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer yes, no, or maybe, and explain your reasoning."
    response = f"Answer: {answer}\n\nExplanation: {explanation}"
    return {"messages": [
        {"role": "system", "content": "You are a helpful and accurate medical assistant. Answer medical questions carefully and provide clear explanations."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]}

sft_data = []
for s in medmcqa:
    sft_data.append(format_medmcqa(s))
for s in pubmedqa:
    sft_data.append(format_pubmedqa(s))

random.shuffle(sft_data)

# 9:1 切分 train / eval
split = int(len(sft_data) * 0.9)
train_data = sft_data[:split]
eval_data = sft_data[split:]

with open("data/sft_train.jsonl", "w") as f:
    for s in train_data:
        f.write(json.dumps(s) + "\n")

with open("data/sft_eval.jsonl", "w") as f:
    for s in eval_data:
        f.write(json.dumps(s) + "\n")

print(f"SFT train: {len(train_data)} 条")
print(f"SFT eval:  {len(eval_data)} 条")
print("\n=== 样本预览 ===")
sample = train_data[0]
for msg in sample["messages"]:
    print(f"\n[{msg['role']}]")
    print(msg["content"][:200])