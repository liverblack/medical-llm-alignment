from datasets import load_dataset
import json, os, random

os.makedirs("data", exist_ok=True)

print("loading MedMCQA...")
medmcqa = load_dataset("openlifescienceai/medmcqa")["train"]
def is_valid_medmcqa(s):
    for opt in ["opa","opb","opc","opd"]:
        if not s[opt] or len(s[opt].strip())==0: return False
    if not s["question"] or len(s["question"].strip())==0: return False
    if s["cop"] not in [0,1,2,3]: return False
    if not s["exp"] or len(s["exp"].strip())<10: return False
    return True
medmcqa_clean = [s for s in medmcqa if is_valid_medmcqa(s)]
print(f"MedMCQA: {len(medmcqa)} -> {len(medmcqa_clean)}")

print("loading PubMedQA...")
pubmedqa = load_dataset("qiaojin/PubMedQA","pqa_labeled")["train"]
def is_valid_pubmedqa(s):
    if not s["question"] or len(s["question"].strip())==0: return False
    if not s["long_answer"] or len(s["long_answer"].strip())<20: return False
    if s["final_decision"] not in ["yes","no","maybe"]: return False
    return True
pubmedqa_clean = [s for s in pubmedqa if is_valid_pubmedqa(s)]
print(f"PubMedQA: {len(pubmedqa)} -> {len(pubmedqa_clean)}")

options_map = {0:"A",1:"B",2:"C",3:"D"}
SYSTEM = "You are a helpful and accurate medical assistant. Answer medical questions carefully and provide clear explanations."

def format_medmcqa(s):
    choices = f"A. {s['opa']}\nB. {s['opb']}\nC. {s['opc']}\nD. {s['opd']}"
    prompt = f"Question: {s['question']}\n\n{choices}\n\nPlease select the correct answer and explain your reasoning."
    response = f"The correct answer is {options_map[s['cop']]}.\n\nExplanation: {s['exp']}"
    return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":response}]}

def format_pubmedqa(s):
    context = " ".join(s["context"]["contexts"])
    prompt = f"Context: {context}\n\nQuestion: {s['question']}\n\nAnswer yes, no, or maybe, and explain your reasoning."
    response = f"Answer: {s['final_decision']}\n\nExplanation: {s['long_answer']}"
    return {"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},{"role":"assistant","content":response}]}

sft_data = [format_medmcqa(s) for s in medmcqa_clean] + [format_pubmedqa(s) for s in pubmedqa_clean]
random.shuffle(sft_data)

split = int(len(sft_data)*0.9)
train_data, eval_data = sft_data[:split], sft_data[split:]

with open("data/sft_train.jsonl","w") as f:
    for s in train_data: f.write(json.dumps(s)+"\n")
with open("data/sft_eval.jsonl","w") as f:
    for s in eval_data: f.write(json.dumps(s)+"\n")

print(f"\nSFT train: {len(train_data)} 条")
print(f"SFT eval:  {len(eval_data)} 条")
