import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SFT  = "outputs/sft-1.5b/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, MODEL_SFT)

REWRITE_SYSTEM = """You are a medical search query optimizer. 
Rewrite the given question into a concise, keyword-rich search query suitable for retrieving relevant medical literature.
Output only the rewritten query, nothing else."""

def rewrite_query(question):
    msgs = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Rewrite this question for medical literature search:\n{question}"}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    rewritten = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return rewritten.strip()

if __name__ == "__main__":
    questions = [
        "What happens to the kidney when there is chronic urethral obstruction?",
        "Which bacteria causes endocarditis in IV drug users?",
        "慢性尿道梗阻会对肾脏造成什么影响？",
        "静脉注射毒品者心内膜炎最常见的病原体是什么？",
        "糖尿病会怎么影响眼睛？",
    ]
    for q in questions:
        rewritten = rewrite_query(q)
        print(f"Original:  {q}")
        print(f"Rewritten: {rewritten}")
        print()