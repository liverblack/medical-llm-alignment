import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

MODEL_BASE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SFT  = "outputs/sft-1.5b/final"
EVAL_FILE  = "data/sft_eval.jsonl"
CORPUS     = "data/rag/corpus.jsonl"
N_SAMPLES  = 300

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
                    return c
    return None

SYSTEM = "You are a helpful and accurate medical assistant. Answer medical questions carefully and provide clear explanations. Always respond in the same language as the user's question."

def build_prompt_with_context(question, context_docs):
    context = "\n\n".join([f"[{i+1}] {d['doc']['text']}" for i, d in enumerate(context_docs)])
    return f"Reference context:\n{context}\n\nBased on the above context, answer the following question:\n{question}"

def evaluate(model, tokenizer, data, retrieve_fn, label):
    correct = 0
    total = 0
    for sample in tqdm(data, desc=label):
        gold = get_label(sample)
        if gold is None:
            continue
        question = sample["messages"][1]["content"]
        if retrieve_fn:
            docs = retrieve_fn(question)
            content = build_prompt_with_context(question, docs)
        else:
            content = question
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)
        if pred is None:
            continue
        total += 1
        if pred == gold:
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"\n{label}: {acc:.4f} ({correct}/{total})")
    return acc

corpus = load_jsonl(CORPUS)
texts = [d["text"] for d in corpus]
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-m3")
vecs = embed_model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

print("Loading reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def retrieve_bm25(query, k=3):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"doc": corpus[i], "score": float(scores[i])} for i in top_k]

def retrieve_dense(query, k=3):
    vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(vec, k)
    return [{"doc": corpus[idx], "score": float(scores[0][i])} for i, idx in enumerate(indices[0])]

def retrieve_rerank(query, k=3):
    bm25_r = retrieve_bm25(query, k=5)
    dense_r = retrieve_dense(query, k=5)
    seen = set()
    candidates = []
    for r in bm25_r + dense_r:
        if r["doc"]["id"] not in seen:
            seen.add(r["doc"]["id"])
            candidates.append(r)
    pairs = [[query, c["doc"]["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{"doc": r[1]["doc"], "score": float(r[0])} for r in ranked[:k]]

eval_data = load_jsonl(EVAL_FILE, N_SAMPLES)
mcq_data = [s for s in eval_data if get_label(s) is not None]

print("Loading SFT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, MODEL_SFT)

results = {}
results["No RAG"] = evaluate(model, tokenizer, mcq_data, None, "No RAG")
results["BM25"]   = evaluate(model, tokenizer, mcq_data, retrieve_bm25, "BM25")
results["Dense"]  = evaluate(model, tokenizer, mcq_data, retrieve_dense, "Dense")
results["Rerank"] = evaluate(model, tokenizer, mcq_data, retrieve_rerank, "Rerank")

print(f"\n{'='*40}")
for k, v in results.items():
    print(f"{k:10s}: {v:.4f}")

os.makedirs("reports", exist_ok=True)
with open("reports/rag_ablation.md", "w") as f:
    f.write("# RAG Ablation Results\n\n")
    f.write("| Method | Accuracy |\n|--------|----------|\n")
    for k, v in results.items():
        f.write(f"| {k} | {v:.4f} |\n")
print("\n结果已保存至 reports/rag_ablation.md")
