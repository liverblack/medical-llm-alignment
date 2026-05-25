import json
import os

os.makedirs("data/rag", exist_ok=True)

def load_jsonl(path):
    data = []
    with open(path) as f:
        for l in f:
            try:
                data.append(json.loads(l))
            except:
                pass
    return data

pubmedqa = load_jsonl("data/pubmedqa_cleaned.jsonl")

corpus = []
for item in pubmedqa:
    contexts = item["context"]["contexts"]
    question = item["question"]
    answer = item["long_answer"]
    pubid = item["pubid"]

    for i, ctx in enumerate(contexts):
        if len(ctx.strip()) < 50:
            continue
        corpus.append({
            "id": f"{pubid}_{i}",
            "text": ctx.strip(),
            "question": question,
            "answer": answer,
            "decision": item["final_decision"]
        })

with open("data/rag/corpus.jsonl", "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")

print(f"文档库构建完成：{len(corpus)} 条")
print("\n=== 样本预览 ===")
print(json.dumps(corpus[0], indent=2)[:500])

medmcqa = load_jsonl("data/medmcqa_cleaned.jsonl")
options_map = {0:"A", 1:"B", 2:"C", 3:"D"}
extra = []
for item in medmcqa:
    if not item.get("exp") or len(item["exp"].strip()) < 30:
        continue
    extra.append({
        "id": f"mcqa_{item['id']}",
        "text": f"Q: {item['question']} A: {options_map[item['cop']]}. {item['exp']}",
        "question": item["question"],
        "answer": item["exp"],
        "decision": options_map[item["cop"]]
    })

corpus.extend(extra)

with open("data/rag/corpus.jsonl", "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")

print(f"扩充后文档库：{len(corpus)} 条")
