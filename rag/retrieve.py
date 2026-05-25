import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import os
import numpy as np
from rank_bm25 import BM25Okapi

def load_corpus(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

corpus = load_corpus("data/rag/corpus.jsonl")
texts = [doc["text"] for doc in corpus]
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

def retrieve_bm25(query, k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"doc": corpus[i], "score": float(scores[i]), "method": "bm25"} for i in top_k]

def retrieve_dense(query, k=5, model=None, index=None):
    import faiss
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    return [{"doc": corpus[idx], "score": float(scores[0][i]), "method": "dense"} for i, idx in enumerate(indices[0])]

def build_dense_index(model):
    import faiss
    print("Building dense index...")
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index

if __name__ == "__main__":
    query = "What is the role of mitochondria in programmed cell death?"

    print("=== BM25 ===")
    results = retrieve_bm25(query, k=3)
    for i, r in enumerate(results):
        print(f"[{i+1}] score={r['score']:.4f} | {r['doc']['text'][:150]}")

    print("\n=== Dense (bge-m3) ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    index = build_dense_index(model)
    results = retrieve_dense(query, k=3, model=model, index=index)
    for i, r in enumerate(results):
        print(f"[{i+1}] score={r['score']:.4f} | {r['doc']['text'][:150]}")