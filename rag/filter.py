BM25_THRESHOLD = 10.0
DENSE_THRESHOLD = 0.4

def filter_retrieve_bm25(query, bm25, corpus, k=3):
    tokens = query.lower().split()
    from rank_bm25 import BM25Okapi
    scores = bm25.get_scores(tokens)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [{"doc": corpus[i], "score": float(scores[i])} for i in top_k]
    filtered = [r for r in results if r["score"] >= BM25_THRESHOLD]
    return filtered

def filter_retrieve_dense(query, embed_model, index, corpus, k=3):
    import numpy as np
    vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(vec, k)
    results = [{"doc": corpus[idx], "score": float(scores[0][i])} for i, idx in enumerate(indices[0])]
    filtered = [r for r in results if r["score"] >= DENSE_THRESHOLD]
    return filtered
