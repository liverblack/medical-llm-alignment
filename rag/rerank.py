import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, candidates, top_k=3):
    pairs = [[query, c["doc"]["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{"doc": r[1]["doc"], "score": float(r[0]), "method": "rerank"} for r in ranked[:top_k]]

if __name__ == "__main__":
    from retrieve import retrieve_bm25, retrieve_dense, build_dense_index
    from sentence_transformers import SentenceTransformer

    query = "What is the role of mitochondria in programmed cell death?"

    embed_model = SentenceTransformer("BAAI/bge-m3")
    index = build_dense_index(embed_model)

    bm25_results = retrieve_bm25(query, k=5)
    dense_results = retrieve_dense(query, k=5, model=embed_model, index=index)

    seen = set()
    candidates = []
    for r in bm25_results + dense_results:
        doc_id = r["doc"]["id"]
        if doc_id not in seen:
            seen.add(doc_id)
            candidates.append(r)

    print(f"merged candidates: {len(candidates)}")
    reranked = rerank(query, candidates, top_k=3)

    print("\n=== Reranked Results ===")
    for i, r in enumerate(reranked):
        print(f"[{i+1}] score={r['score']:.4f} | {r['doc']['text'][:200]}")
