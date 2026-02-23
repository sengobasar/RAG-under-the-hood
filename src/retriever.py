import numpy as np
from sentence_transformers import CrossEncoder


# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def search(index, query_vec, chunks, k=10, final_k=3):

    query_vec = np.array([query_vec])

    scores, indices = index.search(query_vec, k)

    candidates = []

    for idx in indices[0]:

        if 0 <= idx < len(chunks):
            candidates.append(chunks[idx])

    if not candidates:
        return []

    # Rerank
    pairs = [[query_vec, c] for c in candidates]

    rerank_pairs = [[str(query_vec), c] for c in candidates]

    scores = reranker.predict(rerank_pairs)

    ranked = sorted(
        zip(scores, candidates),
        reverse=True
    )

    return [c for _, c in ranked[:final_k]]