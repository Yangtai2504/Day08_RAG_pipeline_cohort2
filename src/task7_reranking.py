"""
Task 7 — Reranking Module.

Implement 2 phương pháp:
    1. RRF (Reciprocal Rank Fusion) — default, không cần API
       Gộp kết quả từ nhiều ranker bằng công thức:
           RRF(d) = Σ 1 / (k + rank_r(d))
       Paper: Cormack et al. 2009 — k=60 là giá trị kinh nghiệm tốt nhất.

    2. Cross-encoder via Jina API — nếu có JINA_API_KEY
       Rerank chính xác hơn nhưng cần API call.
       Model: jina-reranker-v2-base-multilingual (hỗ trợ tiếng Việt)

Default rerank() dùng RRF vì không cần API key.
"""

import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60: smoothing constant từ paper Cormack et al. 2009.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Cross-encoder — Jina Reranker API
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker API (cross-encoder multilingual).
    Fallback về score-based sorting nếu không có API key.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not JINA_API_KEY:
        # Fallback: sort theo score ban đầu
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]

    import requests

    docs = [c["content"] for c in candidates]
    try:
        resp = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": docs,
                "top_n": top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        reranked = resp.json()["results"]

        return [
            {**candidates[r["index"]], "score": round(r["relevance_score"], 4)}
            for r in reranked
        ]
    except Exception as e:
        print(f"  ⚠ Jina API error: {e} — fallback to score-based sort")
        sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_candidates[:top_k]


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR(d) = λ * sim(query, d) - (1-λ) * max(sim(d, selected))
    λ=0.7: ưu tiên relevance hơn diversity (0.5 = cân bằng, 1.0 = chỉ relevance)

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float,
                              'embedding': list[float], 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance(1.0) vs diversity(0.0)
    """
    import numpy as np

    def cosine_sim(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if emb is None:
                relevance = candidates[idx].get("score", 0.0)
            else:
                relevance = cosine_sim(query_embedding, emb)

            max_sim = 0.0
            for sel in selected_indices:
                sel_emb = candidates[sel].get("embedding")
                if sel_emb and emb:
                    max_sim = max(max_sim, cosine_sim(emb, sel_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [
        {**candidates[i], "score": round(candidates[i].get("score", 0.0), 4)}
        for i in selected_indices
    ]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "rrf" | "cross_encoder" | "mmr"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "rrf":
        # RRF với 1 ranked list = sort lại theo score (đơn giản)
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr([], candidates, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
        {"content": "Python programming tutorial", "score": 0.3, "metadata": {}},
    ]

    print("=== RRF Reranking ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="rrf")
    for r in results:
        print(f"  [{r['score']:.6f}] {r['content']}")

    print("\n=== Cross-encoder (Jina API) ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="cross_encoder")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content']}")
