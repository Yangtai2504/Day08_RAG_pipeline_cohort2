"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback.

Logic:
    Query
      ├→ Semantic Search (Task 5)  ──┐
      │                               ├→ RRF Merge → Rerank → Results
      ├→ Lexical Search (Task 6)  ──┘
      │
      └→ Nếu best score < threshold
            └→ Fallback: PageIndex Vectorless (Task 8)
"""

from .task5_semantic_search import semantic_search, hyde_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng score tối thiểu cho RRF — RRF max score ≈ 1/(60+1) ≈ 0.016
# Chỉ fallback khi không tìm được kết quả nào (score cực thấp)
SCORE_THRESHOLD = 0.005
DEFAULT_TOP_K = 5
# Dùng RRF (không cần API) để merge và rerank
RERANK_METHOD = "rrf"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_hyde: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results
          ├→ Lexical Search  → sparse_results
          │
          ├→ RRF Merge → merged_results
          ├→ Rerank (RRF hoặc cross-encoder) → reranked_results
          │
          └→ If best_score < threshold → PageIndex fallback

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng score để quyết định fallback
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str   # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy semantic + lexical song song (fetch nhiều hơn để có candidates)
    fetch_k = top_k * 3

    # HyDE: dùng hypothetical document embedding thay vì embed query trực tiếp
    dense_results = hyde_search(query, top_k=fetch_k) if use_hyde else semantic_search(query, top_k=fetch_k)
    sparse_results = lexical_search(query, top_k=fetch_k)

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)

    # Đánh dấu nguồn retrieval
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback sang PageIndex nếu không đủ tốt
    best_score = final_results[0]["score"] if final_results else 0.0

    if best_score < score_threshold:
        print(f"  ⚠ Hybrid best score ({best_score:.3f}) < threshold ({score_threshold})")
        print(f"  → Fallback: PageIndex vectorless search")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  ✗ PageIndex fallback failed: {e}")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì liên quan tới ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print("=" * 70)
        results = retrieve(q, top_k=3)
        if not results:
            print("  Không có kết quả.")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] "
                  f"{r['content'][:80].replace(chr(10), ' ')}...")
