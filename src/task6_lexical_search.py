"""
Task 6 — Lexical Search Module (BM25).

Sử dụng BM25Okapi từ rank-bm25.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm trong corpus → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d)*(k1+1)) / (tf(qi,d)+k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term frequency saturation), b=0.75 (length normalization)

Khác biệt với semantic search:
    - BM25 tìm theo từ khóa chính xác (keyword matching)
    - Semantic search tìm theo nghĩa (có thể tìm được dù dùng từ đồng nghĩa)
    - Hybrid = kết hợp cả hai → tốt hơn từng loại riêng lẻ

Cài đặt:
    pip install rank-bm25
"""

import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CORPUS LOADING + BM25 INDEX (lazy init)
# =============================================================================

_corpus: list[dict] = []      # List of {'content': str, 'metadata': dict}
_bm25: BM25Okapi | None = None


def _load_corpus() -> list[dict]:
    """Load corpus từ data/standardized/ (cùng source với Task 4)."""
    corpus = []
    if not STANDARDIZED_DIR.exists():
        return corpus

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        parent = md_file.parent.name
        doc_type = "legal" if parent == "legal" else "news"
        corpus.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
            },
        })
    return corpus


def _tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản: lowercase + split theo khoảng trắng.
    Với tiếng Việt, đây là word-level tokenization cơ bản.
    Có thể cải tiến bằng underthesea cho kết quả tốt hơn.
    """
    return text.lower().split()


def _get_bm25() -> tuple[BM25Okapi | None, list[dict]]:
    """Lazy init: build BM25 index lần đầu, cache lại."""
    global _corpus, _bm25

    if _bm25 is not None:
        return _bm25, _corpus

    _corpus = _load_corpus()
    if not _corpus:
        return None, []

    tokenized = [_tokenize(doc["content"]) for doc in _corpus]
    _bm25 = BM25Okapi(tokenized)
    return _bm25, _corpus


# =============================================================================
# PUBLIC API
# =============================================================================

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus bên ngoài (dùng cho testing).

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance
    """
    global _corpus, _bm25
    _corpus = corpus
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    _bm25 = BM25Okapi(tokenized)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (> 0 nếu có keyword match)
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = _get_bm25()

    if bm25 is None or not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices theo score descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break  # BM25 score <= 0 nghĩa là không có keyword match
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": corpus[idx]["metadata"],
        })

    return results


if __name__ == "__main__":
    queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "hình phạt tù năm",
        "nghệ sĩ bị bắt ma tuý",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = lexical_search(q, top_k=3)
        if not results:
            print("  (Không có kết quả — hãy chạy Task 3 trước)")
        for r in results:
            print(f"  [{r['score']:.3f}] [{r['metadata'].get('type', '?')}] "
                  f"{r['content'][:100].replace(chr(10), ' ')}...")
