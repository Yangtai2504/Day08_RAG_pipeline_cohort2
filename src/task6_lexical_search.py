"""
Task 6 — Lexical Search Module (BM25 + TF-IDF).

Implement HAI phương pháp lexical search để so sánh:

─────────────────────────────────────────────────────────────
[1] BM25 (Best Match 25) — dùng rank-bm25
─────────────────────────────────────────────────────────────
  score(q,d) = Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))

  Ưu điểm so với TF-IDF:
  - Term saturation: từ xuất hiện 10 lần không tốt gấp 10 lần từ xuất hiện 1 lần
    (k1 kiểm soát mức bão hoà, mặc định k1=1.5)
  - Length normalization: document dài không được lợi thế bất công
    (b=0.75 kiểm soát mức độ penalty độ dài)
  → BM25 phù hợp cho corpus có document dài ngắn không đều (như pháp luật + báo)

─────────────────────────────────────────────────────────────
[2] TF-IDF — dùng sklearn TfidfVectorizer + cosine similarity
─────────────────────────────────────────────────────────────
  TF(t,d) = số lần t xuất hiện trong d / tổng số từ trong d
  IDF(t)  = log(N / df(t))  [N = số doc, df = số doc chứa term t]
  score   = cosine_similarity(query_vec, doc_vec)

  Hạn chế so với BM25:
  - Không có term saturation: TF tuyến tính → từ lặp nhiều lần được lợi quá mức
  - Không có length normalization thông minh: chỉ dùng L2 norm trong cosine
  → TF-IDF phù hợp cho short documents, BM25 tốt hơn cho long documents

Cài đặt:
    pip install rank-bm25 scikit-learn
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


# =============================================================================
# TF-IDF SEARCH (phương pháp thay thế BM25 — sklearn)
# =============================================================================

_tfidf_vectorizer = None
_tfidf_matrix = None


def _get_tfidf():
    """Lazy init TF-IDF vectorizer."""
    global _tfidf_vectorizer, _tfidf_matrix, _corpus

    if _tfidf_vectorizer is not None:
        return _tfidf_vectorizer, _tfidf_matrix, _corpus

    # Dùng chung corpus với BM25
    _, corpus = _get_bm25()
    if not corpus:
        return None, None, []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [doc["content"] for doc in corpus]
    _tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        # Không dùng stop words vì tiếng Việt không có built-in list
        max_features=50000,
        ngram_range=(1, 2),  # Unigram + bigram → bắt được cụm từ "ma tuý", "cai nghiện"
    )
    _tfidf_matrix = normalize(_tfidf_vectorizer.fit_transform(texts))
    return _tfidf_vectorizer, _tfidf_matrix, corpus


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa bằng TF-IDF + cosine similarity.

    So sánh với BM25:
    - TF-IDF: score tuyến tính theo tần suất từ, không có term saturation
    - BM25: có term saturation (k1) và length normalization (b) → tốt hơn cho
      corpus có độ dài document không đồng đều như dataset pháp luật này

    Args:
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    vectorizer, matrix, corpus = _get_tfidf()
    if vectorizer is None:
        return []

    from sklearn.preprocessing import normalize
    query_vec = normalize(vectorizer.transform([query.lower()]))
    scores = (matrix @ query_vec.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break
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

        bm25_results = lexical_search(q, top_k=3)
        print("  [BM25]")
        if not bm25_results:
            print("    (Không có kết quả — hãy chạy Task 3 trước)")
        for r in bm25_results:
            print(f"    [{r['score']:.3f}] {r['content'][:80].replace(chr(10), ' ')}...")

        tfidf_results = tfidf_search(q, top_k=3)
        print("  [TF-IDF]")
        for r in tfidf_results:
            print(f"    [{r['score']:.3f}] {r['content'][:80].replace(chr(10), ' ')}...")
