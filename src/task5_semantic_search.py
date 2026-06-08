"""
Task 5 — Semantic Search dùng ChromaDB.

Query collection đã được index ở Task 4 bằng vector similarity (cosine).
Bao gồm HyDE (Hypothetical Document Embeddings) để cải thiện recall.
"""

import os
from .task4_chunking_indexing import get_collection, get_embedding_model


def _generate_hypothetical_doc(query: str) -> str:
    """
    HyDE: Dùng LLM sinh ra một đoạn văn bản giả thuyết trả lời query,
    sau đó embed đoạn văn đó thay vì embed query trực tiếp.
    Lý do: hypothetical doc nằm trong 'document space', match tốt hơn với
    các chunks đã được index so với query ngắn gọn.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return query  # fallback về query gốc nếu không có API key

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia pháp luật Việt Nam. "
                        "Viết một đoạn văn bản ngắn (~80 từ) như thể đây là nội dung "
                        "từ văn bản pháp luật hoặc bài báo trả lời câu hỏi sau. "
                        "Chỉ viết nội dung, không giải thích."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query  # fallback về query gốc nếu LLM lỗi


def hyde_search(query: str, top_k: int = 10) -> list[dict]:
    """
    HyDE Search: Hypothetical Document Embeddings.

    Thay vì embed query trực tiếp, sinh ra một hypothetical document
    trả lời query rồi embed document đó. Cải thiện recall ~10-15%
    so với standard semantic search với queries ngắn.

    Args:
        query: Câu truy vấn tự nhiên
        top_k: Số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    hypothetical_doc = _generate_hypothetical_doc(query)
    collection = get_collection()

    if collection.count() == 0:
        return []

    model = get_embedding_model()
    # Embed hypothetical doc thay vì query gốc
    hypo_vector = model.encode(hypothetical_doc).tolist()

    results = collection.query(
        query_embeddings=[hypo_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = max(0.0, 1.0 - dist)
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng cosine similarity trong ChromaDB.

    Args:
        query: Câu truy vấn tự nhiên
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,   # cosine similarity (0–1, càng cao càng tốt)
            'metadata': dict
        }
        Sorted by score descending.
    """
    collection = get_collection()

    if collection.count() == 0:
        print("  ChromaDB collection rong - hay chay Task 4 truoc!")
        return []

    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine: distance = 1 - similarity → similarity = 1 - distance
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta,
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    queries = [
        "hinh phat cho toi tang tru trai phep chat ma tuy",
        "nghe si bi bat vi su dung ma tuy",
        "cai nghien bat buoc theo luat phong chong ma tuy",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = semantic_search(q, top_k=3)
        if not results:
            print("  (Khong co ket qua - hay chay Task 4 truoc)")
        for r in results:
            snippet = r["content"][:100].replace("\n", " ")
            print(f"  [{r['score']:.3f}] [{r['metadata'].get('type', '?')}] {snippet}...")
