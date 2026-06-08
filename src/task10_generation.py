"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve chunks (Task 9)
    2. Reorder để tránh "lost in the middle" (Liu et al. 2023)
    3. Format context với source labels để LLM có thể cite
    4. Call LLM (OpenAI) với SYSTEM_PROMPT yêu cầu citation
    5. Return answer + sources

Tham số LLM:
    temperature=0.3 — RAG cần factual, ít sáng tạo
    top_p=0.9       — nucleus sampling: đủ diverse, không quá random
    top_k=5 chunks  — đủ evidence, không quá dài gây lost-in-the-middle
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

# 5 chunks: đủ evidence cho câu hỏi phức tạp mà không vượt quá context window
TOP_K = 5
# temperature=0.3: ưu tiên factual accuracy hơn creativity cho RAG
TEMPERATURE = 0.3
# top_p=0.9: giữ 90% xác suất tích luỹ — cân bằng giữa coherence và diversity
TOP_P = 0.9

LLM_MODEL = "openai/gpt-4o-mini"  # OpenRouter model ID


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về pháp luật ma tuý Việt Nam và tin tức liên quan.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Luật 73/2021, Điều 3] hoặc [VnExpress, 2024]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING — tránh "lost in the middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI, quên thông tin ở GIỮA.
    Strategy: chunk quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input (by score desc):  [1, 2, 3, 4, 5]
    Output:                 [1, 3, 5, 4, 2]
    (chunk #1 đầu, #2 cuối, còn lại ở giữa theo thứ tự giảm dần)

    Ref: Liu et al. 2023 "Lost in the Middle: How Language Models Use Long Contexts"
    """
    if len(chunks) <= 2:
        return chunks

    # Chia thành vị trí đầu (odd index) và cuối (even index)
    front = chunks[::2]      # index 0, 2, 4 → đặt ở đầu
    back = chunks[1::2]      # index 1, 3    → đặt ở cuối (reversed)

    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string với source labels.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")

        label = f"[Document {i} | {source} | {doc_type}]"
        parts.append(f"{label}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Chunks đã dùng (sau reorder)
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build user message
    user_message = f"""Context:
{context}

---

Câu hỏi: {query}"""

    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "none"

    # Step 5: Call LLM
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "answer": "Toi khong the xac minh thong tin nay tu nguon hien co.",
            "sources": reordered,
            "retrieval_source": retrieval_src,
        }

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        answer = response.choices[0].message.content
    except Exception:
        answer = "Toi khong the xac minh thong tin nay tu nguon hien co."

    # Step 6: Return
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_src,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        try:
            result = generate_with_citation(q)
            print(f"\nA: {result['answer']}")
            print(f"\n[{len(result['sources'])} chunks | via {result['retrieval_source']}]")
        except Exception as e:
            print(f"✗ Lỗi: {e}")
            print("  → Đảm bảo OPENAI_API_KEY đã được set trong .env")
