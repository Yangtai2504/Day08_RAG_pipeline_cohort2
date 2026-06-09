"""
RAG Chatbot — Drug Law Vietnam
Streamlit app voi conversation memory, citation va source display.

Chay:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Them project root vao path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("⚖️ Drug Law RAG")
    st.caption("Tra loi cau hoi ve phap luat ma tuy Viet Nam & tin tuc nghe si")

    st.divider()

    st.subheader("Cau hoi goi y")
    suggestions = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy?",
        "Nghe si nao bi bat vi lien quan ma tuy?",
        "Cai nghien bat buoc theo luat phong chong ma tuy?",
        "Cong Tri bi bat vi toi gi?",
        "Quy trinh xac dinh tinh trang nghien ma tuy?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("Thiet lap")
    top_k = st.slider("So chunks retrieval", 3, 10, 5)

    st.divider()
    st.caption("**Kien truc (Supervisor + Workers):**")
    st.caption("Supervisor → [Semantic ‖ BM25 ‖ TF-IDF] parallel → RRF → Rerank → GPT-4o-mini")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("⚖️ Drug Law RAG Chatbot")
st.caption("He thong hoi dap ve phap luat ma tuy Viet Nam va tin tuc nghe si lien quan")

# Hien thi lich su chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📚 Nguon tham khao ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                    st.text(src["content"][:300] + "..." if len(src["content"]) > 300 else src["content"])
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load RAG pipeline mot lan (cache)."""
    from src.supervisor import retrieve_with_supervisor
    from src.task10_generation import generate_with_citation
    return retrieve_with_supervisor, generate_with_citation


def process_query(query: str, top_k: int = 5) -> dict:
    """Chay RAG pipeline qua Supervisor + Workers."""
    retrieve_fn, generate = load_pipeline()
    from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, LLM_MODEL, TEMPERATURE, TOP_P
    import os
    from openai import OpenAI

    chunks = retrieve_fn(query, top_k=top_k, use_hyde=False)
    if not chunks:
        return {"answer": "Toi khong the xac minh thong tin nay tu nguon hien co.", "sources": [], "retrieval_source": "supervisor"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCau hoi: {query}"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    answer = "Toi khong the xac minh thong tin nay tu nguon hien co."
    if api_key:
        try:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
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
        except Exception as e:
            answer = f"Loi LLM: {e}"

    return {"answer": answer, "sources": reordered, "retrieval_source": "supervisor"}


def build_conversation_context() -> str:
    """Tao context tu lich su chat (conversation memory)."""
    history = st.session_state.messages[-6:]  # Lay 3 turns gan nhat
    if not history:
        return ""
    ctx = "\n".join(
        f"{'Nguoi dung' if m['role'] == 'user' else 'Tro ly'}: {m['content'][:200]}"
        for m in history
        if m["role"] in ("user", "assistant")
    )
    return f"\n\n[Lich su cuoc tro chuyen]\n{ctx}\n"


# Xu ly pending query tu sidebar suggestions
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Nhap cau hoi cua ban...")

if user_input:
    # Them conversation context vao query neu co lich su
    history_ctx = build_conversation_context()
    full_query = user_input
    if history_ctx and len(st.session_state.messages) > 0:
        full_query = user_input + history_ctx

    # Hien thi tin nhan nguoi dung
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Chay RAG va hien thi ket qua
    with st.chat_message("assistant"):
        with st.spinner("Dang tim kiem va tong hop..."):
            try:
                result = process_query(full_query, top_k=top_k)
                answer = result.get("answer", "Toi khong the tra loi cau hoi nay.")
                sources = result.get("sources", [])
            except Exception as e:
                answer = f"Loi he thong: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander(f"📚 Nguon tham khao ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    icon = "⚖️" if doc_type == "legal" else "📰"
                    st.markdown(f"**{icon} [{i}] {source_name}** | Loai: `{doc_type}` | Score: `{score:.4f}`")
                    st.text(src["content"][:300] + "..." if len(src["content"]) > 300 else src["content"])
                    if i < len(sources):
                        st.divider()

    # Luu vao session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🗑️ Xoa lich su chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.caption(f"💬 {len(st.session_state.messages)//2} turns trong phien nay")
with col3:
    st.caption("Powered by BAAI/bge-m3 + ChromaDB + GPT-4o-mini")
