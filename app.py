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
    st.caption("**Kien truc:**")
    st.caption("Query → Semantic + BM25 → RRF → Rerank → GPT-4o-mini")

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
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation
    return retrieve, generate_with_citation


def process_query(query: str, top_k: int = 5) -> dict:
    """Chay RAG pipeline va tra ve ket qua."""
    _, generate = load_pipeline()
    return generate(query, top_k=top_k)


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
