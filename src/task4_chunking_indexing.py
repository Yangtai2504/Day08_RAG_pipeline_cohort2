"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB).

Lựa chọn:
    Chunking : RecursiveCharacterTextSplitter
               — chunk_size=800 đủ giữ nguyên điều khoản pháp luật dài
               — overlap=100 không mất context ở ranh giới chunk

    Embedding: BAAI/bge-m3
               — Multilingual, tốt cho tiếng Việt, 1024 dim
               — Tự động dùng GPU nếu có CUDA

    Vector    : ChromaDB (local persistent, không cần Docker)
    Store       — Lưu tại thư mục chroma_db/ trong project

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"

# =============================================================================
# SINGLETONS
# =============================================================================

_embedding_model = None
_chroma_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        print(f"  Embedding device: {device} ({gpu_name})")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _embedding_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection():
    import chromadb
    client = get_chroma_client()
    # Dùng cosine distance cho similarity search
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        parent = md_file.parent.name
        doc_type = "legal" if parent == "legal" else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng BAAI/bge-m3 (GPU nếu có)."""
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks voi {EMBEDDING_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB (upsert theo batch)."""
    if not chunks:
        print("  Khong co chunks de index.")
        return

    collection = get_collection()

    ids       = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings= [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
    print(f"  Indexed {len(chunks)} chunks vao ChromaDB")


def run_pipeline():
    """Chay toan bo pipeline: load -> chunk -> embed -> index."""
    print("=" * 55)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 55)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")
    if not docs:
        print("Khong co documents. Hay chay Task 1-3 truoc!")
        return

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)

    print("\nIndexing vao ChromaDB...")
    index_to_vectorstore(chunks)

    collection = get_collection()
    print(f"\nCollection '{COLLECTION_NAME}' co {collection.count()} chunks")
    print(f"Luu tai: {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
