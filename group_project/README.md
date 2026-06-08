# Bai Tap Nhom — Drug Law RAG Chatbot + Evaluation

## Kien Truc He Thong

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG CHATBOT PIPELINE                        │
│                                                                  │
│   User Query (Streamlit UI)                                      │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────┐    ┌───────────────┐                            │
│   │ Semantic  │    │ BM25 Lexical  │                            │
│   │  Search   │    │    Search     │                            │
│   │(ChromaDB) │    │ (rank-bm25)   │                            │
│   └─────┬─────┘    └──────┬────────┘                            │
│         │                 │                                      │
│         └────────┬────────┘                                      │
│                  ▼                                               │
│          RRF Reranking (k=60)                                    │
│                  │                                               │
│          score < 0.005?──→ PageIndex Fallback                   │
│                  │                                               │
│          Context Reorder (lost-in-middle mitigation)            │
│                  │                                               │
│         GPT-4o-mini / OpenRouter                                 │
│                  │                                               │
│          Answer + Citations ──→ Streamlit UI                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                                  │
│                                                                  │
│  Legal DOCX (5 files)  ──→  MarkItDown ──→  Markdown             │
│  News Articles (5 URLs) ──→  Crawl4AI  ──→  JSON ──→ Markdown    │
│                                │                                 │
│                    RecursiveCharacterTextSplitter                │
│                    chunk_size=800, overlap=100                   │
│                                │                                 │
│                           BAAI/bge-m3                            │
│                    1232 chunks, 1024 dim                         │
│                                │                                 │
│                         ChromaDB                                 │
│                    (persistent, cosine distance)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Cai Dat

```bash
# Cai dependencies
pip install -r requirements.txt
pip install streamlit ragas langchain-openai datasets

# Tao file .env
cp .env.example .env
# Them OPENROUTER_API_KEY vao .env

# Chay chatbot
streamlit run app.py

# Chay evaluation pipeline
python -m group_project.evaluation.eval_pipeline
```

## Stack Cong Nghe

| Component | Cong cu | Mo ta |
|-----------|---------|-------|
| Data collection | Crawl4AI + MarkItDown | Crawl bai bao + convert DOCX |
| Chunking | langchain RecursiveCharacterTextSplitter | chunk=800, overlap=100 |
| Embedding | BAAI/bge-m3 (GPU) | Multilingual, 1024 dim |
| Vector Store | ChromaDB | Local persistent, cosine |
| Lexical Search | rank-bm25 (BM25Okapi) | Tieng Viet tokenization |
| Reranking | RRF (Reciprocal Rank Fusion) | Merge dense + sparse |
| Fallback | PageIndex vectorless | Khi score < 0.005 |
| LLM | GPT-4o-mini via OpenRouter | temperature=0.3, top_p=0.9 |
| UI | Streamlit | Chat, sources, conversation memory |
| Evaluation | RAGAS | 4 metrics, A/B comparison |

## Ket Qua Evaluation (RAGAS)

| Metric | Config A (Hybrid+RRF) | Config B (Dense-only) |
|--------|----------------------|----------------------|
| Faithfulness | 0.742 | 0.611 |
| Answer Relevancy | 0.821 | 0.784 |
| Context Recall | 0.688 | 0.593 |
| Context Precision | 0.756 | 0.672 |
| **Average** | **0.752** | **0.665** |

**Ket luan:** Config A (Hybrid + RRF) tot hon +8.7% trung binh.
Xem chi tiet: [group_project/evaluation/results.md](evaluation/results.md)

## Cau Truc Thu Muc

```
Day08_RAG_pipeline_cohort2/
├── app.py                         ← Streamlit chatbot (chay: streamlit run app.py)
├── src/
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py  ← ChromaDB + BAAI/bge-m3
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py     ← BM25
│   ├── task7_reranking.py          ← RRF + Cross-encoder
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py ← Hybrid pipeline
│   └── task10_generation.py        ← GPT-4o-mini + citation
├── group_project/
│   ├── README.md                  ← File nay
│   └── evaluation/
│       ├── golden_dataset.json    ← 16 Q&A pairs
│       ├── eval_pipeline.py       ← RAGAS evaluation
│       └── results.md             ← Ket qua + phan tich
└── chroma_db/                     ← ChromaDB data (1232 chunks)
```

## Phan Cong Cong Viec

| Thanh vien | MSSV | Nhiem vu | Trang thai |
|-----------|------|----------|------------|
| (Ten thanh vien 1) | | Task 1-3: Data collection + convert | Done |
| (Ten thanh vien 2) | | Task 4-5: Chunking + Semantic search | Done |
| (Ten thanh vien 3) | | Task 6-7: Lexical + Reranking | Done |
| (Ten thanh vien 4) | | Task 8-10: PageIndex + Pipeline + Gen | Done |
| Toan nhom | | Chatbot UI + Evaluation pipeline | Done |

## Huong Dan Chay Demo

```bash
# 1. Cai packages
pip install -r requirements.txt
pip install streamlit ragas langchain-openai datasets

# 2. Setup .env
echo "OPENROUTER_API_KEY=your_key" > .env

# 3. Chay indexing (neu chua co chroma_db/)
python -m src.task4_chunking_indexing

# 4. Chay chatbot
streamlit run app.py
# Mo trinh duyet: http://localhost:8501

# 5. Chay evaluation
python -m group_project.evaluation.eval_pipeline
```

## Lu y

- ChromaDB data duoc luu trong `chroma_db/` — khong can chay lai task4 neu da co
- OPENROUTER_API_KEY can duoc set trong `.env` de chatbot va eval hoat dong
- PageIndex API key tuy chon — chatbot van hoat dong neu khong co
- JINA_API_KEY tuy chon — RRF duoc dung mac dinh thay the
