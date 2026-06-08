"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là RAG engine không cần vector store — dùng structural understanding
của document (cấu trúc, heading, context) để tìm kiếm.

SDK: https://github.com/VectifyAI/PageIndex
Flow:
    1. submit_document(file_path) → doc_id (async processing)
    2. submit_query(doc_id, query) → retrieval_id
    3. poll get_retrieval(retrieval_id) until status == 'completed'
    4. Return kết quả với source='pageindex'

doc_ids được cache vào pageindex_doc_ids.json để không phải upload lại.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
TMP_PDF_DIR = Path(__file__).parent.parent / "data" / "_tmp_pdf"
DOC_IDS_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"
POLL_INTERVAL = 2   # seconds between polls
POLL_TIMEOUT  = 60  # max seconds to wait per query


def _get_client():
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError(
            "PAGEINDEX_API_KEY chua duoc set trong .env"
        )
    from pageindex.client import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _md_to_pdf(md_file: Path, out_dir: Path) -> Path:
    """Convert markdown file sang PDF don gian bang fpdf2."""
    from fpdf import FPDF

    content = md_file.read_text(encoding="utf-8")
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)

    page_width = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

    for line in content.split("\n"):
        # Strip non-latin1 chars; truncate chuoi qua dai
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        # Truncate de tranh line vuot qua page width (gioi han ~200 chars)
        if len(safe) > 200:
            safe = safe[:200] + "..."
        if not safe.strip():
            pdf.ln(3)
        else:
            pdf.multi_cell(page_width, 4, text=safe)

    out_path = out_dir / f"{md_file.stem}.pdf"
    pdf.output(str(out_path))
    return out_path


def upload_documents() -> dict:
    """
    Convert markdown files → PDF tam thoi → upload len PageIndex.
    Cache doc_ids vao pageindex_doc_ids.json.

    Returns:
        dict: {filename: doc_id}
    """
    client = _get_client()
    TMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache hien co
    cache = {}
    if DOC_IDS_CACHE.exists():
        cache = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        cache_key = md_file.stem
        if cache_key in cache:
            print(f"  (cached) {md_file.name} -> {cache[cache_key]}")
            continue

        print(f"  Converting & uploading: {md_file.name} ...")
        try:
            pdf_path = _md_to_pdf(md_file, TMP_PDF_DIR)
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            if doc_id:
                cache[cache_key] = doc_id
                print(f"  OK {md_file.name} -> {doc_id}")
            else:
                print(f"  WARN no doc_id: {resp}")
        except Exception as e:
            print(f"  FAIL {md_file.name}: {e}")

    DOC_IDS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def _poll_retrieval(client, retrieval_id: str, timeout: int = POLL_TIMEOUT) -> dict:
    """Poll get_retrieval cho den khi status == completed hoac timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status in ("completed", "done", "success"):
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"Retrieval failed: {result}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Retrieval {retrieval_id} timeout after {timeout}s")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval su dung PageIndex.
    Fallback khi hybrid search khong tra ve ket qua phu hop.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    client = _get_client()

    # Load cached doc_ids
    if not DOC_IDS_CACHE.exists():
        print("  Chua co doc_ids. Chay upload_documents() truoc.")
        upload_documents()

    cache = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
    if not cache:
        raise RuntimeError("Khong co documents tren PageIndex. Hay chay upload_documents() truoc.")

    results = []
    for filename, doc_id in list(cache.items())[:top_k]:
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            retrieval = _poll_retrieval(client, retrieval_id)

            # Extract content from retrieval result
            items = (
                retrieval.get("results")
                or retrieval.get("data")
                or retrieval.get("chunks")
                or []
            )
            for item in items[:2]:
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("passage")
                    or str(item)
                )
                score = float(item.get("score", 0.5))
                results.append({
                    "content": content,
                    "score": round(score, 4),
                    "metadata": {"source": filename, "type": "pageindex"},
                    "source": "pageindex",
                })
        except Exception as e:
            print(f"  Query error for {filename}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY chua set trong .env")
    else:
        print("Uploading documents...")
        cache = upload_documents()
        print(f"Uploaded {len(cache)} documents")

        print("\nTest query: 'hinh phat ma tuy'")
        try:
            results = pageindex_search("hinh phat ma tuy", top_k=3)
            for r in results:
                print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
