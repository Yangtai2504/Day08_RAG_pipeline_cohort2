"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown


def _extract_pdf_pdfplumber(filepath: Path) -> str:
    """Fallback: dùng pdfplumber khi MarkItDown trả về rỗng."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n\n".join(pages)

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Thư mục data/landing/legal/ chưa có — bỏ qua")
        return 0

    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists():
            print(f"  (đã có) {output_path.name}")
            converted += 1
            continue

        print(f"  Converting: {filepath.name} ...")
        try:
            result = md.convert(str(filepath))
            content = result.text_content or ""

            header = f"# {filepath.stem}\n\n"
            header += f"**Nguon file:** {filepath.name}\n"
            header += f"**Loai:** Van ban phap luat\n\n---\n\n"

            output_path.write_text(header + content, encoding="utf-8")
            size = len(content)
            print(f"  OK {output_path.name} ({size:,} chars)")
            converted += 1
        except Exception as e:
            print(f"  [FAIL] {filepath.name}: {e}")

    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Thư mục data/landing/news/ chưa có — bỏ qua")
        return 0

    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        if output_path.exists():
            print(f"  (đã có) {output_path.name}")
            converted += 1
            continue

        print(f"  Converting: {filepath.name} ...")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Bỏ qua file bị lỗi crawl
            if data.get("title") == "CRAWL_FAILED" or not data.get("content_markdown"):
                print(f"  ⚠ Bỏ qua {filepath.name} (crawl failed / rỗng)")
                continue

            # Thêm metadata header rõ ràng để LLM cite được
            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            content = data.get("content_markdown", "")

            header = f"# {title}\n\n"
            header += f"**Nguồn:** {url}\n"
            header += f"**Ngày crawl:** {date_crawled}\n"
            header += f"**Loại:** Bài báo\n\n---\n\n"

            output_path.write_text(header + content, encoding="utf-8")
            size = len(content)
            print(f"  ✓ {output_path.name} ({size:,} chars)")
            converted += 1
        except Exception as e:
            print(f"  [FAIL] Loi khi convert {filepath.name}: {e}")

    return converted


def convert_all():
    """Convert toàn bộ files từ landing sang standardized."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    total = n_legal + n_news
    print(f"\n✓ Đã convert {total} file (legal: {n_legal}, news: {n_news})")
    print(f"  Output tại: {OUTPUT_DIR}")
    return total


if __name__ == "__main__":
    convert_all()
