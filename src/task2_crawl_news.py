"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.

Sử dụng Crawl4AI để crawl nội dung bài báo và lưu thành JSON.
Mỗi file JSON chứa: url, title, date_crawled, content_markdown.

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # chạy 1 lần sau khi cài để setup playwright

Hướng dẫn thêm URL:
    Tìm bài báo trên VnExpress / Tuổi Trẻ / Thanh Niên về chủ đề:
    "nghệ sĩ ma tuý", "ca sĩ bị bắt ma tuý", "diễn viên chất cấm"
    Dán URL vào danh sách ARTICLE_URLS bên dưới.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DANH SÁCH URL BÀI BÁO CẦN CRAWL (tối thiểu 5)
# Thêm URL thực từ VnExpress, Tuổi Trẻ, Thanh Niên, Zing News, ...
# =============================================================================

ARTICLE_URLS: list[str] = [
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://znews.vn/nhung-dien-vien-ca-si-viet-vuong-lao-ly-vi-ma-tuy-post1327508.html",
    "https://congly.vn/nhung-nghe-si-bi-ma-tuy-tan-pha-su-nghiep-khi-dang-o-dinh-cao-208883.html",
    "https://danviet.vn/truoc-cong-tri-da-co-loat-sao-viet-lanh-an-tu-tieu-tan-su-nghiep-vi-ma-tuy-d1350035.html",
    "https://suckhoedoisong.vn/hai-nghe-si-noi-tieng-ten-tin-deu-dinh-vao-ma-tuy-16922061309285984.htm"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(
        word_count_threshold=50,
        remove_overlay_elements=True,
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)

    title = (result.metadata or {}).get("title", "Unknown Title")
    # crawl4ai >= 0.4: result.markdown là MarkdownGenerationResult object
    md = result.markdown
    if md is None:
        content = ""
    elif hasattr(md, "fit_markdown"):
        content = md.fit_markdown or md.raw_markdown or ""
    else:
        content = str(md)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content or "",
    }


def save_article(article: dict, index: int) -> Path:
    """Lưu article dict thành file JSON."""
    filename = f"article_{index:02d}.json"
    filepath = DATA_DIR / filename
    filepath.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filepath


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    if not ARTICLE_URLS:
        print("⚠ ARTICLE_URLS còn trống. Hãy thêm URL bài báo vào trước khi chạy!")
        return

    print(f"Crawling {len(ARTICLE_URLS)} bài báo...\n")

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url[:80]}...")
        try:
            article = await crawl_article(url)
            filepath = save_article(article, i)
            word_count = len(article["content_markdown"].split())
            print(f"  ✓ Saved: {filepath.name} | Title: {article['title'][:50]} | ~{word_count} words")
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            # Lưu placeholder để không bị mất vị trí index
            save_article(
                {"url": url, "title": "CRAWL_FAILED", "date_crawled": datetime.now().isoformat(),
                 "content_markdown": "", "error": str(e)},
                i,
            )

    print(f"\n✓ Hoàn thành! Files lưu tại: {DATA_DIR}")


def list_crawled_files() -> list[Path]:
    """Liệt kê các file bài báo đã crawl."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.json"))


if __name__ == "__main__":
    files = list_crawled_files()
    print(f"Hiện có {len(files)} file trong {DATA_DIR}")

    if not ARTICLE_URLS:
        print("""
⚠ Hãy thêm URL bài báo vào ARTICLE_URLS trước khi chạy!

Gợi ý tìm kiếm:
  - Google: "nghệ sĩ Việt Nam bị bắt ma tuý site:vnexpress.net"
  - Google: "ca sĩ diễn viên ma tuý site:tuoitre.vn"
  - Google: "châu việt cường ma tuý"
  - Google: "phú lê ma tuý"
  - Keyword: "kiều minh tuấn", "dương minh tuấn", ...
        """)
    else:
        asyncio.run(crawl_all())
