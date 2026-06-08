"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tải tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Chạy script này để setup thư mục và tải tự động (nếu có URL).
    3. Hoặc tải thủ công và đặt vào data/landing/legal/

Nguồn tải thủ công:
    - https://thuvienphapluat.vn  (tìm theo tên luật, tải PDF)
    - https://vanban.chinhphu.vn  (văn bản chính phủ)
    - https://luatvietnam.vn

Văn bản cần tải (tối thiểu 3):
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật PCMT
    - Bộ luật Hình sự 2015 (sửa đổi 2017) — Chương XX: Tội phạm về ma tuý
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý và tiền chất
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """
    Tải file từ URL về DATA_DIR.

    Args:
        url: Direct download URL của file
        filename: Tên file lưu trên disk

    Returns:
        Path đến file đã tải
    """
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  (đã có) {filename} — bỏ qua")
        return filepath

    print(f"  Đang tải: {filename} ...")
    resp = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        stream=True,
    )
    resp.raise_for_status()

    filepath.write_bytes(resp.content)
    size_kb = filepath.stat().st_size // 1024
    print(f"  ✓ Đã tải: {filename} ({size_kb} KB)")
    return filepath


def list_downloaded_files() -> list[Path]:
    """Liệt kê các file pháp luật đã có trong DATA_DIR."""
    if not DATA_DIR.exists():
        return []
    valid_exts = {".pdf", ".docx", ".doc"}
    files = [f for f in DATA_DIR.iterdir()
             if f.is_file() and f.suffix.lower() in valid_exts]
    return sorted(files)


# =============================================================================
# DANH SÁCH VĂN BẢN CẦN TẢI
# Điền direct download URL nếu có. Nếu không có URL, tải thủ công.
# =============================================================================

LEGAL_DOCS: list[tuple[str, str]] = [
    # (url, filename) — Thêm vào đây nếu có direct download link
    # Ví dụ:
    # ("https://example.gov.vn/luat-73-2021.pdf", "luat-phong-chong-ma-tuy-2021.pdf"),
]


if __name__ == "__main__":
    setup_directory()

    if LEGAL_DOCS:
        print(f"\nĐang tải {len(LEGAL_DOCS)} văn bản...")
        for url, filename in LEGAL_DOCS:
            try:
                download_file(url, filename)
            except Exception as e:
                print(f"  ✗ Lỗi khi tải {filename}: {e}")
    else:
        print("""
⚠  Chưa có URL để tải tự động.

Hướng dẫn tải THỦ CÔNG (cần tải tối thiểu 3 file):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
   → https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/
     Luu-lai-thanh: luat-phong-chong-ma-tuy-2021.pdf

2. Nghị định 105/2021/NĐ-CP
   → https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/
     Luu-lai-thanh: nghi-dinh-105-2021.pdf

3. Bộ luật Hình sự 2015 (Chương XX — Tội phạm về ma tuý)
   → https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/
     Luu-lai-thanh: bo-luat-hinh-su-2015-chuong-xx.pdf

4. Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
   → https://vanban.chinhphu.vn/
     Luu-lai-thanh: nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf

Thư mục đích: {data_dir}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(data_dir=DATA_DIR))

    files = list_downloaded_files()
    print(f"\n📂 Hiện có {len(files)} file trong {DATA_DIR}:")
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"   - {f.name} ({size_kb} KB)")

    if len(files) < 3:
        print(f"\n⚠  Cần thêm {3 - len(files)} file nữa để đủ yêu cầu tối thiểu!")
    else:
        print(f"\n✓ Đủ {len(files)} file — Task 1 hoàn thành!")
