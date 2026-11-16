"""Read and normalize source data. TODO: implement functions."""
import unicodedata
import re
import pypdf
import pdfplumber   
from pathlib import Path
from .models import DocumentPage

# Bắt chính xác cấu trúc mục lục của FPT 2025: "01 THÔNG ĐIỆP BAN LÃNH ĐẠO"
FPT_SECTION_RE = re.compile(r'^(0[1-8])[ \t]+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ\s]+)', re.MULTILINE)

def get_page_to_section_map(pdf_path: Path) -> dict:
    """Đọc Sidebar (Outlines/Bookmarks) của PDF và tạo bản đồ trang"""
    page_map = {}
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            outlines = reader.outline
            
            if not outlines:
                return page_map
                
            for item in outlines:
                if isinstance(item, pypdf.generic.Destination):
                    title = item.title
                    page_num = reader.get_page_number(item.page) + 1 
                    page_map[page_num] = title
    except Exception as e:
        print(f"Lỗi khi đọc Bookmark: {e}")
        
    return page_map

def table_to_markdown(table_data):
    """Chuyển mảng 2 chiều của bảng thành chuỗi Markdown"""
    if not table_data or not table_data[0]:
        return ""
        
    md_string = "\n\n**[BẢNG DỮ LIỆU]**\n"
    # Lọc bỏ None và nối chuỗi cho Header
    md_string += "| " + " | ".join(str(item).replace('\n', ' ') if item else "" for item in table_data[0]) + " |\n"
    md_string += "|---" * len(table_data[0]) + "|\n"
    # Các dòng dữ liệu
    for row in table_data[1:]:
        md_string += "| " + " | ".join(str(item).replace('\n', ' ') if item else "" for item in row) + " |\n"
        
    return md_string + "\n"

def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    pages = []
    
    # 1. Trích xuất bản đồ Sidebar trước (Ưu tiên 1)
    sidebar_map = get_page_to_section_map(path)
    current_section = "Tổng quan"
    
    with pdfplumber.open(path) as pdf:
        for page_num, page_obj in enumerate(pdf.pages, start=1):
            # 2. Ưu tiên cập nhật Section từ Sidebar nếu có
            if page_num in sidebar_map:
                current_section = sidebar_map[page_num]
                
            raw_text = page_obj.extract_text()
            text = raw_text if raw_text else ""
            
            # Trích xuất bảng
            tables = page_obj.extract_tables()
            for table in tables:
                text += table_to_markdown(table)
                
            # 3. Phương án dự phòng (Fallback): Dùng Regex quét trực tiếp text trên trang
            section_matches = FPT_SECTION_RE.findall(text)
            if section_matches:
                # Lấy match cuối cùng trên trang, ví dụ: ('01', 'THÔNG ĐIỆP BAN LÃNH ĐẠO')
                current_section = f"{section_matches[-1][0]} {section_matches[-1][1].strip()}"
                
            pages.append(DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=normalize_text(text),
                metadata={"section": current_section}
            ))
            
    return pages

def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    if not text:
        return ""
        
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()