from markitdown import MarkItDown
import re

# Tạo đối tượng chuyển đổi
md = MarkItDown()

# Đường dẫn đến file Word
input_path = "TÀI LIỆU MÔ TẢ SẢN PHẨM_PHỔ THÔNG (1) (2).docx"

# Chuyển đổi sang Markdown
result = md.convert(input_path)
markdown = result.text_content

# Hàm chuyển đổi tiêu đề in đậm thành heading Markdown
def convert_to_headings(text: str) -> str:
    # Heading cấp 1: **A. LMS360...** → # A. LMS360...
    text = re.sub(r'^\*\*([A-F]\.\s.+?)\*\*$', r'# \1', text, flags=re.MULTILINE)

    # Heading cấp 2: **I. TỔNG QUAN** → ## I. TỔNG QUAN
    text = re.sub(r'^\*\*(I{1,3}|IV|V|VI{0,1})\.\s+(.+?)\*\*$', r'## \1. \2', text, flags=re.MULTILINE)

    # Heading cấp 3: **1. Tên mục nhỏ** → ### 1. Tên mục nhỏ
    text = re.sub(r'^\*\*(\d+)\.\s+(.+?)\*\*$', r'### \1. \2', text, flags=re.MULTILINE)

    return text

# Chuyển tiêu đề in đậm thành markdown heading
cleaned_markdown = convert_to_headings(markdown)

# Ghi ra file .txt (vẫn chứa nội dung Markdown chuẩn)
with open("output_1.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_markdown)

print("✅ Đã chuyển file Word sang Markdown với cấu trúc heading hợp lệ trong output_1.txt")
