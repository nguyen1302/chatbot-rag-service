def classify(question: str) -> str:
    """
    Rule-based intent classifier:
    - Nếu chứa từ khoá nội bộ → trả 'internal'
    - Ngược lại → 'external'
    """
    internal_keywords = [
        "lms360", "sản phẩm", "tính năng", "bách khoa", "giá", "gói", "học sinh", "trường học",
        "giáo viên", "quản lý lớp", "điểm", "lịch sử", "công ty", "bktech"
    ]

    lower_q = question.lower()
    for kw in internal_keywords:
        if kw in lower_q:
            return "internal"

    return "external"
