import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.services.embedder import embed_question
from app.services.retriever import retrieve_top_chunks

def classify(question: str) -> str:
    """
    Rule-based intent classifier:
    - Nếu chứa từ khoá nội bộ → trả 'internal'
    - Ngược lại → 'external'
    """
     # Sử dụng cả ML model và keyword-based detection
    ml_intent = predict_intent(question)
    keyword_intent = detect_question_type(question)
    
    print(f"[ML INTENT]: {ml_intent}")
    print(f"[KEYWORD INTENT]: {keyword_intent}")
    
    # Quyết định intent cuối cùng
    if keyword_intent == "hybrid":
        final_intent = "hybrid"
    elif keyword_intent == ml_intent:
        final_intent = keyword_intent
    elif keyword_intent == "internal" and ml_intent == "external":
        final_intent = "internal"
    elif keyword_intent == "external" and ml_intent == "internal":
        final_intent = "external"
    else:
        final_intent = ml_intent
    return final_intent


# --- ĐỌC DỮ LIỆU Ý ĐỊNH ---
with open("./data_train_in_ex/internal.txt", "r", encoding="utf-8") as f:
    internal_data = [(line.strip(), "internal") for line in f.readlines() if line.strip()]

with open("./data_train_in_ex/external.txt", "r", encoding="utf-8") as f:
    external_data = [(line.strip(), "external") for line in f.readlines() if line.strip()]

data = internal_data + external_data
df = pd.DataFrame(data, columns=["text", "label"])

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])
pipeline.fit(df["text"], df["label"])


#dựa đoán bằng mô hình lr
def predict_intent(question):
    return pipeline.predict([question])[0]

def detect_question_type(user_input):
    user_input_lower = user_input.lower()
    
    # Từ khóa nội bộ với trọng số (từ tài liệu giới thiệu + mô tả sản phẩm)
    internal_keywords = {
        # Sản phẩm & hệ thống chính – trọng số cao
        "lms360": 4, "qlth": 4, "bách khoa": 4, "kiểm định chất lượng": 4, "thi đua khen thưởng": 4,
        "hệ sinh thái": 3, "sản phẩm": 3, "truyền thông nội bộ": 3, "học liệu số": 3,
        "chữ ký số": 3, "kho học liệu": 3, "phòng thí nghiệm mô phỏng": 3, "hệ thống kiểm định": 3,
        "thi đua": 3, "khen thưởng": 3, "tự đánh giá": 3, "chuyển đổi số": 3,
        "đào tạo nhân lực số": 3, "học bạ số": 3, "giải pháp số hóa": 3,
        "ngân hàng học liệu": 3, "trí tuệ nhân tạo": 3, "giám sát học tập": 3,
        "phân tích kết quả học tập": 3, "chuyển đổi phương pháp giảng dạy": 3,

        # Tính năng sản phẩm – trọng số trung bình
        "điểm danh thông minh": 2, "phụ huynh": 2, "giáo viên": 2, "học sinh": 2,"hệ thống":2,
        "bài giảng số": 2, "ai bk": 2, "chatbot ai": 2, "chấm điểm tự động": 2,
        "kiểm tra đánh giá": 2, "trộn đề": 2, "camera ai": 2, "điểm số": 2,
        "quản lý lớp học": 2, "thống kê báo cáo": 2, "ứng dụng ai": 2, "phân tích học tập": 2,
        "chatbot bk": 2, "thí nghiệm mô phỏng": 2, "chấm điểm ai": 2, "soạn giảng tự động": 2,
        "trình chiếu bài giảng": 2, "gợi ý học tập": 2, "nhắc nhở học tập": 2, "đăng nhập vnedi": 2,
        "báo cáo học tập": 2, "tự luận": 2, "trắc nghiệm": 2, "điền khuyết": 2,
        "giao tiếp hai chiều": 2, "thời khóa biểu tự động": 2, "chữ ký số học bạ": 2,
        "y tế học đường": 2, "bmi": 2, "trực tuyến kết hợp trực tiếp": 2,
        "phê duyệt học bạ": 2, "gửi học bạ điện tử": 2, "thi thử thpt": 2,"học tập":2 ,"bài giảng":2,

        # Thuộc tính thương hiệu / tổ chức – trọng số thấp
        "tập đoàn": 1, "công ty": 1, "giới thiệu": 1, "quốc thắng": 1, "tầm nhìn": 1, "sứ mệnh": 1,
        "ar": 1, "vr": 1, "blockchain": 1, "huỳnh quốc thắng": 1, "bền vững": 1,
        "hợp tác quốc tế": 1, "tư duy phản biện": 1, "giải quyết vấn đề": 1,
        "an sinh tinh thần": 1, "đổi mới sáng tạo": 1, "mô hình blended learning": 1,
        "giáo dục cá nhân hóa": 1, "ứng dụng công nghệ": 1
    }


    # Từ khóa external giữ nguyên (hoặc bổ sung sau nếu cần)
    external_keywords = {
        # Comparison indicators – high weight
        "so sánh": 3, "khác biệt": 3, "tương tự": 3, "khác gì": 3,
        "giống": 2, "khác nhau": 2,

        # Ownership/external info – high weight
        "thuộc về": 3, "sở hữu": 3, "của ai": 3,
        "google": 2, "microsoft": 2, "moodle": 2,
        "sở giáo dục": 2, "phòng giáo dục": 2, "bộ gd&đt": 2,

        # Academic subjects – medium weight
        "toán": 2, "phương trình": 2, "đạo hàm": 2, "tích phân": 2,
        "xác suất": 2, "đại số": 2, "hình học": 2, "logarit": 2,
        "vật lý": 2, "hóa học": 2, "sinh học": 2, "di truyền": 2,
        "cơ học": 2, "nhiệt học": 2, "quang học": 2, "lực": 2,
        "giáo dục địa phương": 2, "ngữ văn": 2, "tiếng anh": 2,

        # Education-related concepts – medium weight
        "trí tuệ nhân tạo là gì": 2, "blockchain": 2, "công nghệ thông tin": 2,
        "ngôn ngữ lập trình": 2, "python": 2, "chatgpt": 2, "machine learning": 2,
        "học máy": 2, "deep learning": 2, "neural network": 2,
        "blended learning": 2, "e-learning": 2, "bài giảng trực tuyến": 2,
        "ứng dụng ai trong giáo dục": 2,"trực quan":2,

        # General knowledge – low weight
        "thời tiết": 1, "quốc gia": 1, "tổng thống": 1, "thế giới": 1,
        "lịch sử": 1, "định nghĩa": 1, "giải thích": 1, "pt": 1,
        "bậc": 1, "căn": 1, "tính": 1, "ứng dụng": 1,
        "kiến thức chung": 1, "giải toán": 1, "cách giải": 1,
        "giới hạn": 1, "vector": 1, "ma trận": 1, "thống kê": 1,
        "đồ thị": 1, "hàm số": 1, "chu vi": 1, "diện tích": 1,
        "năng lượng": 1, "động năng": 1, "thế năng": 1, "phản ứng": 1,
        "phân tử": 1, "nguyên tử": 1
    }


    # Tính điểm trọng số
    internal_score = sum(weight for keyword, weight in internal_keywords.items() 
                         if keyword in user_input_lower)
    external_score = sum(weight for keyword, weight in external_keywords.items() 
                         if keyword in user_input_lower)

    print(f"🔍 Phân tích: Internal_score={internal_score}, External_score={external_score}")
    
    # Logic quyết định ý định
    if internal_score >= 2 and external_score >= 2:
        return "hybrid"
    elif internal_score >= 1 and external_score >= 3:
        return "hybrid"
    elif internal_score > external_score:
        return "internal"
    elif external_score > 0:
        return "external"
    else:
        return ""
    

def check_question_followup(question):
    follow_up_indicators = [
        # Đại từ chỉ định
        "nó", "này", "đó", "chúng", "cái này", "cái đó", "việc này", "việc đó",
        "trường hợp này", "trường hợp đó", "hệ thống này", "hệ thống đó", "họ", "sản phẩm này", "ứng dụng này",

        # Hỏi về tính năng/đặc điểm/chi tiết
        "làm gì", "dùng để làm gì", "có tác dụng gì", "dùng như thế nào", "có chức năng gì",
        "tính năng", "chức năng", "hoạt động", "cách vận hành", "cách dùng",
        "đặc điểm", "cấu trúc", "cơ chế", "ưu điểm", "nhược điểm",
        "áp dụng như thế nào", "triển khai thế nào", "vận hành thế nào", "tích hợp ra sao",

        # Hỏi chi tiết, mở rộng
        "giải thích thêm", "giải thích rõ", "nói rõ hơn", "chi tiết hơn", "cụ thể hơn",
        "ví dụ", "minh họa", "mô phỏng", "cho ví dụ", "demo", "mô tả thêm",
        "trường hợp cụ thể", "thực tế triển khai", "ứng dụng thực tế",

        # So sánh
        "khác gì", "giống gì", "so với", "tương tự", "khác nhau chỗ nào", "giống nhau chỗ nào",
        "điểm khác biệt", "điểm tương đồng", "có gì khác", "có gì giống",

        # Gợi ý follow-up theo ngữ cảnh liên tục
        "tiếp theo thì sao", "trong trường hợp đó", "sau đó", "vậy tiếp theo", "tiếp theo làm gì",
        "kế tiếp là gì", "sau bước đó", "sau bước này", "tiếp đến", "nếu vậy thì sao"
    ]

    has_follow_up = any(indicator in question.lower() for indicator in follow_up_indicators)
    
    return has_follow_up


