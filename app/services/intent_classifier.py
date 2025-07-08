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
    # === SẢN PHẨM & HỆ THỐNG CHÍNH === (Trọng số cao - 4-5)
    "lms360": 5, "qlth": 5, "sms360": 5, "hệ sinh thái": 4, 
    "kiểm định chất lượng": 4, "thi đua khen thưởng": 4, "bách khoa": 4,
    "hệ thống kiểm định": 4, "chuyển đổi số": 4, "giải pháp số hóa": 4,
    
    # === TÍNH NĂNG SẢN PHẨM === (Trọng số trung bình - 3)
    "học liệu số": 3, "chữ ký số": 3, "kho học liệu": 3, "ngân hàng học liệu": 3,
    "phòng thí nghiệm mô phỏng": 3, "thí nghiệm mô phỏng": 3, "học bạ số": 3,
    "điểm danh thông minh": 3, "camera ai": 3, "chatbot ai": 3, "ai bk": 3, "chatbot bk": 3,
    "chấm điểm tự động": 3, "chấm điểm ai": 3, "soạn giảng tự động": 3,
    "trộn đề": 3, "kiểm tra đánh giá": 3, "thống kê báo cáo": 3,
    "phân tích kết quả học tập": 3, "phân tích học tập": 3, "giám sát học tập": 3,
    "gợi ý học tập": 3, "nhắc nhở học tập": 3, "báo cáo học tập": 3,
    "y tế học đường": 3, "bmi": 3, "chữ ký số học bạ": 3, "gửi học bạ điện tử": 3,
    
    # === CHỨC NĂNG HỖ TRỢ === (Trọng số thấp - 2)
    "đào tạo nhân lực số": 2, "tự đánh giá": 2, "truyền thông nội bộ": 2,
    "quản lý lớp học": 2, "ứng dụng ai": 2, "trình chiếu bài giảng": 2,
    "giao tiếp hai chiều": 2, "thời khóa biểu tự động": 2, "đăng nhập vnedi": 2,
    "phê duyệt học bạ": 2, "thi thử thpt": 2, "trực tuyến kết hợp trực tiếp": 2,
    "chuyển đổi phương pháp giảng dạy": 2, "mô hình blended learning": 2,
    "giáo dục cá nhân hóa": 2, "tự luận": 2, "trắc nghiệm": 2, "điền khuyết": 2,
    
    # === THÔNG TIN CÔNG TY === (Trọng số thấp - 1)
    "tập đoàn": 1, "công ty": 1, "quốc thắng": 1, "huỳnh quốc thắng": 1,
    "tầm nhìn": 1, "sứ mệnh": 1, "bền vững": 1, "hợp tác quốc tế": 1,
    "đổi mới sáng tạo": 1, "an sinh tinh thần": 1, "ứng dụng công nghệ": 1,
    
    # === CHƯƠNG TRÌNH GIÁO DỤC 2018 - TIẾNG VIỆT === (Trọng số cao - 4-5)
    "chương trình giáo dục phổ thông 2018": 5,"chương trình phổ thông 2018":5, "chương trình gdpt 2018": 5,"chương trình gdpt2018": 5,
    "gdpt2018":5,"gdpt 2018":5,
    "tiếng việt lớp 1": 4, "tiếng việt lớp 2": 4, "tiếng việt lớp 3": 4, 
    "tiếng việt lớp 4": 4, "tiếng việt lớp 5": 4, "sgk tiếng việt": 4,
    
    # === KỸ NĂNG & PHƯƠNG PHÁP === (Trọng số trung bình - 3)
    "mục tiêu môn học": 3, "yêu cầu cần đạt": 3, "kỹ năng nghe": 3, 
    "kỹ năng nói": 3, "kỹ năng đọc": 3, "kỹ năng viết": 3, "đọc hiểu": 3,
    "chính tả": 3, "viết chữ": 3, "tập viết": 3, "học vần": 3, "âm vần": 3,
    "ngữ âm tiếng việt": 3, "ngữ pháp tiếng việt": 3, "bố cục bài văn": 3,
    "viết bài văn": 3, "viết đoạn văn": 3, "phân tích văn bản": 3,
    "tư duy phản biện": 3, "cảm thụ văn học": 3, "nghị luận đơn giản": 3,
    "thuyết trình": 3, "giao tiếp nhóm": 3, "thảo luận nhóm": 3,
    
    # === ĐÁNH GIÁ & KIỂM TRA === (Trọng số thấp - 2)
    "đánh giá thường xuyên": 2, "đánh giá định kỳ": 2, "đánh giá bằng nhận xét": 2,
    "phiếu nhận xét": 2, "phiếu tự đánh giá": 2, "bài kiểm tra": 2,
    "sách giáo khoa": 2, "tập một": 2, "tập hai": 2, "chân trời sáng tạo": 2,
    "kết nối tri thức": 2, "cánh diều": 2, "vở ô li": 2, "vở luyện viết": 2,
    
    # === HOẠT ĐỘNG HỌC TẬP === (Trọng số thấp - 2)
    "bài đọc hiểu": 2, "bài viết chính tả": 2, "nghe kể chuyện": 2,
    "trả lời câu hỏi": 2, "viết theo mẫu": 2, "ngắt nghỉ": 2, "kể lại chuyện": 2,
    "kể chuyện": 2, "kể chuyện theo tranh": 2, "trình bày cảm xúc": 2, "giọng đọc": 2,
    "phản hồi lời nói": 2, "truyện": 2, "thơ": 2, "thơ lục bát": 2, "truyện kể": 2,
    "văn bản thông tin": 2, "văn bản miêu tả": 2, "văn bản nghị luận": 2,
    "văn bản hướng dẫn": 2, "thuyết minh": 2, "đọc thành tiếng": 2,
    "nêu ý kiến cá nhân": 2, "gạch chân từ khóa": 2, "vẽ sơ đồ tư duy": 2,
    "viết dàn ý": 2, "tóm tắt văn bản": 2, "so sánh liên hệ": 2,
    "biện pháp tu từ": 2, "từ ngữ nghệ thuật": 2, "so sánh nhân hóa": 2,
    "dấu câu": 2, "phép liên kết câu": 2, "ngữ điệu khi đọc": 2,
    "truyện tranh": 2, "truyện ngụ ngôn": 2, "truyện cổ tích": 2, "đoạn văn miêu tả": 2,
    "tranh minh hoạ": 2, "liên hệ tranh với chi tiết": 2, "nội dung văn bản": 2,
    "trật tự sự việc": 2, "trình tự sự kiện": 2, "nghi thức giao tiếp": 2,
    "âm vần thanh": 2, "phân biệt c và k": 2, "phân biệt g và gh": 2, "phân biệt ng và ngh": 2,
    "viết hoa tên riêng": 2, "kể lại câu chuyện": 2,
    
    # === ĐỐI TƯỢNG & KỸ THUẬT === (Trọng số thấp - 1)
    "học sinh": 1, "giáo viên": 1, "phụ huynh": 1, "hệ thống": 1,
    "học tập": 1, "bài giảng": 1, "sản phẩm": 1, "học sinh lớp 1": 1,
    "học sinh lớp 2": 1, "học sinh lớp 3": 1, "học sinh lớp 4": 1, "học sinh lớp 5": 1,
    "giáo viên lớp 1": 1, "giáo viên lớp 2": 1, "giáo viên lớp 3": 1, 
    "giáo viên lớp 4": 1, "giáo viên lớp 5": 1, "trường tiểu học": 1,
    "bài học lớp 1": 1, "bài học lớp 2": 1, "bài học lớp 3": 1, 
    "bài học lớp 4": 1, "bài học lớp 5": 1, "giảng dạy tiếng việt": 1,
    "phát triển năng lực ngôn ngữ": 1, "rèn luyện tiếng việt": 1, "tự học tiếng việt": 1,
    "kỹ năng giao tiếp": 1, "thuyết trình ngắn": 1, "giao tiếp học đường": 1,
    "hoạt động nhóm": 1, "vốn sống học sinh": 1, "giới thiệu môn học": 1,
    "luyện tập tiếng việt": 1, "trình bày cảm nghĩ": 1, "trình bày quan điểm": 1,
    "viết báo cáo": 1, "viết thư": 1, "trình bày logic": 1, "ar": 1, "vr": 1,
    "blockchain": 1, "giải quyết vấn đề": 1, "miêu tả": 1, "giới thiệu": 1,
    "biển báo": 1, "tín hiệu đơn giản": 1, "ngữ điệu đọc": 1, "nhân vật yêu thích": 1,
    "lời nhân vật": 1, "hành động nhân vật": 1, "ngoại hình nhân vật": 1,
    "từ xưng hô": 1, "chào hỏi": 1, "xin lỗi": 1, "cảm ơn": 1, "xin phép": 1,
    "giới thiệu bản thân": 1, "đặt dấu thanh": 1, "viết đúng tư thế": 1,
    "viết chữ hoa": 1, "viết chữ thường": 1, "viết số từ 0 đến 9": 1,
    "tập chép": 1, "nghe viết": 1, "tốc độ viết": 1, "chữ và dấu thanh": 1,
    "vốn từ chủ điểm": 1, "từ chỉ sự vật": 1, "từ chỉ hoạt động": 1, "từ chỉ đặc điểm": 1,
    "nhìn vào người nghe": 1, "đặt câu hỏi": 1, "nghe hiểu thông báo": 1,
    "nghe hiểu hướng dẫn": 1, "nghe hiểu nội quy": 1, "lắng nghe tích cực": 1,
    "ngồi nghe đúng tư thế": 1
    }

    # Từ khóa external giữ nguyên (hoặc bổ sung sau nếu cần)
    external_keywords = {
    "so sánh": 5, "khác biệt": 4, "tương tự": 4, "khác gì": 4, "khác nhau": 4,
    "giống": 3, "thuộc về": 3, "sở hữu": 3, "của ai": 3,
    
    # === CÔNG TY/TỔ CHỨC BÊN NGOÀI === (Trọng số cao - 4)
    "google": 4, "microsoft": 4, "moodle": 4, "chatgpt": 4, "sở giáo dục": 4,
    "phòng giáo dục": 4, "bộ gd&đt": 4, "bộ giáo dục và đào tạo": 4,
    
    # === MÔN HỌC KHÁC === (Trọng số trung bình - 3)
    "toán": 3, "toán học": 3, "phương trình": 3, "đạo hàm": 3, "tích phân": 3,
    "xác suất": 3, "đại số": 3, "hình học": 3, "logarit": 3, "hàm số": 3,
    "vật lý": 3, "cơ học": 3, "nhiệt học": 3, "quang học": 3, "lực": 3,
    "hóa học": 3, "phản ứng": 3, "phân tử": 3, "nguyên tử": 3,
    "sinh học": 3, "di truyền": 3, "tiếng anh": 3, "ngữ văn": 3,
    "giáo dục địa phương": 3, "địa lý": 3, "lịch sử": 3,
    
    # === CÔNG NGHỆ TỔNG QUÁT === (Trọng số trung bình - 3)
    "trí tuệ nhân tạo là gì": 3, "công nghệ thông tin": 3, "ngôn ngữ lập trình": 3,
    "python": 3, "machine learning": 3, "học máy": 3, "deep learning": 3,
    "neural network": 3, "e-learning": 3, "blended learning": 3,
    "bài giảng trực tuyến": 3, "ứng dụng ai trong giáo dục": 3,
    
    # === KIẾN THỨC CHUNG === (Trọng số thấp - 2)
    "thời tiết": 2, "quốc gia": 2, "tổng thống": 2, "thế giới": 2,
    "định nghĩa": 2, "giải thích": 2, "kiến thức chung": 2, "giải toán": 2,
    "cách giải": 2, "trực quan": 2, "đồ thị": 2, "chu vi": 2, "diện tích": 2,
    "năng lượng": 2, "động năng": 2, "thế năng": 2, "thống kê": 2,
    
    # === THUẬT NGỮ TOÁN HỌC === (Trọng số thấp - 1)
    "pt": 1, "bậc": 1, "căn": 1, "tính": 1, "ứng dụng": 1,
    "giới hạn": 1, "vector": 1, "ma trận": 1
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

def check_feedback_followup(question: str) -> bool:
    """
    Kiểm tra xem câu hỏi hiện tại có phải là một phản hồi tiêu cực hoặc góp ý sửa câu trả lời trước không.
    Trả về True nếu có, ngược lại False.
    """
    feedback_indicators = [
        # Từ khóa thể hiện phản hồi tiêu cực
        "sai rồi", "chưa đúng", "không đúng", "không phải", "trả lời sai", "trả lời chưa chính xác", 
        "chưa chính xác", "không chính xác", "trả lời nhầm", "trả lời lệch", "giải thích sai", "viết nhầm",

        # Từ thể hiện nghi ngờ, yêu cầu làm lại
        "bạn chắc không", "bạn nhầm rồi", "kiểm tra lại", "xem lại đi", "sai dữ kiện", 
        "lỗi logic", "không đúng đề", "sai context", "không đúng ngữ cảnh",

        # Hỏi lại bằng kiểu nghi vấn phản bác
        "thật vậy sao", "chắc không", "có đúng vậy không", "bạn có nhầm gì không"
    ]

    return any(keyword in question.lower() for keyword in feedback_indicators)


