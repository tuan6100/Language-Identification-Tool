import os
import pytest

from python.algorithms.classification.naive_bayes import NaiveBayes
from src.main.python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from src.main.python.models.language_name import language_names


@pytest.mark.parametrize("text,expected_lang", [
    (
        "Machine learning workflows are often composed of different parts. A typical pipeline consists of a pre-processing step...",
        "English"
    ),
    (
        "Trong phương thức predict() , chúng tôi sử dụng từng mô hình đã được đào tạo để dự đoán một đầu vào. "
        "Sau đó, tổng hợp các dự đoán để đưa ra dự đoán cuối cùng. Trong trường hợp này, chúng tôi thêm tất cả các "
        "dự đoán vào một mảng và chọn dự đoán phổ biến nhất làm kết quả tổng thể của chúng tôi. Chúng tôi đã sử dụng "
        "hàm mode của Scipy để đơn giản hóa việc tìm dự đoán phổ biến nhất.",
        "Vietnamese"
    ),
    (
        "Nous sommes trois. Est-ce qu'il y a encore des places disponibles ?",
        "French"
    ),
    (
        "Без денег ничего не можешь купить",
        "Russian"
    ),
    (
        "自衛隊で1番の美マッチョを決める大会「自衛隊ベストボディ2018」が開催されるよ！",
        "Japanese"
    ),
    (
        "需要在家休息一两天。特此向您告知并望批准",
        "Chinese"
    ),
    (
        "Basitleştirmek gerekirse, bu 4 göstergeyi açıklamak için...",
        "Turkish"
    ),
    (
        "Το βιβλίο είναι ενδιαφέρον.",
        "Greek"
    )
])
def test_language_prediction_debug(text, expected_lang):
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "naive_bayes_model.pkl")
    model = None
    processor = None
    try:
        print("Trying to load GPU model...")
        model, processor = NaiveBayesCUDAOptimized.load_model(model_path, use_gpu=True)
        print("GPU model loaded successfully")
    except Exception as e:
        print(f"GPU model failed: {e}")
        try:
            print("Trying to load CPU model...")
            model, processor = NaiveBayes.load_model(model_path)
            print("CPU model loaded successfully")
        except Exception as e:
            print(f"CPU model failed: {e}")
            pytest.fail("Could not load any model")
    x_sample = processor.transform([text])
    prediction = model.predict(x_sample)
    assert language_names[prediction] == expected_lang

