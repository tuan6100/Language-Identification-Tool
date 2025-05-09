import pytest
from src.main.python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from src.main.python.models.data import DataLoaderFactory
from src.main.python.models.language_name import language_names
from src.main.python.models.text_processor import TextProcessor


@pytest.fixture(scope="module")
def trained_model_and_processor():
    text_processor = TextProcessor(ngram_range=(1, 2), max_features=2000)
    loader = DataLoaderFactory.create_loader(
        "huggingface",
        dataset_name="papluca/language-identification",
    )
    x_train, y_train = loader.load_data(loader.dataset_name, "train")
    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    nb_model = NaiveBayesCUDAOptimized(alpha=1.0, use_gpu=True)
    nb_model.fit(x_train_processed, y_train, feature_names)

    return nb_model, text_processor


@pytest.mark.parametrize("text,expected_lang", [
    (
        "Machine learning workflows are often composed of different parts. A typical pipeline consists of a pre-processing step...",
        "English"
    ),
    (
        "Connection Pooling: Giải pháp giúp tối ưu hóa việc sử dụng kết nối đến cơ sở dữ liệu...",
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
def test_language_prediction(trained_model_and_processor, text, expected_lang):
    model, processor = trained_model_and_processor
    x_sample = processor.transform([text])
    prediction = model.predict(x_sample)
    assert language_names[prediction] == expected_lang
