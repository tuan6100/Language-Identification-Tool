import time

from python.utils.dataset_comparison import check_exact_duplicates, compare_ngram_structure
from python.utils.model_evaluation import evaluate_model
from src.main.python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from src.main.python.models.data import DataLoaderFactory
from src.main.python.models.text_processor import TextProcessor


def main():
    print('Đang đọc dữ liệu...')
    data_source = "huggingface"
    if data_source == "csv":
        loader = DataLoaderFactory.create_loader("csv")
        x_train, y_train = loader.load_data("training")
        x_test, y_test = loader.load_data("test")
    else:
        loader = DataLoaderFactory.create_loader(
            "huggingface",
            dataset_name="papluca/language-identification",
        )
        x_train, y_train = loader.load_data(loader.dataset_name, "train")
        x_test, y_test = loader.load_data(loader.dataset_name, "test")

    print(f'Số mẫu train: {len(x_train)}')
    print(f'Số mẫu test: {len(x_test)}')
    print(f'Số ngôn ngữ: {len(set(y_test))}')
    print()
    print(f'So sánh trùng lặp giữa tập train và test')
    check_exact_duplicates(x_train, x_test)
    print()

    print('Đang xử lý văn bản...')
    text_processor = TextProcessor(ngram_range=(1, 2), max_features=2000)

    # Fit và transform dữ liệu train
    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    # Transform dữ liệu test
    x_test_processed = text_processor.transform(x_test)
    print(f'Số đặc trưng: {len(feature_names)}')
    print("So sánh cấu trúc n-gram")
    compare_ngram_structure(x_train_processed, x_test_processed, feature_names)

    print('Đang huấn luyện mô hình...')
    nb_model = NaiveBayesCUDAOptimized(alpha=0.01, use_gpu=True)
    nb_model.fit(x_train_processed , y_train, feature_names)
    print('Đang dự đoán...')
    y_pred = nb_model.predict(x_test_processed)
    print('\nĐánh giá mô hình:')
    languages = sorted(list(set(y_test)))
    evaluate_model(y_test, y_pred, languages)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Tổng thời gian: %s s' % (time.time() - start_time))
