import logging
import time

import coloredlogs

from python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from python.utils.dataset_comparison import compare_ngram_structure, check_exact_duplicates
from python.utils.model_evaluation import evaluate_model
from src.main.python.algorithms.classification.naive_bayes import NaiveBayes
from src.main.python.models.data import DataLoaderFactory
from src.main.python.models.text_processor import TextProcessor

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)
coloredlogs.install(logger=logger)

def main():
    try:
        logger.info("Đang đọc dữ liệu từ data folder")
        loader = DataLoaderFactory.create_loader('csv')
        x_train, y_train = loader.load_data('training')
        x_test, y_test = loader.load_data('test')
    except:
        logger.info("Đang đọc dữ liệu từ huggingface")
        loader = DataLoaderFactory.create_loader(
            'huggingface',
            dataset_name='papluca/language-identification',
        )
        x_train, y_train = loader.load_data(loader.dataset_name, 'train')
        x_test, y_test = loader.load_data(loader.dataset_name, 'test')

    logger.info(f'Số mẫu train: {len(x_train)}')
    logger.info(f'Số mẫu test: {len(x_test)}')
    logger.info(f'Số ngôn ngữ: {len(set(y_test))}')
    print()
    logger.info(f'So sánh trùng lặp giữa tập train và test')
    check_exact_duplicates(x_train, x_test)
    print()

    logger.info('Đang xử lý văn bản...')
    text_processor = TextProcessor(ngram_range=(1, 3), max_features=7500)

    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    x_test_processed = text_processor.transform(x_test)
    logger.info(f'Số đặc trưng: {len(feature_names)}')
    logger.info('So sánh cấu trúc n-gram')
    compare_ngram_structure(x_train_processed, x_test_processed, feature_names)

    logger.info('Đang huấn luyện mô hình...')
    try:
        logger.info(f"Sử dụng GPU - Optimized Version")
        nb_model = NaiveBayesCUDAOptimized(alpha=0.001, use_gpu=True)
        nb_model.fit(x_train_processed, y_train, feature_names)
        logger.info('Đang dự đoán...')
        y_pred = nb_model.predict(x_test_processed)
    except:
        nb_model = NaiveBayes(alpha=0.001)
        nb_model.fit(x_train_processed, y_train, feature_names)
        logger.info('Đang dự đoán...')
        y_pred = nb_model.predict(x_test_processed) # [en vi en fr ....]

    logger.info('\nĐánh giá mô hình:')
    languages = sorted(list(set(y_test)))
    evaluate_model(y_test, y_pred, languages)


if __name__ == '__main__':
    start_time = time.time()
    main()
    logger.info('Thời gian chạy: %s s' % (time.time() - start_time))