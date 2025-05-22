import logging
import time
import os
import joblib

import coloredlogs

from algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from utils.dataset_comparison import compare_ngram_structure, check_exact_duplicates
from utils.model_evaluation import evaluate_model
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

    # Đường dẫn để lưu model - sử dụng đường dẫn tuyệt đối
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    models_dir = os.path.join(project_root, 'models')
    
    # Đảm bảo thư mục models tồn tại
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'naive_bayes_model.joblib')
    processor_path = os.path.join(models_dir, 'text_processor.joblib')
    
    logger.info(f"Model sẽ được lưu tại: {model_path}")
    logger.info(f"Text processor sẽ được lưu tại: {processor_path}")
    
    # Kiểm tra nếu model đã tồn tại
    if os.path.exists(model_path) and os.path.exists(processor_path):
        logger.info('Đang tải model đã lưu...')
        nb_model = joblib.load(model_path)
        loaded_text_processor = joblib.load(processor_path)
        # Sử dụng processor đã lưu để xử lý dữ liệu test
        x_test_processed = loaded_text_processor.transform(x_test)
    else:
        logger.info('Đang huấn luyện mô hình mới...')
        try:
            logger.info(f"Sử dụng GPU - Optimized Version")
            nb_model = NaiveBayesCUDAOptimized(alpha=0.001, use_gpu=False)
            nb_model.fit(x_train_processed, y_train, feature_names)
            
            # Lưu model và text processor
            logger.info('Đang lưu model...')
            joblib.dump(nb_model, model_path)
            joblib.dump(text_processor, processor_path)
            logger.info('Đã lưu model thành công')
            
        except:
            logger.info(f"Sử dụng CPU ")
            nb_model = NaiveBayes(alpha=0.001)
            nb_model.fit(x_train_processed, y_train, feature_names)
            
            # Lưu model và text processor
            logger.info('Đang lưu model...')
            joblib.dump(nb_model, model_path)
            joblib.dump(text_processor, processor_path)
            logger.info('Đã lưu model thành công')
    
    logger.info('Đang dự đoán...')
    y_pred = nb_model.predict(x_test_processed) # [en vi en fr ....]

    logger.info('\nĐánh giá mô hình:')
    languages = sorted(list(set(y_test)))
    evaluate_model(y_test, y_pred, languages)


if __name__ == '__main__':
    start_time = time.time()
    main()
    logger.info('Thời gian chạy: %s s' % (time.time() - start_time))