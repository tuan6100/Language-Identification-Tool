import logging
import time

import coloredlogs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.main.python.algorithms.classification.naive_bayes import NaiveBayes
from src.main.python.models.data import DataLoaderFactory
from src.main.python.models.language_name import language_names
from src.main.python.models.text_processor import TextProcessor

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def evaluate_model(y_true, y_pred, languages):
    """
    Đánh giá hiệu suất mô hình

    Args:
        y_true: list, nhãn thực
        y_pred: list, nhãn dự đoán
        languages: list, danh sách ngôn ngữ
    """
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f'Accuracy: {accuracy:.4f}')

    logger.info('Classification Report:')
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=languages)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=languages, yticklabels=languages)
    plt.title('Confusion Matrix - Language Detection')
    plt.xlabel('Predicted Language')
    plt.ylabel('True Language')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    logger.info('Đang đọc dữ liệu...')
    data_source = 'huggingface'
    if data_source == "csv":
        loader = DataLoaderFactory.create_loader("csv")
        x_train, y_train = loader.load_data("training")
        x_test, y_test = loader.load_data("test")
        x_valid, y_valid = loader.load_data("validation")
    else:
        loader = DataLoaderFactory.create_loader(
            "huggingface",
            dataset_name="papluca/language-identification",
        )
        x_train, y_train = loader.load_data(loader.dataset_name, "train")
        x_test, y_test = loader.load_data(loader.dataset_name, "test")
        x_valid, y_valid = loader.load_data(loader.dataset_name, "validation")

    logger.info(f'Số mẫu train: {len(x_train)}')
    logger.info(f'Số mẫu test: {len(x_test)}')
    logger.info(f'Số ngôn ngữ: {len(set(y_test))}')

    logger.info('Đang xử lý văn bản...')
    text_processor = TextProcessor(ngram_range=(1, 2), max_features=2000)

    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    x_test_processed = text_processor.transform(x_test)
    logger.info(f'Số đặc trưng: {len(feature_names)}')
    logger.info('Đang huấn luyện mô hình...')
    nb_model = NaiveBayes(alpha=1.0)
    nb_model.fit(x_train_processed, y_train, feature_names)
    logger.info('Đang dự đoán...')
    y_pred = nb_model.predict(x_test_processed)
    logger.info('\nĐánh giá mô hình:')
    languages = sorted(list(set(y_test)))
    evaluate_model(y_test, y_pred, languages)

    # logger.info('\nTest với văn bản mới:')
    # test_samples = [
    #     'To simplify, we will reuse the problem of cancer diagnosis to explain these 4 indicators.',
    #     'Để đơn giản hóa, ta sẽ sử dụng lại bài toán về chẩn đoán ung thư để giải thích 4 chỉ số này',
    #     'Pour simplifier, nous allons réutiliser le problème du diagnostic du cancer pour expliquer ces 4 indicateurs.',
    #     '簡単にするために、がん診断の問題を再利用して、これら 4 つの指標を説明します。',
    #     '为了简化起见，我们将重新使用癌症诊断的问题来解释这4个指标。',
    #     'เพื่อให้เข้าใจง่ายขึ้น เราจะนำปัญหาการวินิจฉัยโรคมะเร็งมาอธิบายตัวบ่งชี้ทั้ง 4 ประการนี้อีกครั้ง',
    #     'Basitleştirmek gerekirse, bu 4 göstergeyi açıklamak için kanser tanısı sorununu yeniden kullanacağız.'
    # ]
    #
    # for text in test_samples:
    #     x_sample = text_processor.transform([text])
    #
    #     prediction = nb_model.predict(x_sample)
    #     probabilities = nb_model.predict_proba(x_sample)
    #
    #     logger.info(f'\nVăn bản: {text}')
    #     logger.info(f'Ngôn ngữ dự đoán: {prediction} - {language_names[prediction]}')
    #
    #     sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    #     logger.info('Top 3 xác suất:')
    #     for lang, prob in sorted_probs:
    #         print(f'  {lang}: {prob:.4f}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    logger.info('Thời gian chạy: %s s' % (time.time() - start_time))