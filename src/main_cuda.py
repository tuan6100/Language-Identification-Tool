import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time

from src.algorithms.classification.naive_bayes import NaiveBayes
from src.algorithms.classification.naive_bayes_cuda import NaiveBayesCUDA
from src.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from src.models.data import DataLoaderFactory, CSVDataLoader
from src.models.language_name import language_names
from src.models.text_processor import TextProcessor


def evaluate_model(y_true, y_pred, languages):
    """Đánh giá hiệu suất mô hình"""
    # Tính accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print()

    # Tạo classification report
    print('Classification Report:')
    print(classification_report(y_true, y_pred))

    # Vẽ confusion matrix
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


def compare_all_implementations(x_train, x_test, y_train, y_test, feature_names):
    """So sánh tất cả các implementation"""
    print("\n" + "="*70)
    print("So sánh hiệu suất CPU vs GPU vs GPU Optimized")
    print("="*70)

    # Test CPU
    print("\n1. CHẠY VỚI CPU:")
    cpu_start = time.time()
    cpu_model = NaiveBayes(alpha=1.0)
    cpu_model.fit(x_train, y_train, feature_names)
    y_pred_cpu = cpu_model.predict(x_test)
    cpu_time = time.time() - cpu_start
    cpu_accuracy = accuracy_score(y_test, y_pred_cpu)

    # Test GPU (original)
    print("\n2. CHẠY VỚI GPU (original):")
    gpu_start = time.time()
    gpu_model = NaiveBayesCUDA(alpha=1.0, use_gpu=True)
    gpu_model.fit(x_train, y_train, feature_names)
    y_pred_gpu = gpu_model.predict(x_test)
    gpu_time = time.time() - gpu_start
    gpu_accuracy = accuracy_score(y_test, y_pred_gpu)

    # Test GPU Optimized
    print("\n3. CHẠY VỚI GPU (optimized):")
    gpu_opt_start = time.time()
    gpu_opt_model = NaiveBayesCUDAOptimized(alpha=1.0, use_gpu=True)
    gpu_opt_model.fit(x_train, y_train, feature_names)
    y_pred_gpu_opt = gpu_opt_model.predict(x_test)
    gpu_opt_time = time.time() - gpu_opt_start
    gpu_opt_accuracy = accuracy_score(y_test, y_pred_gpu_opt)

    # In kết quả so sánh
    print("\n" + "="*70)
    print("KẾT QUẢ SO SÁNH:")
    print("-" * 70)
    print(f"{'Implementation':<20} {'Time (s)':<15} {'Accuracy':<12} {'Speedup':<10}")
    print("-" * 70)
    print(f"{'CPU':<20} {cpu_time:<15.2f} {cpu_accuracy:<12.4f} {'1.00x':<10}")
    print(f"{'GPU (original)':<20} {gpu_time:<15.2f} {gpu_accuracy:<12.4f} {f'{cpu_time/gpu_time:.2f}x':<10}")
    print(f"{'GPU (optimized)':<20} {gpu_opt_time:<15.2f} {gpu_opt_accuracy:<12.4f} {f'{cpu_time/gpu_opt_time:.2f}x':<10}")
    print("="*70)

    return cpu_model, gpu_model, gpu_opt_model


def main():
    print('Đang đọc dữ liệu...')
    data_source = "huggingface"
    if data_source == "csv":
        loader = DataLoaderFactory.create_loader("csv")
        data_path = CSVDataLoader.get_default_path()
        x, y = loader.load_data(data_path)
    else:
        loader = DataLoaderFactory.create_loader(
            "huggingface",
            dataset_name="papluca/language-identification",
            split="train"
        )
        x, y = loader.load_data()

    # Chia train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f'Số mẫu train: {len(x_train)}')
    print(f'Số mẫu test: {len(x_test)}')
    print(f'Số ngôn ngữ: {len(set(y))}')
    print()

    print('Đang xử lý văn bản...')
    text_processor = TextProcessor(ngram_range=(1, 2), max_features=1000)

    # Fit và transform dữ liệu train
    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    # Transform dữ liệu test
    x_test_processed = text_processor.transform(x_test)

    print(f'Số đặc trưng: {len(feature_names)}')
    print()

    # So sánh tất cả implementations
    cpu_model, gpu_model, gpu_opt_model = compare_all_implementations(
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        feature_names
    )

    # Sử dụng model GPU optimized cho phần còn lại
    nb_model = gpu_opt_model

    # Test trên văn bản mới
    print("\nTest với văn bản mới (GPU Optimized):")
    test_samples = [
        'To simplify, we will reuse the problem of cancer diagnosis to explain these 4 indicators.',
        'Để đơn giản hóa, ta sẽ sử dụng lại bài toán về chẩn đoán ung thư để giải thích 4 chỉ số này',
        'Pour simplifier, nous allons réutiliser le problème du diagnostic du cancer pour expliquer ces 4 indicateurs.',
        '簡単にするために、がん診断の問題を再利用して、これら 4 つの指標を説明します。',
        '为了简化起见，我们将重新使用癌症诊断的问题来解释这4个指标。',
        'เพื่อให้เข้าใจง่ายขึ้น เราจะนำปัญหาการวินิจฉัยโรคมะเร็งมาอธิบายตัวบ่งชี้ทั้ง 4 ประการนี้อีกครั้ง',
        'Basitleştirmek gerekirse, bu 4 göstergeyi açıklamak için kanser tanısı sorununu yeniden kullanacağız.'
    ]

    for text in test_samples:
        # Xử lý văn bản
        x_sample = text_processor.transform([text])

        # Dự đoán
        prediction = nb_model.predict(x_sample)
        probabilities = nb_model.predict_proba(x_sample)

        print(f"\nVăn bản: {text}")
        print(f'Ngôn ngữ dự đoán: {prediction} - {language_names[prediction]}')

        # In top 3 xác suất cao nhất
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 xác suất:")
        for lang, prob in sorted_probs:
            print(f"  {lang}: {prob:.4f}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Tổng thời gian: %s s' % (time.time() - start_time))