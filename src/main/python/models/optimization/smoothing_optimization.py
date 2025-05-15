import time

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

from python.algorithms.classification.naive_bayes import NaiveBayes
from python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from python.models.data import DataLoaderFactory
from python.models.text_processor import TextProcessor
from python.utils.model_evaluation import evaluate_model


def optimize_alpha(x_train_processed, y_train, x_valid_processed, y_valid, feature_names):
    """
    Tìm giá trị alpha tối ưu sử dụng tập validation

    Args:
        x_train_processed: Dữ liệu huấn luyện đã xử lý
        y_train: Nhãn của dữ liệu huấn luyện
        x_valid_processed: Dữ liệu validation đã xử lý
        y_valid: Nhãn của dữ liệu validation
        feature_names: Tên các đặc trưng

    Returns:
        best_alpha: Giá trị alpha tối ưu
        results: Kết quả đánh giá với các giá trị alpha khác nhau
    """
    # Các giá trị alpha để thử nghiệm
    # Sử dụng thang logarithmic để khám phá nhiều giá trị
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

    results = []
    best_accuracy = 0
    best_alpha = 1.0  # Giá trị mặc định

    print("Bắt đầu tìm giá trị alpha tối ưu...")
    print(f"{'Alpha':<10} {'Accuracy':<10} {'F1 Score':<10}")
    print("-" * 30)

    for alpha in alpha_values:
        # Khởi tạo mô hình với giá trị alpha hiện tại
        try:
            nb_model = NaiveBayesCUDAOptimized(alpha=alpha, use_gpu=True)
        except:
            print("CUDA is required")
            nb_model = NaiveBayes(alpha=alpha)

        # Huấn luyện mô hình
        nb_model.fit(x_train_processed, y_train, feature_names)

        # Dự đoán trên tập validation
        y_pred = nb_model.predict(x_valid_processed)

        # Tính toán các meteric
        accuracy = accuracy_score(y_valid, y_pred)
        f1 = classification_report(y_valid, y_pred, output_dict=True)['weighted avg']['f1-score']

        # Lưu kết quả
        results.append({
            'alpha': alpha,
            'accuracy': accuracy,
            'f1_score': f1
        })

        print(f"{alpha:<10.3f} {accuracy:<10.4f} {f1:<10.4f}")

        # Cập nhật alpha tốt nhất dựa trên accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    print("\nGiá trị alpha tối ưu:", best_alpha)
    print(f"Accuracy tốt nhất trên tập validation: {best_accuracy:.4f}")

    return best_alpha, results


def visualize_alpha_tuning(results):
    """
    Trực quan hóa kết quả điều chỉnh alpha

    Args:
        results: Danh sách kết quả từ quá trình điều chỉnh alpha
    """
    # Chuyển đổi dữ liệu để vẽ
    alphas = [result['alpha'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    f1_scores = [result['f1_score'] for result in results]

    plt.figure(figsize=(12, 6))

    # Vẽ biểu đồ accuracy
    plt.subplot(1, 2, 1)
    plt.semilogx(alphas, accuracies, 'o-', linewidth=2, markersize=8)
    plt.axvline(alphas[accuracies.index(max(accuracies))], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Alpha (logarithmic scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Alpha')
    plt.grid(True, alpha=0.3)

    # Vẽ biểu đồ F1 score
    plt.subplot(1, 2, 2)
    plt.semilogx(alphas, f1_scores, 'o-', linewidth=2, markersize=8, color='orange')
    plt.axvline(alphas[f1_scores.index(max(f1_scores))], color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Alpha (logarithmic scale)')
    plt.ylabel('F1 Score (weighted)')
    plt.title('F1 Score vs Alpha')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alpha_tuning_results.png')
    plt.show()


def analyze_alpha_effect(best_alpha, x_train_processed, y_train, x_valid_processed, y_valid, feature_names, languages):
    """
    Phân tích chi tiết hiệu suất của mô hình với alpha tối ưu

    Args:
        best_alpha: Giá trị alpha tối ưu
        x_train_processed, y_train: Dữ liệu huấn luyện
        x_valid_processed, y_valid: Dữ liệu validation
        feature_names: Tên các đặc trưng
        languages: Danh sách các ngôn ngữ
    """
    print("\nPhân tích chi tiết với alpha tối ưu:", best_alpha)

    # Huấn luyện mô hình với alpha tối ưu
    nb_model = NaiveBayesCUDAOptimized(alpha=best_alpha, use_gpu=True)
    nb_model.fit(x_train_processed, y_train, feature_names)

    # Dự đoán trên tập validation
    y_pred = nb_model.predict(x_valid_processed)

    # Đánh giá chi tiết
    print("\nĐánh giá mô hình trên tập validation với alpha =", best_alpha)
    evaluate_model(y_valid, y_pred, languages)

    # Phân tích các lớp có hiệu suất tốt và kém
    report = classification_report(y_valid, y_pred, output_dict=True)

    # Sắp xếp các ngôn ngữ theo F1-score
    lang_scores = [(lang, report[lang]['f1-score']) for lang in languages if lang in report]
    lang_scores.sort(key=lambda x: x[1])

    print("\nCác ngôn ngữ có hiệu suất thấp nhất (F1-score):")
    for lang, score in lang_scores[:3]:  # 3 ngôn ngữ có hiệu suất kém nhất
        print(f"{lang}: {score:.4f}")

    print("\nCác ngôn ngữ có hiệu suất cao nhất (F1-score):")
    for lang, score in reversed(lang_scores[-3:]):  # 3 ngôn ngữ có hiệu suất tốt nhất
        print(f"{lang}: {score:.4f}")

    return nb_model


def main():
    print('Đang đọc dữ liệu...')
    data_source = "huggingface"
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

    print(f'Số mẫu train: {len(x_train)}')
    print(f'Số mẫu validation: {len(x_valid)}')
    print(f'Số ngôn ngữ: {len(set(y_valid))}')
    print()

    print('Đang xử lý văn bản...')
    text_processor = TextProcessor(ngram_range=(1, 2), max_features=2000)

    # Fit và transform dữ liệu train
    x_train_processed = text_processor.fit_transform(x_train)
    feature_names = text_processor.get_feature_names()

    x_test_processed = text_processor.transform(x_test)
    x_valid_processed = text_processor.transform(x_valid)

    print(f'Số đặc trưng: {len(feature_names)}')

    # Các ngôn ngữ
    languages = sorted(list(set(y_train)))

    # -------------------------------------
    # Bước 1: Tối ưu hóa tham số alpha
    # -------------------------------------
    print('\n=== Tối ưu hóa tham số alpha sử dụng tập validation ===\n')
    best_alpha, tuning_results = optimize_alpha(
        x_train_processed, y_train,
        x_valid_processed, y_valid,
        feature_names
    )

    # Trực quan hóa kết quả tối ưu hóa
    visualize_alpha_tuning(tuning_results)

    # Phân tích chi tiết với alpha tối ưu
    optimized_model = analyze_alpha_effect(
        best_alpha,
        x_train_processed, y_train,
        x_valid_processed, y_valid,
        feature_names, languages
    )

    # -------------------------------------
    # Bước 2: Đánh giá trên tập kiểm tra với alpha tối ưu
    # -------------------------------------
    print('\n=== Đánh giá mô hình trên tập kiểm tra với alpha tối ưu ===\n')
    y_pred = optimized_model.predict(x_test_processed)
    evaluate_model(y_test, y_pred, languages)

    # -------------------------------------
    # Bước 3: So sánh với mô hình mặc định (alpha=1.0)
    # -------------------------------------
    if best_alpha != 1.0:
        print('\n=== So sánh với mô hình mặc định (alpha=1.0) ===\n')
        default_model = NaiveBayesCUDAOptimized(alpha=1.0, use_gpu=True)
        default_model.fit(x_train_processed, y_train, feature_names)
        default_y_pred = default_model.predict(x_test_processed)

        default_accuracy = accuracy_score(y_test, default_y_pred)
        optimized_accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy với alpha mặc định (1.0): {default_accuracy:.4f}')
        print(f'Accuracy với alpha tối ưu ({best_alpha}): {optimized_accuracy:.4f}')
        print(f'Cải thiện: {(optimized_accuracy - default_accuracy) * 100:.2f}%')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Tổng thời gian: %s s' % (time.time() - start_time))