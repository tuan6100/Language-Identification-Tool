import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import time
from collections import defaultdict
import random

def learning_curve(model_class, x_data, y_data, x_test, y_test, text_processor,
                   train_sizes=None, cv_folds=3, model_params=None,
                   title="Learning Curve", figsize=(12, 8)):
    """
    Vẽ learning curve để kiểm định underfitting và overfitting

    Args:
        model_class: Class của model (NaiveBayes hoặc NaiveBayesCUDAOptimized)
        x_data: Dữ liệu văn bản đầy đủ
        y_data: Nhãn đầy đủ tương ứng
        x_test: Dữ liệu test
        y_test: Nhãn test
        text_processor: TextProcessor để xử lý văn bản
        train_sizes: List kích thước tập train (mặc định từ 500 đến toàn bộ)
        cv_folds: Số folds cho cross-validation
        model_params: Parameters cho model
        title: Tiêu đề biểu đồ
        figsize: Kích thước figure

    Returns:
        dict: Kết quả learning curve
    """

    if model_params is None:
        model_params = {'alpha': 0.001}

    # Tính số lượng mẫu trên mỗi lớp
    class_counts = defaultdict(int)
    for label in y_data:
        class_counts[label] += 1

    min_samples_per_class = min(class_counts.values())
    print(f"Số mẫu tối thiểu trên mỗi lớp: {min_samples_per_class}")
    print(f"Phân phối lớp: {dict(class_counts)}")

    # Thiết lập train_sizes mặc định
    if train_sizes is None:
        max_size_per_class = min(3500, min_samples_per_class)
        train_sizes = list(range(500, max_size_per_class + 1, 500))
        # Thêm một số kích thước nhỏ hơn để thấy rõ underfitting
        train_sizes = [100, 200] + train_sizes
        train_sizes = sorted(list(set(train_sizes)))

    print(f"Kích thước train sẽ test: {train_sizes}")

    # Chuẩn bị dữ liệu theo lớp
    class_data = defaultdict(list)
    for i, label in enumerate(y_data):
        class_data[label].append((x_data[i], label))

    # Shuffle dữ liệu trong mỗi lớp
    for label in class_data:
        random.shuffle(class_data[label])

    # Xử lý dữ liệu test một lần
    print("Đang xử lý dữ liệu test...")
    x_test_processed = text_processor.transform(x_test)

    # Lưu kết quả
    results = {
        'train_sizes': [],
        'train_scores_mean': [],
        'train_scores_std': [],
        'test_scores_mean': [],
        'test_scores_std': [],
        'fit_times_mean': [],
        'fit_times_std': []
    }

    print("\n=== BẮT ĐẦU LEARNING CURVE ===")

    for size_per_class in train_sizes:
        print(f"\n--- Training với {size_per_class} mẫu/lớp ---")

        # Kiểm tra xem có đủ dữ liệu không
        if size_per_class > min_samples_per_class:
            print(f"Không đủ dữ liệu cho {size_per_class} mẫu/lớp, bỏ qua.")
            continue

        total_train_size = size_per_class * len(class_data)

        # Cross-validation
        train_scores = []
        test_scores = []
        fit_times = []

        for fold in range(cv_folds):
            print(f"  Fold {fold + 1}/{cv_folds}...")

            # Tạo tập train cho fold này
            fold_x_train = []
            fold_y_train = []

            for label, data_list in class_data.items():
                # Chia dữ liệu thành cv_folds phần
                fold_size = len(data_list) // cv_folds
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size

                # Lấy dữ liệu cho fold hiện tại (phần còn lại làm validation)
                fold_data = data_list[:start_idx] + data_list[end_idx:]

                # Lấy size_per_class mẫu đầu tiên
                selected_data = fold_data[:size_per_class]

                for text, label_item in selected_data:
                    fold_x_train.append(text)
                    fold_y_train.append(label_item)

            # Xử lý dữ liệu train cho fold này
            fold_x_train_processed = text_processor.transform(fold_x_train)
            feature_names = text_processor.get_feature_names()

            # Train model
            start_time = time.time()

            try:
                # Thử GPU model trước
                if 'use_gpu' in model_class.__init__.__code__.co_varnames:
                    model = model_class(use_gpu=True, **model_params)
                else:
                    model = model_class(**model_params)
            except:
                model = model_class(**model_params)

            model.fit(fold_x_train_processed, fold_y_train, feature_names)
            fit_time = time.time() - start_time
            fit_times.append(fit_time)

            # Đánh giá trên tập train
            train_pred = model.predict(fold_x_train_processed)
            train_accuracy = accuracy_score(fold_y_train, train_pred)
            train_scores.append(train_accuracy)

            # Đánh giá trên tập test
            test_pred = model.predict(x_test_processed)
            test_accuracy = accuracy_score(y_test, test_pred)
            test_scores.append(test_accuracy)

            print(f"    Train accuracy: {train_accuracy:.4f}")
            print(f"    Test accuracy: {test_accuracy:.4f}")
            print(f"    Fit time: {fit_time:.2f}s")

        # Tính trung bình và độ lệch chuẩn
        results['train_sizes'].append(total_train_size)
        results['train_scores_mean'].append(np.mean(train_scores))
        results['train_scores_std'].append(np.std(train_scores))
        results['test_scores_mean'].append(np.mean(test_scores))
        results['test_scores_std'].append(np.std(test_scores))
        results['fit_times_mean'].append(np.mean(fit_times))
        results['fit_times_std'].append(np.std(fit_times))

        print(f"  Trung bình - Train: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"  Trung bình - Test: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")

    # Vẽ biểu đồ
    plot_learning_curve(results, title, figsize)

    # Phân tích kết quả
    analyze_learning_curve(results)

    return results

def plot_learning_curve(results, title="Learning Curve", figsize=(12, 8)):
    """Vẽ biểu đồ learning curve"""

    plt.figure(figsize=figsize)

    train_sizes = results['train_sizes']
    train_scores_mean = results['train_scores_mean']
    train_scores_std = results['train_scores_std']
    test_scores_mean = results['test_scores_mean']
    test_scores_std = results['test_scores_std']

    # Vẽ đường train score
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue',
             label='Training Accuracy', linewidth=2, markersize=8)
    plt.fill_between(train_sizes,
                     np.array(train_scores_mean) - np.array(train_scores_std),
                     np.array(train_scores_mean) + np.array(train_scores_std),
                     alpha=0.2, color='blue')

    # Vẽ đường test score
    plt.plot(train_sizes, test_scores_mean, 'o-', color='red',
             label='Testing Accuracy', linewidth=2, markersize=8)
    plt.fill_between(train_sizes,
                     np.array(test_scores_mean) - np.array(test_scores_std),
                     np.array(test_scores_mean) + np.array(test_scores_std),
                     alpha=0.2, color='red')

    # Tùy chỉnh biểu đồ
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Thiết lập trục
    plt.ylim(0, 1.05)
    plt.xlim(min(train_sizes) * 0.9, max(train_sizes) * 1.1)

    # Thêm chú thích về overfitting/underfitting
    max_gap = max(np.array(train_scores_mean) - np.array(test_scores_mean))
    plt.text(0.02, 0.98, f'Max Train-Test Gap: {max_gap:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

def analyze_learning_curve(results):
    """Phân tích kết quả learning curve"""

    print("\n" + "="*60)
    print("PHÂN TÍCH LEARNING CURVE")
    print("="*60)

    train_scores = np.array(results['train_scores_mean'])
    test_scores = np.array(results['test_scores_mean'])
    train_sizes = np.array(results['train_sizes'])

    # 1. Phân tích Underfitting
    print("\n1. PHÂN TÍCH UNDERFITTING:")
    initial_train_score = train_scores[0]
    initial_test_score = test_scores[0]

    if initial_train_score < 0.7 or initial_test_score < 0.7:
        print("   ⚠️  UNDERFITTING được phát hiện!")
        print(f"   - Train accuracy ban đầu: {initial_train_score:.3f}")
        print(f"   - Test accuracy ban đầu: {initial_test_score:.3f}")
        print("   📋 Khuyến nghị:")
        print("      • Giảm regularization (alpha)")
        print("      • Tăng số lượng features (max_features)")
        print("      • Thử mô hình phức tạp hơn")
    else:
        print("   ✅ Không có dấu hiệu underfitting rõ ràng")

    # 2. Phân tích Overfitting
    print("\n2. PHÂN TÍCH OVERFITTING:")
    gaps = train_scores - test_scores
    max_gap = np.max(gaps)
    avg_gap = np.mean(gaps)

    print(f"   - Gap trung bình (Train - Test): {avg_gap:.3f}")
    print(f"   - Gap lớn nhất: {max_gap:.3f}")

    if max_gap > 0.1:
        print("   ⚠️  OVERFITTING được phát hiện!")
        print("   📋 Khuyến nghị:")
        print("      • Tăng regularization (alpha)")
        print("      • Giảm số lượng features")
        print("      • Thu thập thêm dữ liệu training")
        print("      • Sử dụng cross-validation")
    elif max_gap > 0.05:
        print("   ⚠️  Có dấu hiệu overfitting nhẹ")
        print("   📋 Khuyến nghị: Theo dõi và có thể tăng nhẹ regularization")
    else:
        print("   ✅ Không có dấu hiệu overfitting đáng kể")

    # 3. Phân tích xu hướng
    print("\n3. PHÂN TÍCH XU HƯỚNG:")

    # Xu hướng train score
    if len(train_scores) >= 3:
        train_trend = np.polyfit(range(len(train_scores)), train_scores, 1)[0]
        test_trend = np.polyfit(range(len(test_scores)), test_scores, 1)[0]

        print(f"   - Xu hướng Train accuracy: {train_trend:+.6f}/step")
        print(f"   - Xu hướng Test accuracy: {test_trend:+.6f}/step")

        if test_trend > 0.001:
            print("   📈 Test accuracy đang tăng - có thể cần thêm dữ liệu")
        elif test_trend < -0.001:
            print("   📉 Test accuracy đang giảm - có dấu hiệu overfitting")
        else:
            print("   ➡️  Test accuracy ổn định - mô hình đã hội tụ")

    # 4. Khuyến nghị tổng quát
    print("\n4. KHUYẾN NGHỊ TỔNG QUÁT:")

    final_train_score = train_scores[-1]
    final_test_score = test_scores[-1]
    final_gap = final_train_score - final_test_score

    if final_test_score > 0.9:
        print("   🎯 Hiệu suất tốt!")
    elif final_test_score > 0.8:
        print("   👍 Hiệu suất khá tốt")
    else:
        print("   🔧 Cần cải thiện hiệu suất")

    if final_gap < 0.03:
        print("   ⚖️  Mô hình cân bằng tốt")
    elif final_gap < 0.1:
        print("   ⚖️  Mô hình tương đối cân bằng")
    else:
        print("   ⚠️  Mô hình không cân bằng - cần điều chỉnh")

    # 5. Thống kê chi tiết
    print(f"\n5. THỐNG KÊ CHI TIẾT:")
    print(f"   - Kích thước training nhỏ nhất: {min(train_sizes):,}")
    print(f"   - Kích thước training lớn nhất: {max(train_sizes):,}")
    print(f"   - Train accuracy cuối: {final_train_score:.4f}")
    print(f"   - Test accuracy cuối: {final_test_score:.4f}")
    print(f"   - Thời gian fit trung bình: {np.mean(results['fit_times_mean']):.2f}s")

# Hàm sử dụng với dữ liệu thực tế
def run_learning_curve_analysis():
    """
    Chạy phân tích learning curve với dữ liệu thực tế
    """
    from src.main.python.models.data import DataLoaderFactory
    from src.main.python.models.text_processor import TextProcessor
    from src.main.python.algorithms.classification.naive_bayes import NaiveBayes
    from src.main.python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized

    print("=== PHÂN TÍCH LEARNING CURVE CHO NAIVE BAYES ===")

    # Load dữ liệu
    print("Đang tải dữ liệu...")
    loader = DataLoaderFactory.create_loader(
        "huggingface",
        dataset_name="papluca/language-identification",
    )
    x_train, y_train = loader.load_data(loader.dataset_name, "train")
    x_test, y_test = loader.load_data(loader.dataset_name, "test")

    # Chuẩn bị text processor
    print("Đang chuẩn bị text processor...")
    text_processor = TextProcessor(ngram_range=(1, 3), max_features=5000)
    text_processor.fit_transform(x_train)

    # Chạy learning curve cho CPU model
    print("\n" + "="*50)
    print("LEARNING CURVE CHO NAIVE BAYES (CPU)")
    print("="*50)

    cpu_results = learning_curve(
        model_class=NaiveBayes,
        x_data=x_train,
        y_data=y_train,
        x_test=x_test,
        y_test=y_test,
        text_processor=text_processor,
        train_sizes=[100, 200, 500, 1000, 1500, 2000, 2500, 3000],
        cv_folds=3,
        model_params={'alpha': 0.001},
        title="Learning Curve - Naive Bayes (CPU)",
        figsize=(12, 8)
    )

    # Chạy learning curve cho GPU model (nếu có)
    try:
        print("\n" + "="*50)
        print("LEARNING CURVE CHO NAIVE BAYES CUDA (GPU)")
        print("="*50)

        gpu_results = learning_curve(
            model_class=NaiveBayesCUDAOptimized,
            x_data=x_train,
            y_data=y_train,
            x_test=x_test,
            y_test=y_test,
            text_processor=text_processor,
            train_sizes=[100, 200, 500, 1000, 1500, 2000, 2500, 3000],
            cv_folds=3,
            model_params={'alpha': 0.001, 'use_gpu': True},
            title="Learning Curve - Naive Bayes CUDA (GPU)",
            figsize=(12, 8)
        )

        # So sánh CPU vs GPU
        compare_cpu_gpu_performance(cpu_results, gpu_results)

    except Exception as e:
        print(f"Không thể chạy GPU learning curve: {e}")

    return cpu_results

def compare_cpu_gpu_performance(cpu_results, gpu_results):
    """So sánh hiệu suất giữa CPU và GPU"""

    print("\n" + "="*50)
    print("SO SÁNH HIỆU SUẤT CPU vs GPU")
    print("="*50)

    # So sánh thời gian training
    cpu_times = np.array(cpu_results['fit_times_mean'])
    gpu_times = np.array(gpu_results['fit_times_mean'])

    print(f"Thời gian training trung bình:")
    print(f"  CPU: {np.mean(cpu_times):.2f}s ± {np.std(cpu_times):.2f}s")
    print(f"  GPU: {np.mean(gpu_times):.2f}s ± {np.std(gpu_times):.2f}s")
    print(f"  Speedup: {np.mean(cpu_times)/np.mean(gpu_times):.2f}x")

    # So sánh accuracy
    cpu_final_acc = cpu_results['test_scores_mean'][-1]
    gpu_final_acc = gpu_results['test_scores_mean'][-1]

    print(f"\nAccuracy cuối cùng:")
    print(f"  CPU: {cpu_final_acc:.4f}")
    print(f"  GPU: {gpu_final_acc:.4f}")
    print(f"  Chênh lệch: {abs(gpu_final_acc - cpu_final_acc):.4f}")

    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 5))

    # Subplot 1: So sánh accuracy
    plt.subplot(1, 3, 1)
    train_sizes = cpu_results['train_sizes']
    plt.plot(train_sizes, cpu_results['test_scores_mean'], 'o-', label='CPU', linewidth=2)
    plt.plot(train_sizes, gpu_results['test_scores_mean'], 's-', label='GPU', linewidth=2)
    plt.xlabel('Training Size')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: So sánh thời gian training
    plt.subplot(1, 3, 2)
    plt.plot(train_sizes, cpu_times, 'o-', label='CPU', linewidth=2)
    plt.plot(train_sizes, gpu_times, 's-', label='GPU', linewidth=2)
    plt.xlabel('Training Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Speedup
    plt.subplot(1, 3, 3)
    speedup = cpu_times / gpu_times
    plt.plot(train_sizes, speedup, 'g^-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
    plt.xlabel('Training Size')
    plt.ylabel('Speedup (CPU time / GPU time)')
    plt.title('GPU Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Chạy phân tích learning curve
    results = run_learning_curve_analysis()