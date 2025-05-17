import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def evaluate_model(y_true, y_pred, languages):
    """
    Đánh giá hiệu suất mô hình
    Args:
        y_true (list hoặc array-like): Nhãn chuẩn của tập dữ liệu.
        y_pred (list hoặc array-like): Nhãn dự đoán sau khi train và test.
        languages (list): Danh sách các nhãn ngôn ngữ được sử dụng cho Confusion Matrix.
    """

    # Tính accuracy
    # y_true = [en, vi, fr, en, ....]
    # y_pred = [vi, vi, en, fr]



    # accuracy = accuracy_score(y_true, y_pred)
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    # print()
    accuracy = custom_accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    cm = custom_confusion_matrix(y_true, y_pred, labels=languages)
    # Tạo classification report
    print('Classification Report:')
    print(classification_report(y_true, y_pred))

    # Vẽ confusion matrix
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



# Khong dung sklearn
def custom_accuracy_score(y_true, y_pred):
    """
    Tính tỷ lệ chính xác của mô hình.

    Args:
        y_true (list): Danh sách nhãn thực tế.
        y_pred (list): Danh sách nhãn mô hình dự đoán.

    Returns:
        float: Accuracy (từ 0.0 đến 1.0)
    """
    correct = 0
    total = len(y_true)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct += 1
    return correct / total if total > 0 else 0.0

def classification_report(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    report = {}
    total_correct = 0
    total_samples = len(y_true)

    for label in labels:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        support = sum(yt == label for yt in y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[label] = {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1-score': round(f1_score, 2),
            'support': support
        }

        total_correct += tp

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Tính macro average
    precisions = [v['precision'] for v in report.values()]
    recalls = [v['recall'] for v in report.values()]
    f1s = [v['f1-score'] for v in report.values()]
    supports = [v['support'] for v in report.values()]
    total_support = sum(supports)

    macro_avg = {
        'precision': round(np.mean(precisions), 2),
        'recall': round(np.mean(recalls), 2),
        'f1-score': round(np.mean(f1s), 2),
        'support': total_support
    }

    weighted_avg = {
        'precision': round(np.average(precisions, weights=supports), 2),
        'recall': round(np.average(recalls, weights=supports), 2),
        'f1-score': round(np.average(f1s, weights=supports), 2),
        'support': total_support
    }

    # In kết quả
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    for label, metrics in report.items():
        print(
            f"{label:<10} {metrics['precision']:>10.2f} {metrics['recall']:>10.2f} {metrics['f1-score']:>10.2f} {metrics['support']:>10}")

    print(f"\n{'Accuracy':<10} {accuracy:>10.2f}")
    return report


def custom_confusion_matrix(y_true, y_pred, labels):
    n_classes = len(labels)
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for true_label, pred_label in zip(y_true, y_pred):
        try:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            cm[true_idx][pred_idx] += 1
        except KeyError:
            continue
    return cm