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
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print()

    # Tạo classification report
    print('Classification Report:')
    # f1 = 2 / ( precision^-1 + recall^-1)
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

# Khong dung sklearn
def accuracy_score(y_true, y_pred):
    return 0
    # TODO: Quan


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
    print(
        f"{'Macro avg':<10} {macro_avg['precision']:>10.2f} {macro_avg['recall']:>10.2f} {macro_avg['f1-score']:>10.2f} {macro_avg['support']:>10}")
    print(
        f"{'Weighted avg':<10} {weighted_avg['precision']:>10.2f} {weighted_avg['recall']:>10.2f} {weighted_avg['f1-score']:>10.2f} {weighted_avg['support']:>10}")

    return report


def confusion_matrix(y_true, y_pred, labels):
    return 0
    # TODO: Tuan Anh