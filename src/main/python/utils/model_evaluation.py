import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from prettytable import PrettyTable

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
    custom_classification_report(cm, labels=languages)
    print()


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


def custom_classification_report(cm, labels):
    report = {}
    cm = np.array(cm)
    total_samples = cm.sum()
    precisions = []
    recalls = []
    f1s = []
    supports = cm.sum(axis=1)
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[:, i].sum()
        fn = cm[i, :].sum()
        precision = tp / fp if fp > 0 else 0.0
        recall = tp / fn if fn > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_score)

        report[label] = {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1-score': round(f1_score, 2),
            'support': int(supports[i])
        }
    tp_sum = np.sum([cm[i][i] for i in range(len(labels))])
    fp_sum = np.sum([np.sum(cm[:, i]) - cm[i][i] for i in range(len(labels))])
    fn_sum = np.sum([np.sum(cm[i, :]) - cm[i][i] for i in range(len(labels))])

    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    weighted_precision = np.average(precisions, weights=supports) if sum(supports) > 0 else 0.0
    weighted_recall = np.average(recalls, weights=supports) if sum(supports) > 0 else 0.0
    weighted_f1 = np.average(f1s, weights=supports) if sum(supports) > 0 else 0.0

    table = PrettyTable()
    table.field_names = ["language", "precision", "recall", "f1-score", "support"]
    for i, label in enumerate(labels):
        metrics = report[label]
        table.add_row([
            label,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            f"{metrics['support']}"
        ])
    table.add_row(["", "", "", "", ""], divider=True)
    table.add_row([
        "micro avg",
        f"{micro_precision:.2f}",
        f"{micro_recall:.2f}",
        f"{micro_f1:.2f}",
        f"{total_samples}"
    ])
    table.add_row([
        "macro avg",
        f"{macro_precision:.2f}",
        f"{macro_recall:.2f}",
        f"{macro_f1:.2f}",
        f"{total_samples}"
    ])
    table.add_row([
        "weighted avg",
        f"{weighted_precision:.2f}",
        f"{weighted_recall:.2f}",
        f"{weighted_f1:.2f}",
        f"{total_samples}"
    ])
    table.align = "r"
    table.align[""] = "l"
    print(table)

