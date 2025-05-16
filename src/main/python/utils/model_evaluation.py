import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

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


def custom_classification_report(confusion_matrix):
    """
    Tạo báo cáo phân loại với precision, recall và F1-score
    dựa trên ma trận nhầm lẫn.

    Returns:
        dict: A report

    Examples:
        precision    recall  f1-score   support
            en       0.99      1.00      0.99       500
            fr       0.99      1.00      1.00       500
            ja       1.00      1.00      1.00       500
            vi       1.00      1.00      1.00       500
        accuracy                           0.99     10000
        macro avg       0.99      0.99      0.99     10000
        weighted avg       0.99      0.99      0.99     10000
    """

    # TODO: Cuong
    results = {
        'class_metrics': {},
        'weighted_avg': {
            'precision': 0,
            'recall': 0,
            'f1-score': 0,
            'support': 0
        }
    }
    return results


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