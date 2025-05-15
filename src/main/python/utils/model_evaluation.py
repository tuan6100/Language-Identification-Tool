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
    return 0
    # TODO: Cuong


def confusion_matrix(y_true, y_pred, labels):
    return 0
    # TODO: Tuan Anh