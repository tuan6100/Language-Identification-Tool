import numpy as np
from collections import defaultdict

def check_exact_duplicates(x_train, x_test):
    """
    Kiểm tra trùng lặp chính xác giữa các câu trong tập train và test
    Args:
        x_train: Danh sách các câu trong tập train
        x_test: Danh sách các câu trong tập test
    Returns:
        duplicates: Danh sách các câu trùng lặp
        duplicate_indices: Dictionary với key là chỉ số trong tập test và value là danh sách chỉ số trong tập train
    """
    normalize = lambda text: text.strip().lower()
    train_texts = {}
    for i, text in enumerate(x_train):
        norm_text = normalize(text)
        if norm_text in train_texts:
            train_texts[norm_text].append(i)
        else:
            train_texts[norm_text] = [i]
    duplicates = []
    duplicate_indices = defaultdict(list)
    duplicate_count = 0
    for i, text in enumerate(x_test):
        norm_text = normalize(text)
        if norm_text in train_texts:
            duplicates.append(text)
            duplicate_indices[i] = train_texts[norm_text]
            duplicate_count += 1
    duplicate_ratio = duplicate_count / len(x_test) * 100
    print(f"Số lượng câu trùng lặp: {duplicate_count} / {len(x_test)} ({duplicate_ratio:.2f}%)")
    if duplicates:
        print("\nMột số ví dụ câu trùng lặp:")
        for i, dup_text in enumerate(duplicates[:20]):
            print(f"Ví dụ {i+1}: {dup_text[:100]}..." if len(dup_text) > 100 else f"Ví dụ {i+1}: {dup_text}")
        if len(duplicates) > 20:
            print(f"... và {len(duplicates) - 20} câu trùng lặp khác")
    return duplicates, duplicate_indices


def compare_ngram_structure(x_train_processed, x_test_processed, feature_names):
    """
    So sánh cấu trúc n-gram giữa tập train và tập test

    Args:
        x_train_processed: Dữ liệu train đã xử lý
        x_test_processed: Dữ liệu test đã xử lý
        feature_names: Tên các đặc trưng
    """
    # Phân loại đặc trưng theo độ dài n-gram
    unigrams = []
    bigrams = []

    for feature in feature_names:
        if ' ' in feature:
            bigrams.append(feature)
        else:
            unigrams.append(feature)
    train_feature_freq = np.array(x_train_processed.mean(axis=0)).flatten()
    test_feature_freq = np.array(x_test_processed.mean(axis=0)).flatten()
    train_unigram_indices = [feature_names.index(f) for f in unigrams]
    train_bigram_indices = [feature_names.index(f) for f in bigrams]
    test_unigram_indices = train_unigram_indices
    test_bigram_indices = train_bigram_indices
    train_unigram_freq = np.mean(train_feature_freq[train_unigram_indices]) if train_unigram_indices else 0
    train_bigram_freq = np.mean(train_feature_freq[train_bigram_indices]) if train_bigram_indices else 0
    test_unigram_freq = np.mean(test_feature_freq[test_unigram_indices]) if test_unigram_indices else 0
    test_bigram_freq = np.mean(test_feature_freq[test_bigram_indices]) if test_bigram_indices else 0
    print(f"Số lượng unigram: {len(unigrams)}")
    print(f"Số lượng bigram: {len(bigrams)}")
    print("\nTần suất trung bình của n-gram:")
    print(f"Unigram - Train: {train_unigram_freq:.6f}, Test: {test_unigram_freq:.6f}, "
          f"Khác biệt: {abs(train_unigram_freq - test_unigram_freq):.6f}")
    print(f"Bigram - Train: {train_bigram_freq:.6f}, Test: {test_bigram_freq:.6f}, "
          f"Khác biệt: {abs(train_bigram_freq - test_bigram_freq):.6f}")
