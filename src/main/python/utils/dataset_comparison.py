import numpy as np
from collections import defaultdict


# def jaccard_similarity(str1, str2, n=3):
#     """
#     Calculate Jaccard similarity between two strings using n-grams.
#
#     This function computes the Jaccard similarity coefficient between two strings
#     by first breaking them into character n-grams and then calculating the ratio
#     of intersection to union between these n-gram sets.
#
#     Args:
#         str1 (str): First string to compare
#         str2 (str): Second string to compare
#         n (int): Size of n-grams (default: 3 for trigrams)
#
#     Returns:
#         float: Similarity score between 0 and 1, where 1 means identical
#     """
#     # Normalize strings
#     str1 = str1.lower().strip()
#     str2 = str2.lower().strip()
#
#     # Create n-grams
#     ngrams1 = set()
#     for i in range(len(str1) - n + 1):
#         ngrams1.add(str1[i:i+n])
#
#     ngrams2 = set()
#     for i in range(len(str2) - n + 1):
#         ngrams2.add(str2[i:i+n])
#
#     # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
#     intersection = len(ngrams1.intersection(ngrams2))
#     union = len(ngrams1.union(ngrams2))
#
#     return intersection / union if union > 0 else 0
#
# def check_fuzzy_duplicates(x_train, x_test, threshold=0.8, n=2, batch_size=2000):
#     """
#     Check for fuzzy duplicates between training and testing datasets using GPU acceleration.
#
#     This function uses CuPy for GPU-accelerated operations to identify near-duplicate
#     text samples between training and test sets based on Jaccard similarity. Processing
#     is done in batches to avoid memory overload.
#
#     Args:
#         x_train (list): List of text samples in the training set
#         x_test (list): List of text samples in the test set
#         threshold (float): Similarity threshold to consider as duplicate (default: 0.8)
#         n (int): Size of n-grams for Jaccard similarity (default: 2)
#         batch_size (int): Batch size for processing (default: 1000)
#
#     Returns:
#         list: List of tuples (test_idx, train_idx, similarity, test_text, train_text)
#               containing all detected fuzzy duplicates
#     """
#     # Import CuPy for GPU acceleration
#     try:
#         import cupy as cp
#         use_gpu = True
#         print("Using GPU acceleration with CuPy")
#     except ImportError:
#         print("CuPy not available, falling back to CPU processing")
#         cp = np
#         use_gpu = False
#
#     print(f"Checking fuzzy duplicates with Jaccard (threshold = {threshold}) between train and test sets...")
#     print("Warning: This process may take a long time with large datasets!")
#
#     # Normalize texts
#     normalize = lambda text: text.strip().lower()
#
#     # Prepare data
#     norm_train = [normalize(text) for text in x_train]
#     norm_test = [normalize(text) for text in x_test]
#
#     # Initialize results list
#     fuzzy_duplicates = []
#
#     # Process in batches
#     num_batches = int(np.ceil(len(norm_test) / batch_size))
#
#     print(f"Processing {len(norm_test)} test samples in {num_batches} batches...")
#
#     start_time = time.time()
#     total_fuzzy_duplicates = 0
#
#     # Pre-compute lengths of all training samples (for faster filtering)
#     train_lengths = cp.array([len(text) for text in norm_train]) if use_gpu else np.array([len(text) for text in norm_train])
#
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(norm_test))
#
#         print(f"Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx} to {end_idx-1})...")
#         batch_start_time = time.time()
#
#         # Pre-compute test lengths for current batch (for faster filtering)
#         batch_test_lengths = cp.array([len(norm_test[i]) for i in range(start_idx, end_idx)]) if use_gpu else np.array([len(norm_test[i]) for i in range(start_idx, end_idx)])
#
#         # For each test sample in the batch, compute length ratios with all training samples
#         if use_gpu:
#             # GPU vectorized computation of length ratios
#             length_ratios = cp.zeros((end_idx - start_idx, len(norm_train)))
#             for i in range(end_idx - start_idx):
#                 test_len = batch_test_lengths[i]
#                 min_lengths = cp.minimum(test_len, train_lengths)
#                 max_lengths = cp.maximum(test_len, train_lengths)
#                 length_ratios[i] = min_lengths / max_lengths
#
#             # Get potential match indices where length ratio is above threshold
#             potential_matches = []
#             for i in range(length_ratios.shape[0]):
#                 matches = cp.where(length_ratios[i] >= threshold * 0.7)[0]
#                 potential_matches.append((i + start_idx, matches.get()))
#         else:
#             # CPU fallback
#             potential_matches = []
#             for i in range(start_idx, end_idx):
#                 test_len = len(norm_test[i])
#                 min_lengths = np.minimum(test_len, train_lengths)
#                 max_lengths = np.maximum(test_len, train_lengths)
#                 length_ratios = min_lengths / max_lengths
#                 matches = np.where(length_ratios >= threshold * 0.7)[0]
#                 potential_matches.append((i, matches))
#
#         batch_duplicates = 0
#         # Process potential matches
#         for test_idx, train_indices in potential_matches:
#             test_text = norm_test[test_idx]
#
#             for j in train_indices:
#                 train_text = norm_train[j]
#
#                 # Calculate Jaccard similarity
#                 sim = jaccard_similarity(test_text, train_text, n)
#
#                 if sim >= threshold:
#                     fuzzy_duplicates.append((test_idx, j, sim, test_text, train_text))
#                     batch_duplicates += 1
#
#         total_fuzzy_duplicates += batch_duplicates
#
#         batch_time = time.time() - batch_start_time
#         print(f"  Completed batch in {batch_time:.2f} seconds, found {batch_duplicates} duplicates")
#
#     total_time = time.time() - start_time
#
#     # Sort results by similarity (descending)
#     fuzzy_duplicates.sort(key=lambda x: x[2], reverse=True)
#
#     # Calculate ratio of test samples with fuzzy duplicates
#     duplicate_test_indices = set(dup[0] for dup in fuzzy_duplicates)
#     duplicate_ratio = len(duplicate_test_indices) / len(x_test) * 100
#
#     print(f"\nTotal time: {total_time:.2f} seconds")
#     print(f"Number of fuzzy duplicate pairs: {len(fuzzy_duplicates)}")
#     print(f"Number of test samples with duplicates: {len(duplicate_test_indices)} / {len(x_test)} ({duplicate_ratio:.2f}%)")
#
#     # Show some examples of fuzzy duplicates
#     if fuzzy_duplicates:
#         print("\nSome examples of fuzzy duplicates (sorted by similarity):")
#         for i, (test_idx, train_idx, sim, test_text, train_text) in enumerate(fuzzy_duplicates[:5]):
#             print(f"Example {i+1} (sim={sim:.4f}):")
#             print(f"  Test #{test_idx}: {test_text[:100]}..." if len(test_text) > 100 else f"  Test #{test_idx}: {test_text}")
#             print(f"  Train #{train_idx}: {train_text[:100]}..." if len(train_text) > 100 else f"  Train #{train_idx}: {train_text}")
#             print()
#
#         if len(fuzzy_duplicates) > 5:
#             print(f"... and {len(fuzzy_duplicates) - 5} other duplicate pairs")
#
#     return fuzzy_duplicates

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
        if ' ' in feature:  # Nếu có khoảng trắng, đó là bigram
            bigrams.append(feature)
        else:
            unigrams.append(feature)

    # Tính tần suất trung bình cho mỗi loại n-gram
    train_feature_freq = np.array(x_train_processed.mean(axis=0)).flatten()
    test_feature_freq = np.array(x_test_processed.mean(axis=0)).flatten()

    # Tính tần suất trung bình cho unigram và bigram
    train_unigram_indices = [feature_names.index(f) for f in unigrams]
    train_bigram_indices = [feature_names.index(f) for f in bigrams]

    test_unigram_indices = train_unigram_indices
    test_bigram_indices = train_bigram_indices

    # Tần suất trung bình
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
