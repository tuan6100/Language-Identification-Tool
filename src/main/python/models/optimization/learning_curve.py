import os

from matplotlib import pyplot as plt
from sklearn.model_selection import LearningCurveDisplay

from python.algorithms.classification.naive_bayes import NaiveBayes
from python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from python.models.data import DataLoaderFactory
from python.models.text_processor import TextProcessor


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from random import shuffle

def stratified_kfold_split(x, y, k=5, seed=42):
    """
    Tự chia K-Fold có giữ phân phối lớp (stratified)
    Trả về list các (train_indices, val_indices) cho mỗi fold
    """
    np.random.seed(seed)
    y = np.array(y)
    x = np.array(x)
    class_indices = defaultdict(list)

    # Gom index theo từng lớp
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    # Shuffle index trong từng lớp
    for label in class_indices:
        shuffle(class_indices[label])

    # Chia mỗi lớp thành k phần
    folds = [[] for _ in range(k)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    # Tạo các tập (train_index, val_index)
    split_indices = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        split_indices.append((train_idx, val_idx))

    return split_indices


def learning_curve(model, x, y, feature_names):
    LearningCurveDisplay.from_estimator(
        estimator=model,
        X=x,
        y=y,
        fit_params={'feature_names': feature_names},
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        cv=5,
        scoring="accuracy",
        score_type="both",
        n_jobs=8,
        std_display_style='fill_between',
    )
    plt.title("Learning Curve - MultinomialNB")
    plt.grid(True)
    plt.show()


def main():
    loader = DataLoaderFactory.create_loader(
        'huggingface',
        dataset_name='papluca/language-identification',
    )
    model = NaiveBayesCUDAOptimized(alpha=0.001, use_gpu=True)
    processor = TextProcessor(ngram_range=(1, 3), max_features=5000)
    x_train, y_train = loader.load_data(loader.dataset_name, 'train')
    x_train_processed = processor.fit_transform(x_train)
    learning_curve(model, x_train_processed , y_train, processor.get_feature_names())


if __name__ == '__main__':
    main()
