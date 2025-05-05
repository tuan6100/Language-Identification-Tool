import cupy as cp
import numpy as np
from collections import defaultdict, Counter
import time


class NaiveBayesCUDA:
    def __init__(self, alpha=1.0, use_gpu=True):
        """
        Khởi tạo thuật toán Naive Bayes với hỗ trợ CUDA

        Args:
            alpha: float, tham số Laplace smoothing
            use_gpu: bool, sử dụng GPU nếu có thể
        """
        self.alpha = alpha
        self.use_gpu = use_gpu and cp.cuda.is_available()

        # Chọn backend (CuPy hoặc NumPy)
        self.xp = cp if self.use_gpu else np

        self.classes = []
        self.class_priors = {}
        self.feature_probs = defaultdict(dict)
        self.feature_vocab = []

        print(f"Sử dụng {'GPU' if self.use_gpu else 'CPU'}")

    def compute_priors(self, y):
        """
        Tính xác suất tiên nghiệm P(C) cho mỗi lớp

        Args:
            y: array-like, nhãn lớp
        """
        total_samples = len(y)
        self.classes = list(set(y))

        for cls in self.classes:
            count = sum(1 for label in y if label == cls)
            self.class_priors[cls] = count / total_samples

    def compute_likelihoods(self, x, y, feature_names):
        """
        Tính xác suất đặc trưng P(F|C) cho mỗi đặc trưng và lớp

        Args:
            x: array-like, ma trận đặc trưng
            y: array-like, nhãn lớp
            feature_names: list, tên các đặc trưng
        """
        self.feature_vocab = feature_names
        x_gpu = self.xp.array(x)
        for cls in self.classes:
            mask = np.array([1 if label == cls else 0 for label in y])
            mask_gpu = self.xp.array(mask)

            class_samples = x_gpu[mask_gpu == 1]

            feature_counts = self.xp.sum(class_samples, axis=0)
            total_features = self.xp.sum(feature_counts)

            vocab_size = len(feature_names)
            smoothed_probs = (feature_counts + self.alpha) / (total_features + self.alpha * vocab_size)

            if self.use_gpu:
                smoothed_probs = smoothed_probs.get()

            for i, prob in enumerate(smoothed_probs):
                self.feature_probs[cls][i] = float(prob)

    def fit(self, x, y, feature_names):
        """
        Huấn luyện mô hình Naive Bayes

        Args:
            x: array-like, ma trận đặc trưng
            y: array-like, nhãn lớp
            feature_names: list, tên các đặc trưng
        """
        start_time = time.time()

        # Chuyển sang numpy array nếu cần
        x = np.array(x)
        y = np.array(y)

        # Tính xác suất tiên nghiệm
        self.compute_priors(y)

        # Tính xác suất đặc trưng có điều kiện
        self.compute_likelihoods(x, y, feature_names)

        end_time = time.time()
        print(f"Thời gian huấn luyện: {end_time - start_time:.2f} giây")

    def predict_proba(self, x):
        """
        Tính xác suất dự đoán cho mỗi mẫu

        Args:
            x: array-like, ma trận đặc trưng test

        Returns:
            dict: xác suất cho mỗi lớp
        """
        start_time = time.time()

        x = self.xp.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        all_probs = []
        batch_size = 10000  # Xử lý theo batch để tối ưu GPU memory

        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            batch_probs = []

            # Chuẩn bị ma trận xác suất cho tất cả classes
            prob_matrix = self.xp.zeros((len(batch), len(self.classes)))

            for j, cls in enumerate(self.classes):
                # Tính log-posterior cho toàn bộ batch
                # Initialize log_posterior as an array with the same shape as the batch
                log_posterior = self.xp.full(len(batch), self.xp.log(self.class_priors[cls]))

                # Vector hóa phép nhân
                for feature_idx in range(len(self.feature_vocab)):
                    if feature_idx in self.feature_probs[cls]:
                        feature_val = batch[:, feature_idx]
                        mask = feature_val > 0
                        log_posterior += mask * feature_val * self.xp.log(self.feature_probs[cls][feature_idx])

                prob_matrix[:, j] = log_posterior

            # Chuyển từ log-space sang probability space
            prob_matrix_exp = self.xp.exp(prob_matrix - self.xp.max(prob_matrix, axis=1, keepdims=True))

            # Chuẩn hóa
            prob_matrix_normalized = prob_matrix_exp / self.xp.sum(prob_matrix_exp, axis=1, keepdims=True)

            # Chuyển về CPU
            if self.use_gpu:
                prob_matrix_normalized = prob_matrix_normalized.get()

            # Chuyển sang dictionary format
            for row in prob_matrix_normalized:
                sample_probs = {}
                for j, cls in enumerate(self.classes):
                    sample_probs[cls] = float(row[j])
                batch_probs.append(sample_probs)

            all_probs.extend(batch_probs)

        end_time = time.time()
        print(f"Thời gian dự đoán: {end_time - start_time:.2f} giây")

        return all_probs[0] if len(x) == 1 else all_probs

    def predict(self, x):
        """
        Dự đoán lớp cho mỗi mẫu

        Args:
            x: array-like, ma trận đặc trưng test

        Returns:
            list: lớp dự đoán
        """
        probs = self.predict_proba(x)

        if isinstance(probs, dict):
            return max(probs, key=probs.get)
        else:
            return [max(prob_dict, key=prob_dict.get) for prob_dict in probs]
