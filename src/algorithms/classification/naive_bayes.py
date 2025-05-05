import numpy as np
from collections import defaultdict, Counter
import time

class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Khởi tạo thuật toán Naive Bayes
        Args:
            alpha: float, tham số Laplace smoothing
        """
        self.alpha = alpha
        self.classes = []
        self.class_priors = {}
        self.feature_probs = defaultdict(dict)
        self.feature_vocab = []

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

        for cls in self.classes:
            # Lấy tất cả mẫu thuộc lớp này
            class_samples = x[y == cls]

            # Đếm tần suất mỗi đặc trưng trong lớp
            feature_counts = Counter()
            total_features = 0

            for sample in class_samples:
                for i, count in enumerate(sample):
                    if count > 0:
                        feature_counts[i] += count
                        total_features += count

            # Tính xác suất với Laplace smoothing
            vocab_size = len(feature_names)
            for i, feature in enumerate(feature_names):
                count = feature_counts[i]
                self.feature_probs[cls][i] = (count + self.alpha) / (total_features + self.alpha * vocab_size)

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
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        all_probs = []

        for sample in x:
            sample_probs = {}

            for cls in self.classes:
                # Tính log-posterior: log P(C|x) = log P(C) + sum(log P(Fi|C))
                log_posterior = np.log(self.class_priors[cls])

                for i, count in enumerate(sample):
                    if count > 0 and i in self.feature_probs[cls]:
                        log_posterior += count * np.log(self.feature_probs[cls][i])

                sample_probs[cls] = log_posterior

            # Chuyển từ log-space sang probability space
            max_log_posterior = max(sample_probs.values())
            for cls in sample_probs:
                sample_probs[cls] = np.exp(sample_probs[cls] - max_log_posterior)

            # Chuẩn hóa
            total_prob = sum(sample_probs.values())
            for cls in sample_probs:
                sample_probs[cls] /= total_prob

            all_probs.append(sample_probs)

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