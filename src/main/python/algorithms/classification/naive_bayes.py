import json
import pickle
from datetime import datetime

import numpy as np
from collections import defaultdict, Counter
import time
import os


class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Khởi tạo thuật toán Naive Bayes
        Args:
            alpha: float, tham số Laplace smoothing
        """
        self.alpha = alpha
        self.classes_ = []
        self.class_priors = {}
        self.feature_probs = defaultdict(dict)
        self.feature_vocab = []
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def compute_priors(self, y):
        """
        Tính xác suất tiên nghiệm P(Ci) cho mỗi lớp

        Args:
            y: array-like, nhãn lớp
        """
        total_samples = len(y)
        self.classes_ = list(set(y))

        for cls in self.classes_:
            count = sum(1 for label in y if label == cls)
            self.class_priors[cls] = count / total_samples

    def compute_likelihoods(self, x, y, feature_names):
        """
        Tính xác suất đặc trưng P(Tj|Ci ) cho mỗi đặc trưng và lớp

        Args:
            x: array-like, ma trận đặc trưng
            y: array-like, nhãn lớp
            feature_names: list, tên các đặc trưng
        """
        self.feature_vocab = feature_names

        for cls in self.classes_:
            class_samples = x[y == cls]

            feature_counts = Counter()
            total_features = 0

            for sample in class_samples:
                for i, count in enumerate(sample):
                    if count > 0:
                        feature_counts[i] += count
                        total_features += count

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
        x = np.array(x)
        y = np.array(y)
        self.compute_priors(y)
        self.compute_likelihoods(x, y, feature_names)
        end_time = time.time()
        print(f"Thời gian huấn luyện: {end_time - start_time:.2f} giây")

    def predict(self, x):
        """
        Tính xác suất dự đoán cho mỗi mẫu
        Args:
            x: array-like, ma trận đặc trưng test
        Returns:
            list: lớp dự đoán
        """
        start_time = time.time()
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        all_log_probs = []
        for sample in x:
            sample_probs = {}
            for cls in self.classes_:
                # Tính log-posterior: log P(C|x) = log P(C) + sum(log P(Ti|C))
                log_posterior = np.log(self.class_priors[cls])
                for i, count in enumerate(sample):
                    if count > 0 and i in self.feature_probs[cls]:
                        log_posterior += count * np.log(self.feature_probs[cls][i])
                sample_probs[cls] = log_posterior
            all_log_probs.append(sample_probs)

        end_time = time.time()
        print(f"Thời gian dự đoán: {end_time - start_time:.2f} giây")
        return [max(prob_dict, key=prob_dict.get) for prob_dict in all_log_probs]

    def save_model(self, model_path, text_processor):
        """
        Lưu mô hình vào đĩa sử dụng pickle

        Args:
            model_path: Đường dẫn lưu mô hình
            text_processor: processor đã được sử dụng để train
        """

        if os.path.exists(model_path):
            os.remove(model_path)
        model_data = {
            'alpha': self.alpha,
            'classes_': self.classes_,
            'class_priors': self.class_priors,
            'feature_probs': dict(self.feature_probs),
            'feature_vocab': self.feature_vocab
        }
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                "model": model_data,
                "processor": text_processor
            }, f)
        print(f"Đã lưu mô hình vào {model_path}")
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        metadata = {
            'published at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'alpha': self.alpha,
            'model_type': 'NaiveBayes',
            'classes_': self.classes_,
            'num_features': len(self.feature_vocab),
            'class_priors': self.class_priors,
            'num_classes': len(self.classes_)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu metadata vào {metadata_path}")

    @classmethod
    def load_model(cls, model_path):
        model = cls()
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model_data = data["model"]
            text_processor = data["processor"]
        model.alpha = model_data['alpha']
        model.classes_ = model_data['classes_']
        model.class_priors = model_data['class_priors']
        model.feature_vocab = model_data['feature_vocab']
        model.feature_probs = defaultdict(dict)
        for cls_name, probs in model_data['feature_probs'].items():
            model.feature_probs[cls_name] = probs
        print(f"Đã tải mô hình từ {model_path}")
        return model, text_processor
