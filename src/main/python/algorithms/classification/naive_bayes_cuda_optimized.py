import time
import os
from datetime import datetime

import cupy as cp
import numpy as np
import pickle
import json

from python.models.text_processor import TextProcessor


class NaiveBayesCUDAOptimized:
    def __init__(self, alpha=1.0, use_gpu=True):
        """
        Khởi tạo thuật toán Naive Bayes tối ưu GPU

        Args:
            alpha: float, tham số Laplace smoothing
            use_gpu: bool, sử dụng GPU nếu có thể
        """
        self.alpha = alpha
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.classes = []
        self.class_priors = {}
        self.feature_probs_gpu = {}
        self.feature_vocab = []

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def compute_priors(self, y_idx):
        """Tính xác suất tiên nghiệm P(C) cho mỗi lớp"""
        unique_classes, counts = self.xp.unique(y_idx, return_counts=True)
        total_samples = len(y_idx)
        self.classes = unique_classes.tolist()
        for cls, count in zip(unique_classes, counts):
            cls_label = self.idx_to_label[int(cls)]
            self.class_priors[cls_label] = float(count / total_samples)


    def compute_likelihoods(self, x, y_idx, feature_names):
        """Tính xác suất đặc trưng P(F|C) cho mỗi đặc trưng và lớp"""
        self.feature_vocab = feature_names
        x_gpu = self.xp.array(x)
        y_gpu = self.xp.array(y_idx)
        for cls_idx in self.classes:
            mask = (y_gpu == cls_idx)
            class_samples = x_gpu[mask]
            feature_counts = self.xp.sum(class_samples, axis=0)
            total_features = self.xp.sum(feature_counts)
            vocab_size = len(feature_names)
            smoothed_probs = (feature_counts + self.alpha) / (total_features + self.alpha * vocab_size)
            cls_label = self.idx_to_label[int(cls_idx)]
            self.feature_probs_gpu[cls_label] = smoothed_probs


    def fit(self, x, y, feature_names):
        """Huấn luyện mô hình Naive Bayes"""
        start_time = time.time()
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        unique_labels = np.unique(y)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        y_idx = np.array([self.label_to_idx[label] for label in y])
        x = self.xp.array(x)
        y_idx = self.xp.array(y_idx)
        self.compute_priors(y_idx)
        self.compute_likelihoods(x, y_idx, feature_names)
        end_time = time.time()
        print(f"Thời gian huấn luyện: {end_time - start_time:.2f} seconds")


    def predict(self, x):
        """Tính xác suất dự đoán cho mỗi mẫu"""
        start_time = time.time()
        x_gpu = self.xp.array(x)
        if len(x_gpu.shape) == 1:
            x_gpu = x_gpu.reshape(1, -1)
        n_samples = len(x_gpu)
        n_classes = len(self.classes)
        log_prob_matrix = self.xp.zeros((n_samples, n_classes))
        for j, cls_idx in enumerate(self.classes):
            cls_label = self.idx_to_label[int(cls_idx)]
            log_posterior = self.xp.full(n_samples, self.xp.log(self.class_priors[cls_label]))
            feature_probs = self.feature_probs_gpu[cls_label]
            non_zero_mask = x_gpu > 0
            log_feature_probs = self.xp.log(feature_probs)
            log_posterior += self.xp.sum(
                x_gpu * log_feature_probs[self.xp.newaxis, :] * non_zero_mask,
                axis=1
            )
            log_prob_matrix[:, j] = log_posterior
        predicted_indices = self.xp.argmax(log_prob_matrix, axis=1)
        if self.use_gpu:
            predicted_indices = predicted_indices.get()
        if len(predicted_indices) == 1:
            result = self.idx_to_label[predicted_indices[0]]
        else:
            result = [self.idx_to_label[idx] for idx in predicted_indices]

        end_time = time.time()
        print(f"Thời gian dự đoán tối ưu (GPU): {end_time - start_time:.2f} giây")
        return result

    def save_model(self, model_path, text_processor):
        """
        Lưu mô hình vào đĩa sử dụng pickle

        Args:
            model_path: Đường dẫn lưu mô hình
            text_processor: processor đã được sử dụng để train
        """

        if os.path.exists(model_path):
            os.remove(model_path)
        feature_probs_cpu = {}
        for cls, probs in self.feature_probs_gpu.items():
            if self.use_gpu:
                feature_probs_cpu[cls] = probs.get()
            else:
                feature_probs_cpu[cls] = probs

        model_data = {
            'alpha': self.alpha,
            'use_gpu': self.use_gpu,
            'classes_': self.classes,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'class_priors': self.class_priors,
            'feature_probs': feature_probs_cpu,
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
            'use_gpu': self.use_gpu,
            'classes_': [str(cls) for cls in self.classes],
            'num_features': len(self.feature_vocab),
            'class_priors': self.class_priors,
            'num_classes': len(self.classes)
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu metadata vào {metadata_path}")


    @classmethod
    def load_model(cls, model_path, use_gpu=True) -> tuple['NaiveBayesCUDAOptimized', TextProcessor]:
        model = cls(alpha=1.0, use_gpu=use_gpu)
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model_data = data["model"]
            text_processor = data["processor"]
        model.alpha = model_data['alpha']
        model.classes = model_data['classes_']
        model.label_to_idx = model_data['label_to_idx']
        model.idx_to_label = model_data['idx_to_label']
        model.class_priors = model_data['class_priors']
        model.feature_vocab = model_data['feature_vocab']
        model.feature_probs_gpu = {}
        for cls_name, probs in model_data['feature_probs'].items():
            if model.use_gpu:
                model.feature_probs_gpu[cls_name] = cp.array(probs)
            else:
                model.feature_probs_gpu[cls_name] = probs

        print(f"Đã tải mô hình từ {model_path}")
        return model, text_processor
