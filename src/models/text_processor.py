import re
import numpy as np
from collections import Counter

class TextProcessor:
    def __init__(self, ngram_range=(1, 2), max_features=500):
        """
        Khởi tạo bộ xử lý văn bản

        Args:
            ngram_range: tuple, (min_n, max_n) cho n-grams
            max_features: int, số lượng đặc trưng tối đa
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.feature_names = []
        self.feature_counts = {}

    def clean_text(self, text):
        """
        Làm sạch văn bản

        Args:
            text: str, văn bản đầu vào

        Returns:
            str: văn bản đã được làm sạch
        """
        if not isinstance(text, str):
            return ""

        # Chuyển về chữ thường
        text = text.lower()

        # Giữ lại chữ cái, số và ký tự đặc biệt của một số ngôn ngữ
        text = re.sub(r'[^a-zA-Zà-ÿ0-9\u0E00-\u0E7F\u4E00-\u9FFF\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3130-\u318F\s]', ' ', text)

        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_ngrams(self, text):
        """
        Trích xuất n-grams từ văn bản

        Args:
            text: str, văn bản đầu vào

        Returns:
            list: danh sách n-grams
        """
        ngrams = []
        cleaned_text = self.clean_text(text)

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(cleaned_text) - n + 1):
                ngram = cleaned_text[i:i+n]
                if not ngram.isspace():
                    ngrams.append(ngram)

        return ngrams

    def fit_transform(self, texts, labels=None):
        """
        Học vocabulary và chuyển đổi văn bản thành ma trận đặc trưng

        Args:
            texts: list, danh sách văn bản
            labels: list, nhãn tương ứng (không bắt buộc)

        Returns:
            numpy.array: ma trận đặc trưng
        """
        # Đếm tất cả n-grams
        total_ngrams = Counter()
        for text in texts:
            ngrams = self.extract_ngrams(text)
            total_ngrams.update(ngrams)

        # Chọn top features
        if self.max_features:
            most_common = total_ngrams.most_common(self.max_features)
            self.feature_names = [item[0] for item in most_common]
        else:
            self.feature_names = list(total_ngrams.keys())

        # Tạo mapping từ feature đến index
        feature_to_idx = {feature: idx for idx, feature in enumerate(self.feature_names)}

        # Chuyển đổi văn bản thành ma trận
        X = []
        for text in texts:
            ngrams = self.extract_ngrams(text)
            feature_vector = [0] * len(self.feature_names)

            for ngram in ngrams:
                if ngram in feature_to_idx:
                    feature_vector[feature_to_idx[ngram]] += 1

            X.append(feature_vector)

        return np.array(X)

    def transform(self, texts):
        """
        Chuyển đổi văn bản thành ma trận đặc trưng bằng vocabulary đã học

        Args:
            texts: list, danh sách văn bản

        Returns:
            numpy.array: ma trận đặc trưng
        """
        if not self.feature_names:
            raise ValueError("Cần chạy fit_transform trước khi sử dụng transform")

        feature_to_idx = {feature: idx for idx, feature in enumerate(self.feature_names)}

        X = []
        for text in texts:
            ngrams = self.extract_ngrams(text)
            feature_vector = [0] * len(self.feature_names)

            for ngram in ngrams:
                if ngram in feature_to_idx:
                    feature_vector[feature_to_idx[ngram]] += 1

            X.append(feature_vector)

        return np.array(X)

    def get_feature_names(self):
        """
        Lấy danh sách tên đặc trưng

        Returns:
            list: danh sách tên đặc trưng
        """
        return self.feature_names