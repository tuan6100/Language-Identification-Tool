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
        text = text.lower()
        text = re.sub(
            r'[^\w\s'
            r'\u00C0-\u024F'  # Latin Extended-A/B 
            r'\u1E00-\u1EFF'  # Latin Extended Additional 
            r'\u0E00-\u0E7F'  # Thai
            r'\u0400-\u04FF'  # Cyrillic (Russian, Ukraina...)
            r'\u0600-\u06FF'  # Arabic
            r'\u0900-\u097F'  # Devanagari (Hindi, Nepali...)
            r'\u4E00-\u9FFF'  # CJK Unified Ideographs (Chinese, Japanese, Korean)
            r'\u3040-\u309F'  # Hiragana (Japanese)
            r'\u30A0-\u30FF'  # Katakana (Japanese)
            r'\uAC00-\uD7AF'  # Hangul Syllables (Korean)
            r'\u3130-\u318F'  # Hangul Compatibility Jamo
            r'\u0590-\u05FF'  # Hebrew
            r']',
            ' ',
            text
        )
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_ngrams(self, text):
        """
        Trích xuất n-grams từ văn bản
        Args:
            text: str, văn bản đầu vào
        Returns:
            list: danh sách char-level-n-grams
        """
        ngrams = []
        cleaned_text = self.clean_text(text)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(cleaned_text) - n + 1):
                ngram = cleaned_text[i:i + n]
                if not ngram.isspace():
                    ngrams.append(ngram)

        return ngrams

    def fit_transform(self, texts):
        """
        Học vocabulary và chuyển đổi văn bản thành ma trận đặc trưng
        Args:
            texts: list, danh sách văn bản
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
        x = []
        for text in texts:
            ngrams = self.extract_ngrams(text)
            feature_vector = [0] * len(self.feature_names)

            for ngram in ngrams:
                if ngram in feature_to_idx:
                    feature_vector[feature_to_idx[ngram]] += 1

            x.append(feature_vector)

        return np.array(x)

    def transform(self, texts):
        """
        Chuyển đổi văn bản thành ma trận đặc trưng bằng vocabulary đã học

        Args:
            texts: list, danh sách văn bản

        Returns:
            numpy.array: ma trận đặc trưng
        """
        if not self.feature_names:
            self.fit_transform(texts)
            self.transform(texts)

        feature_to_idx = {feature: idx for idx, feature in enumerate(self.feature_names)}

        x = []
        for text in texts:
            ngrams = self.extract_ngrams(text)
            feature_vector = [0] * len(self.feature_names)

            for ngram in ngrams:
                if ngram in feature_to_idx:
                    feature_vector[feature_to_idx[ngram]] += 1

            x.append(feature_vector)

        return np.array(x)

    def get_feature_names(self):
        return self.feature_names
