from datasets import load_dataset


class HuggingFaceDataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self, dataset_name = None, split = None):
        name = dataset_name or self.dataset_name
        split = split
        dataset = load_dataset(name)
        data = dataset[split]
        x = self._extract_text(data)
        y = self._extract_labels(data)
        return list(x), list(y)

    def _extract_text(self, data):
        text_columns = ['text', 'sentence', 'content', 'Text']
        for col in text_columns:
            if col in data.column_names:
                return data[col]

        raise ValueError(f"Không tìm thấy cột văn bản. Các cột có sẵn: {data.column_names}")

    def _extract_labels(self, data):
        label_columns = ['labels', 'label', 'language', 'lang']
        for col in label_columns:
            if col in data.column_names:
                labels = data[col]
                if hasattr(data.features[col], 'names'):
                    language_names = data.features[col].names
                    return [language_names[label] for label in labels]
                return labels
        raise ValueError(f"Không tìm thấy cột nhãn. Các cột có sẵn: {data.column_names}")

    @staticmethod
    def list_datasets():
        popular_datasets = [
            "papluca/language-identification",
        ]
        return popular_datasets