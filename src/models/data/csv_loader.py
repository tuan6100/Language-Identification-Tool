import pandas as pd
from pathlib import Path


class CSVDataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path

    def load_data(self, file_path=None):
        """
        Đọc dữ liệu từ file CSV

        Args:
            file_path: str or Path, đường dẫn đến file CSV (optional)

        Returns:
            x: list, danh sách văn bản
            y: list, danh sách nhãn ngôn ngữ
        """
        path = file_path or self.file_path
        if path is None:
            raise ValueError("Cần cung cấp đường dẫn file CSV")

        df = pd.read_csv(path)
        x = df['Text'].tolist()
        y = df['language'].tolist()
        return x, y

    @staticmethod
    def get_default_path():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        data_path = project_root / 'data' / 'raw' / 'dataset.csv'
        return data_path