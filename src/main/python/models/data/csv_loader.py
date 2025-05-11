import pandas as pd
from pathlib import Path


class CSVDataLoader:
    def __init__(self):
        self.paths = self.get_default_path()

    def load_data(self, split):
        if split == "training":
            path = self.paths[0]
        elif split == "test":
            path = self.paths[1]
        elif split == "validation":
            path = self.paths[2]
        else:
            raise ValueError(f"Unknown split: {split}")
        df = pd.read_csv(path)
        x = df['text'].tolist()
        y = df['language'].tolist()
        return x, y

    @staticmethod
    def get_default_path():
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        training_data_path = project_root / 'data' / 'training' / 'dataset.csv'
        if not training_data_path.exists():
            training_data_path = project_root / 'data' / 'training' / 'trainingset.csv'
        test_data_path = project_root / 'data' / 'test' / 'dataset.csv'
        if not test_data_path.exists():
            test_data_path = project_root / 'data' / 'test' / 'testset.csv'
        validation_data_path = project_root / 'data' / 'validation' / 'dataset.csv'
        if not validation_data_path.exists():
            validation_data_path = project_root / 'data' / 'validation' / 'validationset.csv'
        return training_data_path, test_data_path, validation_data_path