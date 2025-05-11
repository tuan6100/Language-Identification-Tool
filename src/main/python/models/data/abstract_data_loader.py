from abc import ABC, abstractmethod

class BaseDataLoader(ABC):

    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass


class DataLoaderFactory:

    @staticmethod
    def create_loader(loader_type="csv", **kwargs):
        if loader_type.lower() == "csv":
            from src.main.python.models.data import CSVDataLoader
            return CSVDataLoader()
        elif loader_type.lower() == "huggingface":
            from src.main.python.models.data import HuggingFaceDataLoader
            return HuggingFaceDataLoader(**kwargs)
        else:
            raise ValueError(f"Không hỗ trợ loại loader: {loader_type}")