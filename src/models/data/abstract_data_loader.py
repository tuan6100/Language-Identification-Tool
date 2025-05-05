from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """
    Abstract base class cho data loaders
    """

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        Phương thức trừu tượng để load dữ liệu

        Returns:
            x: list, danh sách văn bản
            y: list, danh sách nhãn ngôn ngữ
        """
        pass


class DataLoaderFactory:
    """
    Factory pattern để tạo data loader phù hợp
    """

    @staticmethod
    def create_loader(loader_type="csv", **kwargs):
        """
        Tạo data loader theo loại được chỉ định

        Args:
            loader_type: str, loại loader ('csv', 'huggingface')
            **kwargs: các tham số cho loader

        Returns:
            DataLoader: instance của data loader
        """
        if loader_type.lower() == "csv":
            from src.models.data.csv_loader import CSVDataLoader
            return CSVDataLoader(**kwargs)
        elif loader_type.lower() == "huggingface":
            from src.models.data.huggingface_loader import HuggingFaceDataLoader
            return HuggingFaceDataLoader(**kwargs)
        else:
            raise ValueError(f"Không hỗ trợ loại loader: {loader_type}")