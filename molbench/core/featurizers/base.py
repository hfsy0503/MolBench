from abc import ABC, abstractmethod
import numpy as np

class BaseFeaturizer(ABC):
    """所有描述符必须实现 .transform(smiles_list) -> np.ndarray"""
    @abstractmethod
    def transform(self, smiles_list): ...