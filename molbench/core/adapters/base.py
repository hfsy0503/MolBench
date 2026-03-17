from abc import ABC, abstractmethod
import numpy as np

class BenchModel(ABC):
    """
    抽象基类：只定义接口，不实现逻辑
    """
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BenchModel":
        """
        训练模型
        Args:
            X: 特征矩阵  y: 标签向量
            **kwargs: 训练参数（如 epochs, batch_size）
        Returns:
            self: 支持链式调用
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测，返回预测值"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """预测概率; 仅分类模型实现此方法"""
        return None
    
    @abstractmethod
    def get_params(self, deep=True) -> dict:
        """获取模型参数, 以供超参数优化使用；仅返回可序列化的超参数"""
        return self.__dict__
    
    @abstractmethod
    def get_task_type(self) -> str:
        """返回模型任务类型：'regressor' | 'binary' | 'multiclass' """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到指定路径"""
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str) -> "BenchModel":
        """从指定路径加载模型"""
        raise NotImplementedError