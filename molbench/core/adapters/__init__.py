"""
core.adapters - 模型适配器模块
提供统一的模型接口，支持 sklearn、GNN、Transformer text 等多种模型
"""

from .base import BenchModel
from .sklearn_adapter import SklearnModel
from .gnn_adapter import BenchGNN
from .text_adapter import HFTextModel, DeepChemTextCNN
from .custom_model import CustomModel
from .smiles_to_graph import GraphConverter

# 模型适配器注册表（工厂模式）
_ADAPTER_REGISTRY = {
    'sklearn': SklearnModel,
    'gnn': BenchGNN,
    'hf_text': HFTextModel,
    'dc_text': DeepChemTextCNN,
    'custom': CustomModel,
}

def get_adapter(model_type: str, **kwargs):
    """
    工厂函数：根据模型类型获取对应的适配器
    
    Parameters
    ----------
    model_type : str
        模型类型标识，如 'sklearn', 'gnn', 'transformer'
    **kwargs
        传递给适配器的参数
    
    Returns
    -------
    BaseAdapter
        对应的模型适配器实例
    
    Example
    -------
    >>> adapter = get_adapter('gnn', model_name='GCN', hidden_dim=128)
    """
    model_type = model_type.lower()
    if model_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(_ADAPTER_REGISTRY.keys())}")
    
    adapter_class = _ADAPTER_REGISTRY[model_type]
    return adapter_class(**kwargs)

def register_adapter(name: str, adapter_class: type):
    """
    注册自定义适配器

    Parameters
    ----------
    name : str
        适配器名称
    adapter_class : type
        继承自 BaseAdapter 的类
    """
    if not issubclass(adapter_class, BenchModel):
        raise TypeError("Adapter must inherit from BaseAdapter")
    _ADAPTER_REGISTRY[name.lower()] = adapter_class

__all__ = [
    # 类
    'BenchModel',
    'SklearnModel',
    'BenchGNN',
    'HFTextModel',
    'DeepChemTextCNN',
    'CustomModel',
    'GraphConverter'
]