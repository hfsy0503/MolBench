"""
molbench.core - 分子机器学习基准测试核心模块
"""

# 版本信息
__version__ = "0.1.0"

from .data import load_file, load_data, select_task_columns, standardization, split_data
from .featurizers import get_featurizer
from .evaluation import select_extra_metrics, visualizer
from .adapters import CustomModel, SklearnModel, BenchGNN, HFTextModel, DeepChemTextCNN
from .utils import select_task_type, UnifiedModelSelector, optimization, show_results

__all__ = [
    # 数据相关
    'load_file', 
    'load_data', 
    'select_task_columns', 
    'select_task_type',
    'standardization', 
    'split_data',
    
    # 特征化
    'get_featurizer',
    
    # 模型调用与训练
    'UnifiedModelSelector', 
    'optimization', 

    # 评估
    'select_extra_metrics',
    'show_results',
    'visualizer',
    
    # 适配器
    'CustomModel',
    'SklearnModel', 
    'BenchGNN', 
    'HFTextModel', 
    'DeepChemTextCNN', 
]