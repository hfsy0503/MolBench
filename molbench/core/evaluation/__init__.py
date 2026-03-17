"""
core.evaluation - 模型评估模块
提供评估指标计算和结果可视化
"""

from .metrics import (
   evaluation,
   EXTRA_REG_METRICS,
   EXTRA_CLF_METRICS,
   IDX_TO_METRIC,
   get_calibrated_models,
   get_score,
   select_extra_metrics
)
from .visualization import visualizer, ModelComparisonVisualizer

__all__ = [
    # 评估
    'evaluation',
    'EXTRA_REG_METRICS',
    'EXTRA_CLF_METRICS',
    'IDX_TO_METRIC',
    
    # 便捷函数
    'get_calibrated_models',
    'get_score',
    'select_extra_metrics',
    
    # 可视化
    'visualizer',
    'ModelComparisonVisualizer',
]