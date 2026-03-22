"""MolBench - 分子机器学习基准测试框架"""
__version__ = '0.1.0'

from .cli import main as run_cli
from .core.runner_engine import run_benchmark

__all__ = ['run_benchmark', 'run_cli']

def run(config):
    """
    支持库调用的包装
    """
    return run_benchmark(config)