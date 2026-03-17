"""
molbench.configs - 配置管理模块
提供配置解析、生成和验证功能

1. 生成配置
python -m molbench --generate-config basic
输出: molbench_ESOL.yaml

2. 编辑配置（手动或使用脚本）

3. 运行配置
python -m molbench --config molbench_ESOL.yaml
或保持原有命令行模式:
python -m molbench --dataset ESOL --model RandomForest GCN
"""

from .cfg_parser import ConfigParser
from .cfg_generator import ConfigGenerator

__all__ = [
    # 核心类
    'ConfigParser',
    'ConfigGenerator',
    
    # 便捷函数
    'load_config',
    'generate_config',
]


# 便捷函数（一行调用）
def load_config(path: str):
    """加载配置文件"""
    return ConfigParser.load_for_runner(path)

def generate_config(template: str = 'basic', output: str = None, **kwargs):
    """生成配置文件"""
    if kwargs:
        # 从参数生成
        return ConfigGenerator.from_command_line(**kwargs, output_path=output)
    # 交互式生成
    return ConfigGenerator.interactive(template=template, output_path=output)