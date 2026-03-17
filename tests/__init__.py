"""
tests - molbench 测试模块
"""

import os

# 测试数据路径
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

# 确保测试数据目录存在
os.makedirs(TEST_DATA_DIR, exist_ok=True)

__all__ = ['TEST_DATA_DIR']