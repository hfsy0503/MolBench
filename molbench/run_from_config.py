#!/usr/bin/env python
"""
molbench/run_from_config.py - 配置模式快捷入口
实际调用 cli.py 保持统一
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python run_from_config.py <config.yaml>")
    print("或: python -m molbench --config <config.yaml>")
    sys.exit(1)

# 直接调用 cli
from molbench.cli import main as cli_main

# 模拟命令行参数
sys.argv = ['molbench', '--config', sys.argv[1]]
if len(sys.argv) > 2:
    sys.argv.extend(sys.argv[2:])

cli_main()