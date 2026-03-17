"""
日志配置模块
用法：
    from utils.logger import logger
    logger.info("训练开始")
    logger.error("出错了")
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 配置根日志器
def setup_logger(name="molbench", level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_format)
    logger.addHandler(console)
    
    # 文件handler - 使用 RotatingFileHandler
    log_path = log_dir / (log_file or f"{name}.log")

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 单个文件最大 10MB
        backupCount=5,          # 保留5个备份
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# 默认logger
logger = setup_logger()

# 快捷函数
def get_logger(name):
    return setup_logger(name)