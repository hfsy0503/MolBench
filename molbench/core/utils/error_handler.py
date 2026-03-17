"""
错误处理模块
用法：
    from utils.error_handler import safe_execute, ErrorHandler
    
    # 作为装饰器
    @safe_execute(default_return=None)
    def risky_function():
        return 1/0
    
    # 作为上下文管理器
    with ErrorHandler("数据加载失败"):
        data = load_data()
"""
import functools
import traceback
import time
from typing import Optional, Callable, Any, Tuple, Type
from .logger import logger

class ErrorHandler:
    """错误处理器，可用作上下文管理器"""
    
    def __init__(self, msg: str = "", raise_error: bool = False, default: Any = None, 
                 catch_types: Optional[Tuple[Type[Exception], ...]] = None):
        self.msg = msg
        self.raise_error = raise_error
        self.default = default
        self.catch_types = catch_types or (Exception,)  # 捕获所有异常类型
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and issubclass(exc_type, self.catch_types):
            error_msg = f"{self.msg}: {exc_type.__name__} - {exc_val}" if self.msg else f"{exc_type.__name__}: {exc_val}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            if self.raise_error:
                return False  # 重新抛出异常
            return True  # 抑制异常
        elif exc_val is not None:
            return False
        return False


def safe_execute(default_return=None, log_error=True):
    """
    装饰器：安全执行函数，出错时返回默认值
    用法：
        @safe_execute(default_return=[])
        def get_data():
            return risky_operation()
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"函数 {func.__name__} 执行失败: {e}")
                    logger.debug(traceback.format_exc())
                return default_return
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    装饰器：失败重试
    用法：
        @retry(max_attempts=3)
        def unstable_network_call():
            return requests.get(...)
    """
    # 参数校验
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError(f"max_attempts 必须 >= 1，当前: {max_attempts}")
    if delay < 0:
        raise ValueError(f"delay 必须 >= 0，当前: {delay}")
    if backoff < 1:
        raise ValueError(f"backoff 必须 >= 1，当前: {backoff}")
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"重试 {max_attempts} 次后仍然失败: {e}")
                        raise
                    logger.warning(f"第 {attempt + 1} 次尝试失败，{current_delay:.1f}秒后重试: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator