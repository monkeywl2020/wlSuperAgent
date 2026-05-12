# -*- coding: utf-8 -*-
from functools import wraps
from loguru import logger
import asyncio
from typing import Callable, TypeVar, Any, Tuple, Union,Optional

T = TypeVar('T')

#---------------------------------------------------
# 这个是agent容灾使用的，重试该函数的装饰符号
#---------------------------------------------------
def auto_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    retryable_errors: Union[Tuple[Exception, ...], Exception] = (Exception,),
    on_retry_failure: Optional[Callable[[Any], None]] = None
):
    """
    自动重试装饰器（异步版本）
    
    参数：
        max_retries: 最大重试次数（默认3次）
        base_delay: 基础延迟时间（秒，默认0.5）
        retryable_errors: 可重试的异常类型（默认所有异常）
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f) #@wraps(f) 确保函数签名（包括参数名等元信息）不被修改
        async def wrapper(self, *args, **kwargs) -> T: # 这里捕获所有参数 *args, **kwargs 会完整保存所有位置参数和关键字参数
            last_exception = None
            for attempt in range(max_retries):
                try:
                    # 执行装饰符装饰的 函数
                    return await f(self, *args, **kwargs) # 每次重试时都使用完全相同的 args 和 kwargs 调用原函数
                except retryable_errors as e:# 如果异常是retryable_errors 设置的异常，则开始重试
                    last_exception = e
                    if attempt < max_retries - 1:  # 不是最后一次尝试
                        delay = base_delay * (2 ** attempt) # 重试时间重置
                        logger.warning(
                            f"Attempt {attempt+1}/{max_retries} failed, retrying in {delay:.1f}s... "
                            f"(Error: {str(e)})"
                        )
                        await asyncio.sleep(delay) # 延迟后再重试
           
            logger.error(f"All {max_retries} attempts failed")
            # 如果尝试都失败了
            if on_retry_failure:
                on_retry_failure(self) # 调用切换到备用的函数

            raise last_exception if last_exception else RuntimeError("Unknown error")
        return wrapper
    return decorator
