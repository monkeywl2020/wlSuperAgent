# -*- coding: utf-8 -*-
"""BaseAgent 和 PipelineBase 的公共基类"""
from abc import ABC
from abc import abstractmethod
from typing import Any


class Operator(ABC):
    """
    抽象基类“Operator”为实现可调用行为的类定义了一个协议。
    该类旨在通过重写的“__call__”方法进行子类化，
    该方法指定了运算符的执行逻辑。
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calling function"""
