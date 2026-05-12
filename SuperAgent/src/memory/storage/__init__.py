# -*- coding: utf-8 -*-
"""
存储模块

包含Token计数、对话存储等存储相关功能
"""

# 导出存储组件
from .token_counter import TokenCounter  # Token计数
from .dialog_storage import DialogStorage  # 对话存储

# 定义模块导出列表
__all__ = [
    "TokenCounter",  # Token计数器
    "DialogStorage",  # 对话存储
]
