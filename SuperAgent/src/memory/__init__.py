# -*- coding: utf-8 -*-
"""
文件记忆系统 (FileMemory)

参考ReMe设计理念实现的文本存储记忆功能，为Agent提供长期记忆能力
"""

# 注意：不再在顶层导入所有模块，以避免依赖问题
# 使用时直接导入需要的模块：
# from src.memory.file_memory import FileMemory
# from src.memory.core import DirectoryManager

__version__ = "2.0.0"

__all__ = [
    "FileMemory",  # 需要时从 file_memory 导入
    "Msg",  # 需要时从 src.msg 导入
]
