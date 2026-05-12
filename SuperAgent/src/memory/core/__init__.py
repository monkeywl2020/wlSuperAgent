# -*- coding: utf-8 -*-
"""
核心模块

包含数据模型、配置管理和目录管理等核心功能
"""

# 使用 src/msg 中的消息类（不再重复定义）
from src.msg import (
    Msg,  # 消息类
    TextBlock,  # 文本块
    ThinkingBlock,  # 思考块
    ToolUseBlock,  # 工具调用块
    ToolResultBlock,  # 工具结果块
    ImageBlock,  # 图片块
    AudioBlock,  # 音频块
    VideoBlock,  # 视频块
    ContentBlock,  # 内容块联合类型
    ContentBlockTypes,  # 内容块类型字面量
)

from .config import MemoryConfig, load_config, save_config  # 配置管理
from .directory_manager import DirectoryManager  # 目录管理

# 定义模块导出列表
__all__ = [
    "Msg",  # 消息类
    "MemoryConfig",  # 配置类
    "load_config",  # 配置加载函数
    "save_config",  # 配置保存函数
    "DirectoryManager",  # 目录管理器
    # 内容块类型（从 src.msg 导出）
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "AudioBlock",
    "VideoBlock",
    "ContentBlock",
    "ContentBlockTypes",
]