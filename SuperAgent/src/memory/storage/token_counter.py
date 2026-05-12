# -*- coding: utf-8 -*-
"""
存储模块 - Token计数

用于计算消息的token数量
"""

# 导入标准库
from typing import List, Union, Sequence  # 类型注解
import json  # JSON序列化

# 导入第三方库
from loguru import logger  # 日志记录

# 导入消息类（使用 src/msg 中的定义）
from src.msg import Msg, ContentBlock, ToolUseBlock  # 消息类和消息块类型


# 尝试导入tiktoken，可能不存在
try:
    import tiktoken  # Token计数
    TIKTOKEN_AVAILABLE = True  # 标记tiktoken可用
except ImportError:
    TIKTOKEN_AVAILABLE = False  # 标记tiktoken不可用
    logger.warning("tiktoken not installed, using character count approximation")


class TokenCounter:
    """Token计数器 - 用于计算消息的token数量"""

    def __init__(self, model: str = "gpt-4"):
        """
        初始化Token计数器

        Args:
            model: 模型名称，用于选择编码器
        """
        self.model = model  # 模型名称
        self.encoding = None  # 编码器对象
        
        if TIKTOKEN_AVAILABLE:
            try:
                # 尝试加载编码器
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # 如果模型不支持，使用默认编码器
                self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info(f"Token counter initialized, model: {model}")
        else:
            logger.info("Token counter initialized, using character count approximation")

    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量

        Args:
            text: 要计算的文本

        Returns:
            token数量
        """
        if not text:
            return 0
        
        if self.encoding:
            # 使用tiktoken精确计数
            return len(self.encoding.encode(text))
        else:
            # 使用简单近似：约4个字符一个token
            return len(text) // 4 + 1

    def count_messages_tokens(self, messages: List[Union[Msg, dict]]) -> int:
        """
        计算消息列表的总token数量

        Args:
            messages: 消息列表（Msg对象或字典）

        Returns:
            总token数量
        """
        total = 0
        for msg in messages:
            # 角色也有token开销
            total += 4  # role token
            
            # 获取消息内容
            content = None
            
            if isinstance(msg, Msg):
                # Msg对象：使用 get_text_content 方法
                if msg.compressed and msg.summary:
                    # 如果已压缩，使用摘要
                    content = msg.summary
                else:
                    content = msg.get_text_content()
            elif isinstance(msg, dict):
                # 字典格式
                if msg.get("compressed") and msg.get("summary"):
                    content = msg.get("summary")
                else:
                    content = msg.get("content", "")
                    # 如果是内容块列表，提取文本
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        content = " ".join(text_parts)
            
            if content:
                total += self.count_tokens(str(content))
            
            # 如果有工具调用块，添加工具调用token
            tool_use_blocks = None
            
            if isinstance(msg, Msg):
                # Msg对象：使用 get_content_blocks 方法
                tool_use_blocks = msg.get_content_blocks("tool_use")
            elif isinstance(msg, dict):
                # 字典格式
                content = msg.get("content", [])
                if isinstance(content, list):
                    tool_use_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                    
            if tool_use_blocks:
                total += self.count_tokens(json.dumps(tool_use_blocks))
        
        total += 2  # 结束token
        return total
