# -*- coding: utf-8 -*-
"""
压缩模块 - 上下文压缩

压缩历史消息并生成摘要
"""

# 导入标准库
from typing import List, Tuple, Optional  # 类型注解
import json  # JSON序列化

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块和存储模块
from ..core.config import MemoryConfig  # 内存配置
from src.msg import Msg  # 消息类（使用 src/msg 中的定义）
from ..storage.token_counter import TokenCounter  # Token计数器


class Compactor:
    """上下文压缩器 - 压缩历史消息并生成摘要"""

    def __init__(self, config: MemoryConfig, token_counter: TokenCounter):
        """
        初始化上下文压缩器

        Args:
            config: 记忆配置
            token_counter: Token计数器
        """
        self.config = config  # 配置对象
        self.token_counter = token_counter  # Token计数器

    def should_compact(self, messages: List[Msg]) -> bool:
        """
        判断是否需要压缩消息

        Args:
            messages: 消息列表

        Returns:
            是否需要压缩
        """
        # 计算当前token数量
        total_tokens = self.token_counter.count_messages_tokens(messages)
        # 判断是否超过阈值
        threshold = self.config.max_input_length * self.config.compact_ratio
        should = total_tokens > threshold
        if should:
            logger.info(f"Compression triggered: {total_tokens} tokens > {threshold} tokens")
        return should

    def compact_messages(
        self,
        messages: List[Msg],
        llm_compact_func=None
    ) -> Tuple[List[Msg], Optional[str]]:
        """
        压缩消息列表

        Args:
            messages: 消息列表
            llm_compact_func: LLM压缩函数，如果为None则使用简单压缩

        Returns:
            (压缩后的消息列表, 摘要文本)
        """
        if not messages:
            return [], None

        # 分离已压缩和未压缩的消息
        uncompressed = [m for m in messages if not m.compressed]
        already_compressed = [m for m in messages if m.compressed]

        if not uncompressed:
            return messages, None

        # 如果有LLM压缩函数，使用LLM压缩
        if llm_compact_func:
            summary = llm_compact_func(uncompressed)
        else:
            # 否则使用简单压缩：提取关键信息
            summary = self._simple_compact(uncompressed)

        # 将未压缩消息标记为已压缩
        for msg in uncompressed:
            msg.compressed = True
            msg.summary = summary

        # 合并已压缩的消息和新标记的消息
        result = already_compressed + uncompressed

        logger.info(f"Compression completed: {len(uncompressed)} messages -> summary ({len(summary)} chars)")
        return result, summary

    def _simple_compact(self, messages: List[Msg]) -> str:
        """
        简单压缩方法 - 提取消息关键信息

        Args:
            messages: 要压缩的消息列表

        Returns:
            压缩后的摘要
        """
        # 提取关键信息
        user_msgs = [m.get_text_content() for m in messages if m.role == "user"]
        assistant_msgs = [m.get_text_content() for m in messages if m.role == "assistant"]

        # 过滤掉None
        user_msgs = [m for m in user_msgs if m]
        assistant_msgs = [m for m in assistant_msgs if m]

        # 生成简单摘要
        summary_parts = []
        if user_msgs:
            summary_parts.append(f"用户消息数: {len(user_msgs)}")
            summary_parts.append(f"最新用户消息: {user_msgs[-1][:100]}...")
        if assistant_msgs:
            summary_parts.append(f"助手回复数: {len(assistant_msgs)}")
            summary_parts.append(f"最新助手回复: {assistant_msgs[-1][:100]}...")

        return "\n".join(summary_parts)

    def incremental_compact(
        self,
        messages: List[Msg],
        existing_summary: str,
        llm_compact_func=None
    ) -> Tuple[List[Msg], str]:
        """
        增量压缩 - 合并已有摘要和新消息

        Args:
            messages: 消息列表
            existing_summary: 已有的摘要
            llm_compact_func: LLM压缩函数

        Returns:
            (压缩后的消息列表, 新的摘要)
        """
        # 构造包含摘要的上下文
        context_with_summary = [Msg(
            role="system",
            content=f"之前的对话摘要:\n{existing_summary}"
        )]

        # 添加所有消息
        context_with_summary.extend(messages)

        # 压缩
        return self.compact_messages(context_with_summary, llm_compact_func)

    def get_compression_ratio(self, original_tokens: int, compressed_tokens: int) -> float:
        """
        计算压缩率

        Args:
            original_tokens: 原始token数
            compressed_tokens: 压缩后token数

        Returns:
            压缩率（0-1之间）
        """
        if original_tokens == 0:
            return 0.0
        return 1.0 - (compressed_tokens / original_tokens)
