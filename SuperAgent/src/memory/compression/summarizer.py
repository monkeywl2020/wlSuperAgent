# -*- coding: utf-8 -*-
"""
压缩模块 - 异步摘要

生成人类可读的对话摘要
"""

# 导入标准库
import threading  # 线程操作
import datetime  # 日期时间处理
from typing import List, Optional, Any  # 类型注解

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块
from ..core.config import MemoryConfig  # 内存配置
from ..core.directory_manager import DirectoryManager  # 目录管理器
from src.msg import Msg  # 消息类（使用 src/msg 中的定义）
from ..llm import LLMProvider  # LLM接口


class Summarizer:
    """异步摘要器 - 生成人类可读的对话摘要"""

    def __init__(
        self,
        config: MemoryConfig,
        dir_manager: DirectoryManager,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        初始化异步摘要器

        Args:
            config: 记忆配置
            dir_manager: 目录管理器
            llm_provider: LLM接口实例
        """
        self.config = config  # 配置对象
        self.dir_manager = dir_manager  # 目录管理器
        self.llm_provider = llm_provider  # LLM接口
        self.summarizer_thread: Optional[threading.Thread] = None  # 摘要线程

    def _format_messages(self, messages: List[Msg]) -> str:
        """将消息列表格式化为文本"""
        messages_text = ""
        for msg in messages:
            text = msg.get_text_content() or ""
            messages_text += f"{msg.role}: {text}\n"
        return messages_text

    def summarize_async(self, messages: List[Msg], summary: str):
        """
        异步启动摘要生成

        Args:
            messages: 消息列表
            summary: 压缩后的摘要
        """
        # 在后台线程中运行摘要
        self.summarizer_thread = threading.Thread(
            target=self._summarize_worker,
            args=(messages, summary),
            daemon=True  # 守护线程，主进程退出时自动结束
        )
        self.summarizer_thread.start()
        logger.info("Async summary task started")

    def _summarize_worker(self, messages: List[Msg], summary: str):
        """
        摘要工作线程

        Args:
            messages: 消息列表
            summary: 压缩摘要
        """
        try:
            # 生成每日记忆
            self._write_daily_memory(messages, summary)

            # 生成长期记忆（如果LLM可用）
            if self.llm_provider and self.config.longterm_prompt:
                self._update_long_term_memory(messages)

            logger.info("Async summary task completed")
        except Exception as e:
            logger.error(f"Async summary task failed: {e}")

    def _write_daily_memory(self, messages: List[Msg], summary: str):
        """
        生成每日记忆摘要

        读取当天JSONL对话文件，使用LLM生成摘要，写入每日记忆MD文件

        Args:
            messages: 消息列表（备用）
            summary: 压缩摘要
        """
        # 获取今天的日期
        date_str = self.dir_manager.get_today_date()

        # 读取当天的所有JSONL对话记录文件（按 session_id + 日期）
        import json
        dialog_content = []

        # 获取所有当天文件
        dialog_dir = self.dir_manager.get_dialog_dir()
        today_files = sorted(dialog_dir.glob(f"*_{date_str}.jsonl"))

        if not today_files:
            logger.info(f"No dialog files found for {date_str}")
            return

        for dialog_file_path in today_files:
            try:
                with open(dialog_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            msg_dict = json.loads(line)
                            role = msg_dict.get("role", "")
                            content = msg_dict.get("content", "")
                            if role == "user" and content:
                                dialog_content.append(f"user: {content}")
                            elif role == "assistant" and content:
                                dialog_content.append(f"assistant: {content}")
            except Exception as e:
                logger.error(f"Failed to read dialog file {dialog_file_path}: {e}")
                continue

        if not dialog_content:
            logger.info("No dialog content to summarize")
            return

        # 构造对话文本
        messages_text = "\n".join(dialog_content)

        # 使用LLM生成摘要（如果LLM可用）
        summary_content = ""
        if self.llm_provider and self.config.summarize_prompt:
            prompt = self.config.summarize_prompt.format(messages=messages_text)
            try:
                summary_content = self.llm_provider.chat([{"role": "user", "content": prompt}])
                logger.info(f"LLM summary generated for {date_str}")
            except Exception as e:
                logger.error(f"Failed to generate LLM summary: {e}")
                summary_content = ""
        else:
            # 如果没有LLM，使用简单的格式化
            summary_content = f"今日对话摘要：\n{chr(10).join(dialog_content[-10:])}"

        # 获取今天的记忆文件路径
        memory_file_path = self.dir_manager.get_memory_file_path()

        # 构造MD文件内容
        content = f"""# {date_str} 对话摘要

## 对话要点
{summary_content}

---
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        # 写入文件
        memory_file_path.write_text(content, encoding="utf-8")
        logger.info(f"Daily memory written: {memory_file_path}")

    def _update_long_term_memory(self, messages: List[Msg]):
        """
        更新长期记忆

        Args:
            messages: 消息列表
        """
        if not self.llm_provider or not self.config.longterm_prompt:
            return

        # 构造消息文本
        messages_text = self._format_messages(messages)

        # 调用LLM提取长期记忆
        prompt = self.config.longterm_prompt.format(messages=messages_text)
        long_term_content = self.llm_provider.chat([{"role": "user", "content": prompt}])

        # 检查是否有有效内容
        if long_term_content and long_term_content.strip() and long_term_content != "无":
            # 追加到长期记忆文件
            self._append_to_long_term(long_term_content)

    def _append_to_long_term(self, content: str):
        """
        追加内容到长期记忆

        Args:
            content: 要追加的内容
        """
        memory_file = self.dir_manager.get_long_term_memory_file()

        # 读取现有内容
        existing = ""
        if memory_file.exists():
            existing = memory_file.read_text(encoding="utf-8")

        # 追加新内容（添加分隔符）
        new_content = existing + f"\n\n---\n\n{content}"

        # 写回文件
        memory_file.write_text(new_content, encoding="utf-8")
        logger.info("Long-term memory updated")

    def wait_for_completion(self):
        """等待摘要任务完成"""
        if self.summarizer_thread and self.summarizer_thread.is_alive():
            self.summarizer_thread.join()

    def summarize_and_wait(self, messages: List[Msg], summary: str):
        """
        同步执行摘要生成并等待完成

        Args:
            messages: 消息列表
            summary: 压缩摘要
        """
        self.summarize_async(messages, summary)
        self.wait_for_completion()

    def is_running(self) -> bool:
        """
        检查摘要任务是否正在运行

        Returns:
            是否正在运行
        """
        return self.summarizer_thread is not None and self.summarizer_thread.is_alive()
