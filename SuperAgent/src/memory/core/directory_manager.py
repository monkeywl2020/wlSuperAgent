# -*- coding: utf-8 -*-
"""
核心模块 - 目录管理

负责创建和维护记忆系统的目录结构
"""

# 导入标准库
from pathlib import Path  # 路径对象封装
import datetime  # 日期时间处理

# 导入第三方库
from loguru import logger  # 日志记录


class DirectoryManager:
    """目录管理器 - 负责创建和维护记忆系统的目录结构"""

    def __init__(self, working_dir: str):
        """
        初始化目录管理器

        Args:
            working_dir: 工作目录路径
        """
        self.working_dir = Path(working_dir)  # 转换为Path对象
        logger.info(f"=====DirectoryManager initialized with working_dir: {self.working_dir}")
        self.dialog_dir = self.working_dir / "dialog"  # 对话记录目录
        self.tool_result_dir = self.working_dir / "tool_result"  # 工具结果目录
        self.memory_dir = self.working_dir / "memory"  # 每日记忆目录
        self.memory_qa_dir = self.working_dir / "memory_qa"  # 每日记忆QA对目录
        self.memory_file = self.working_dir / "MEMORY.md"  # 长期记忆文件
        self.memory_qa_file = self.working_dir / "MEMORY_QA.md"  # 长期记忆QA对文件

    def ensure_directories(self):
        """创建所有必需的目录结构"""
        # 创建工作目录
        self.working_dir.mkdir(parents=True, exist_ok=True)
        # 创建对话记录目录
        self.dialog_dir.mkdir(parents=True, exist_ok=True)
        # 创建工具结果缓存目录
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)
        # 创建每日记忆目录
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # 创建每日记忆QA对目录
        self.memory_qa_dir.mkdir(parents=True, exist_ok=True)
        # 创建长期记忆文件（如果不存在）
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n", encoding="utf-8")
        # 创建长期记忆QA对文件（如果不存在）
        if not self.memory_qa_file.exists():
            self.memory_qa_file.write_text("# 长期记忆问答对\n\n", encoding="utf-8")
        logger.info(f"Directory structure initialized: {self.working_dir}")

    def get_today_date(self) -> str:
        """获取今天的日期字符串 YYYY-MM-DD"""
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def get_dialog_file_path(self, session_id: str, date_str: str = "") -> Path:
        """
        获取指定 session 和日期的对话记录文件路径

        Args:
            session_id: 会话ID
            date_str: 日期字符串，默认为今天

        Returns:
            对话记录文件路径，格式: dialog/{session_id}_{YYYY-MM-DD}.jsonl
        """
        if not date_str:
            date_str = self.get_today_date()
        return self.dialog_dir / f"{session_id}_{date_str}.jsonl"

    def ensure_dialog_file(self, session_id: str, date_str: str = "") -> Path:
        """
        确保对话文件存在，如果不存在则创建

        Args:
            session_id: 会话ID
            date_str: 日期字符串，默认为今天

        Returns:
            对话记录文件路径
        """
        file_path = self.get_dialog_file_path(session_id, date_str)
        # 确保文件存在（创建空文件）
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            logger.info(f"Created dialog file: {file_path}")
        return file_path

    def get_memory_file_path(self) -> Path:
        """获取今天的记忆文件路径"""
        return self.memory_dir / f"{self.get_today_date()}.md"

    def get_working_dir(self) -> Path:
        """获取工作目录路径"""
        return self.working_dir

    def get_dialog_dir(self) -> Path:
        """获取对话记录目录路径"""
        return self.dialog_dir

    def get_tool_result_dir(self) -> Path:
        """获取工具结果目录路径"""
        return self.tool_result_dir

    def get_memory_dir(self) -> Path:
        """获取每日记忆目录路径"""
        return self.memory_dir

    def get_memory_qa_dir(self) -> Path:
        """获取每日记忆QA对目录路径"""
        return self.memory_qa_dir

    def get_memory_qa_file_path(self) -> Path:
        """获取今天的记忆QA对文件路径"""
        return self.memory_qa_dir / f"{self.get_today_date()}_qa.md"

    def get_long_term_memory_file(self) -> Path:
        """获取长期记忆文件路径"""
        return self.memory_file

    def get_long_term_memory_qa_file(self) -> Path:
        """获取长期记忆QA对文件路径"""
        return self.memory_qa_file
