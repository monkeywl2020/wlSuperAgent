# -*- coding: utf-8 -*-
"""
存储模块 - 对话存储

将消息写入JSONL文件进行持久化，按 session_id + 日期 存储
"""

# 导入标准库
import json  # JSON序列化
import datetime  # 日期时间处理
import asyncio  # 异步编程
from typing import List, Dict, Optional  # 类型注解
from pathlib import Path  # 路径对象封装

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块
from ..core.directory_manager import DirectoryManager  # 目录管理器


class DialogStorage:
    """对话存储 - 将消息写入JSONL文件，按 session_id + 日期 存储"""

    def __init__(self, dir_manager: DirectoryManager):
        """
        初始化对话存储

        Args:
            dir_manager: 目录管理器
        """
        self.dir_manager = dir_manager  # 目录管理器

    def _extract_session_id(self, msg: Dict) -> str:
        """
        从消息中提取 session_id

        Args:
            msg: 消息字典

        Returns:
            session_id，如果不存在则抛出 ValueError
        """
        metadata = msg.get("metadata", {})
        session_id = metadata.get("session_id", "")
        if not session_id:
            raise ValueError("消息必须包含 session_id")
        return session_id

    def _get_date_from_msg(self, msg: Dict) -> str:
        """
        从消息中获取日期，优先使用消息自带的时间戳，否则使用当前时间

        Args:
            msg: 消息字典

        Returns:
            日期字符串 YYYY-MM-DD
        """
        # 优先使用消息的 timestamp
        timestamp = msg.get("timestamp", "")
        if timestamp:
            try:
                # 尝试解析时间戳格式: YYYY-MM-DD HH:MM:SS.mmm
                dt = datetime.datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S")
                return dt.strftime("%Y-%m-%d")
            except (ValueError, IndexError):
                pass

        # 如果没有有效时间戳，使用当前时间
        return self.dir_manager.get_today_date()

    def append_message(self, msg: Dict, session_id: Optional[str] = None, date_str: Optional[str] = None):
        """
        追加单条消息到对话记录文件（追加模式，不覆盖）

        Args:
            msg: 消息字典
            session_id: 会话ID，如果为None则从消息中提取
            date_str: 日期字符串，如果为None则从消息时间戳获取

        Raises:
            ValueError: 消息不包含 session_id
        """
        if not session_id:
            session_id = self._extract_session_id(msg)

        if not date_str:
            date_str = self._get_date_from_msg(msg)

        # 确保文件存在
        file_path = self.dir_manager.ensure_dialog_file(session_id, date_str)

        # 写入消息
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        logger.debug(f"Message appended to: {file_path}")

    def append_messages(self, messages: List[Dict], session_id: Optional[str] = None):
        """
        批量追加消息到对话记录文件（追加模式，不覆盖）

        消息按日期分组存储到不同的文件。如果消息跨日期，会存储到各自日期的文件中。

        Args:
            messages: 消息列表
            session_id: 会话ID，如果为None则从第一条消息中提取
            """
        if not messages:
            return

        if not session_id:
            session_id = self._extract_session_id(messages[0])

        # 按日期分组消息
        date_groups: Dict[str, List[Dict]] = {}
        for msg in messages:
            date_str = self._get_date_from_msg(msg)
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(msg)

        # 将每组消息写入对应日期的文件
        for date_str, msgs in date_groups.items():
            file_path = self.dir_manager.ensure_dialog_file(session_id, date_str)

            with open(file_path, "a", encoding="utf-8") as f:
                for msg in msgs:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            logger.info(f"Messages appended to: {file_path} ({len(msgs)} messages)")

    async def append_message_async(self, msg: Dict, session_id: Optional[str] = None, date_str: Optional[str] = None):
        """
        异步追加单条消息到对话记录文件（追加模式，不覆盖）

        Args:
            msg: 消息字典
            session_id: 会话ID，如果为None则从消息中提取
            date_str: 日期字符串，如果为None则从消息时间戳获取

        Raises:
            ValueError: 消息不包含 session_id
        """
        if not session_id:
            session_id = self._extract_session_id(msg)

        if not date_str:
            date_str = self._get_date_from_msg(msg)

        # 确保文件存在
        file_path = self.dir_manager.ensure_dialog_file(session_id, date_str)

        # 写入消息
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        logger.debug(f"Message appended async to: {file_path}")

    async def append_messages_async(self, messages: List[Dict], session_id: Optional[str] = None):
        """
        异步批量追加消息到对话记录文件（追加模式，不覆盖）

        消息按日期分组存储到不同的文件。如果消息跨日期，会存储到各自日期的文件中。

        Args:
            messages: 消息列表
            session_id: 会话ID，如果为None则从第一条消息中提取
        """
        if not messages:
            return

        if not session_id:
            session_id = self._extract_session_id(messages[0])

        # 按日期分组消息
        date_groups: Dict[str, List[Dict]] = {}
        for msg in messages:
            date_str = self._get_date_from_msg(msg)
            if date_str not in date_groups:
                date_groups[date_str] = []
            date_groups[date_str].append(msg)

        # 将每组消息写入对应日期的文件
        for date_str, msgs in date_groups.items():
            file_path = self.dir_manager.ensure_dialog_file(session_id, date_str)

            with open(file_path, "a", encoding="utf-8") as f:
                for msg in msgs:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            logger.info(f"Messages appended async to: {file_path} ({len(msgs)} messages)")

    def load_recent_dialogs(self, days: int = 0, session_id: str = "") -> List[Dict]:
        """
        加载最近的对话记录（同步版本）

        Args:
            days: 加载的天数，0表示只加载今天的
            session_id: 会话ID，用于过滤特定会话的消息，为空则加载所有会话

        Returns:
            对话消息列表
        """
        messages = []
        today = datetime.datetime.now()

        if days == 0:
            date_str = today.strftime("%Y-%m-%d")
            if session_id:
                file_path = self.dir_manager.get_dialog_file_path(session_id, date_str)
                logger.info(f"load_recent_dialogs ---- file_path:{file_path}")
                if file_path.exists():
                    messages.extend(self._load_from_file(file_path))
            else:
                # 加载所有会话当天的消息
                messages.extend(self._load_all_session_dialogs(date_str))
            return messages

        for i in range(days):
            date = today - datetime.timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            if session_id:
                file_path = self.dir_manager.get_dialog_file_path(session_id, date_str)
                if file_path.exists():
                    messages.extend(self._load_from_file(file_path))
            else:
                messages.extend(self._load_all_session_dialogs(date_str))

        return messages

    def _load_from_file(self, file_path: Path) -> List[Dict]:
        """
        从文件中加载消息

        Args:
            file_path: 文件路径

        Returns:
            消息列表
        """
        messages = []
        if not file_path.exists():
            return messages

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))

        logger.info(f"load_recent_dialogs ---- file_path:{file_path}, messages:{messages}")
        return messages

    def _load_all_session_dialogs(self, date_str: str) -> List[Dict]:
        """
        加载指定日期所有会话的对话记录

        Args:
            date_str: 日期字符串 YYYY-MM-DD

        Returns:
            所有会话的消息列表
        """
        messages = []
        pattern = f"*_{date_str}.jsonl"

        for file_path in self.dir_manager.get_dialog_dir().glob(pattern):
            messages.extend(self._load_from_file(file_path))

        return messages

    async def load_recent_dialogs_async(self, days: int = 0, session_id: str = "") -> List[Dict]:
        """
        异步加载最近的对话记录

        Args:
            days: 加载的天数，0表示只加载今天的
            session_id: 会话ID，用于过滤特定会话的消息，为空则加载所有会话

        Returns:
            对话消息列表
        """
        # 同步版本足够快，直接使用同步方法
        return self.load_recent_dialogs(days, session_id)

    def load_dialog_by_date(self, date_str: str, session_id: str = "") -> List[Dict]:
        """
        加载指定日期的对话记录

        Args:
            date_str: 日期字符串 YYYY-MM-DD
            session_id: 会话ID，为空则加载所有会话

        Returns:
            对话消息列表
        """
        if session_id:
            file_path = self.dir_manager.get_dialog_file_path(session_id, date_str)
            return self._load_from_file(file_path)
        else:
            return self._load_all_session_dialogs(date_str)

    async def load_dialog_by_date_async(self, date_str: str, session_id: str = "") -> List[Dict]:
        """
        异步加载指定日期的对话记录

        Args:
            date_str: 日期字符串 YYYY-MM-DD
            session_id: 会话ID，为空则加载所有会话

        Returns:
            对话消息列表
        """
        return self.load_dialog_by_date(date_str, session_id)

    def get_dialog_files(self, session_id: str = "") -> List[Path]:
        """
        获取对话记录文件列表

        Args:
            session_id: 会话ID，为空则返回所有会话的文件

        Returns:
            对话记录文件路径列表（按时间倒序）
        """
        if session_id:
            return sorted(
                self.dir_manager.get_dialog_dir().glob(f"{session_id}_*.jsonl"),
                reverse=True
            )
        else:
            return sorted(
                self.dir_manager.get_dialog_dir().glob("*.jsonl"),
                reverse=True
            )

    def delete_old_dialogs(self, days: int):
        """
        删除指定天数之前的对话记录

        Args:
            days: 保留的天数
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        deleted_count = 0

        for file_path in self.dir_manager.get_dialog_dir().glob("*.jsonl"):
            # 解析文件名: {session_id}_{YYYY-MM-DD}.jsonl
            filename = file_path.stem
            parts = filename.rsplit("_", 1)
            if len(parts) != 2:
                continue

            date_str = parts[1]
            try:
                file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleting expired dialog: {file_path.name}")
            except ValueError:
                continue

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} expired dialog files")
