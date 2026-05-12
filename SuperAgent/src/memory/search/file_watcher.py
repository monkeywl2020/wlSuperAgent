# -*- coding: utf-8 -*-
"""
搜索模块 - 文件监视

监控记忆目录文件变化
"""

# 导入标准库
from typing import Callable, Optional, Dict  # 类型注解
import time  # 时间操作
import threading  # 线程操作
from pathlib import Path  # 路径对象封装

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块
from ..core.directory_manager import DirectoryManager  # 目录管理器


class FileWatcher:
    """文件监视器 - 监控记忆目录变化"""

    def __init__(
        self,
        dir_manager: DirectoryManager,
        callback: Optional[Callable] = None
    ):
        """
        初始化文件监视器

        Args:
            dir_manager: 目录管理器
            callback: 文件变化回调函数
        """
        self.dir_manager = dir_manager  # 目录管理器
        self.callback = callback  # 回调函数
        self.last_modified: Dict[str, float] = {}  # 上次修改时间记录

    def check_changes(self):
        """检查文件变化"""
        changed = False

        # 检查记忆目录
        memory_dir = self.dir_manager.get_memory_dir()
        if memory_dir.exists():
            for file_path in memory_dir.glob("*.md"):
                mtime = file_path.stat().st_mtime
                last_mtime = self.last_modified.get(str(file_path), 0)

                if mtime > last_mtime:
                    self.last_modified[str(file_path)] = mtime
                    changed = True

        # 检查长期记忆文件
        memory_file = self.dir_manager.get_long_term_memory_file()
        if memory_file.exists():
            mtime = memory_file.stat().st_mtime
            last_mtime = self.last_modified.get(str(memory_file), 0)

            if mtime > last_mtime:
                self.last_modified[str(memory_file)] = mtime
                changed = True

        # 如果有变化，触发回调
        if changed and self.callback:
            self.callback()

    def start_watching(self, interval: float = 1.0):
        """
        开始定期检查文件变化

        Args:
            interval: 检查间隔（秒）
        """

        def watch_loop():
            """监视循环"""
            while True:
                self.check_changes()
                time.sleep(interval)

        # 启动守护线程
        watch_thread = threading.Thread(target=watch_loop, daemon=True)
        watch_thread.start()
        logger.info(f"File watcher started, interval: {interval}s")

    def stop_watching(self):
        """停止文件监视（需要外部配合实现）"""
        logger.info("File watcher stopped")

    def reset(self):
        """重置文件监视状态"""
        self.last_modified = {}
        logger.info("File watcher state reset")
