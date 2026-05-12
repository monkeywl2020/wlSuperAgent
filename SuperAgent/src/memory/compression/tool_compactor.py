# -*- coding: utf-8 -*-
"""
压缩模块 - 工具结果压缩

将超长工具输出写入文件并替换为引用
"""

# 导入标准库
from pathlib import Path  # 路径对象封装
import datetime  # 日期时间处理

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块
from ..core.config import MemoryConfig  # 内存配置
from ..core.directory_manager import DirectoryManager  # 目录管理器


class ToolResultCompactor:
    """工具结果压缩器 - 将超长工具输出写入文件并替换为引用"""

    def __init__(self, config: MemoryConfig, dir_manager: DirectoryManager):
        """
        初始化工具结果压缩器

        Args:
            config: 记忆配置
            dir_manager: 目录管理器
        """
        self.config = config  # 配置对象
        self.dir_manager = dir_manager  # 目录管理器
        self.tool_result_dir = dir_manager.get_tool_result_dir()  # 工具结果目录

    def should_compact(self, content: str) -> bool:
        """
        判断是否需要压缩工具结果

        Args:
            content: 工具结果内容

        Returns:
            是否需要压缩
        """
        need_compact = len(content) > self.config.tool_result.threshold
        logger.info(f"should_compact threshold: {self.config.tool_result.threshold} - length:{len(content)} - need_compact:{need_compact}")
        return need_compact

    def compact(self, tool_call_id: str, content: str) -> str:
        """
        压缩工具结果，将内容写入文件

        Args:
            tool_call_id: 工具调用ID
            content: 工具结果内容

        Returns:
            文件引用路径
        """
        logger.info(f"compact tool_call_id: {tool_call_id} - length:{len(content)}")
        # 生成唯一文件名
        file_name = f"{tool_call_id}.txt"
        file_path = self.tool_result_dir / file_name

        # 写入文件
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Tool result compacted to file: {file_path}")

        # 返回相对路径引用
        return str(file_path)

    def cleanup_old_results(self):
        """清理超过保留期限的工具结果文件"""
        retention_days = self.config.tool_result.retention_days  # 获取保留天数
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=retention_days)  # 计算截止时间

        cleaned_count = 0  # 清理计数

        # 遍历工具结果目录
        for file_path in self.tool_result_dir.glob("*.txt"):
            # 获取文件修改时间
            mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            # 如果文件过旧，删除
            if mtime < cutoff_time:
                file_path.unlink()
                cleaned_count += 1
                logger.info(f"Cleaning expired tool results: {file_path.name}")

        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} expired tool result files")

    def load_from_reference(self, file_path: str) -> str:
        """
        从文件引用加载工具结果

        Args:
            file_path: 文件路径

        Returns:
            文件内容
        """
        path = Path(file_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.warning(f"Tool result file not found: {file_path}")
        return "[工具结果文件已丢失]"

    def get_tool_result_path(self, tool_call_id: str) -> Path:
        """
        获取工具结果文件路径

        Args:
            tool_call_id: 工具调用ID

        Returns:
            文件路径
        """
        return self.tool_result_dir / f"{tool_call_id}.txt"

    def exists(self, tool_call_id: str) -> bool:
        """
        检查工具结果文件是否存在

        Args:
            tool_call_id: 工具调用ID

        Returns:
            文件是否存在
        """
        return self.get_tool_result_path(tool_call_id).exists()
