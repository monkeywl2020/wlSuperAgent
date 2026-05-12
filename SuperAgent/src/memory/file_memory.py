# -*- coding: utf-8 -*-
"""
文件记忆系统主模块

整合所有子模块，提供统一的记忆系统接口
"""

# 导入标准库
import json  # JSON序列化
import uuid  # 生成唯一标识符
from typing import List, Dict, Any, Optional, Callable, Union  # 类型注解
from pathlib import Path  # 路径操作

# 导入第三方库
from loguru import logger  # 日志记录

# 导入核心模块
from .core.config import MemoryConfig  # 配置数据类
from .core.directory_manager import DirectoryManager  # 目录管理

# 导入集中配置
from src.settings import memory_settings

# 导入消息类（使用 src/msg 中的 Msg）
from src.msg import Msg as Message

# 导入存储模块
from .storage.token_counter import TokenCounter  # Token计数
from .storage.dialog_storage import DialogStorage  # 对话存储
from .vector_store import SQLiteVectorStoreWithIndex  # SQLite向量存储

# 导入压缩模块
from .compression.tool_compactor import ToolResultCompactor  # 工具压缩
from .compression.compactor import Compactor  # 上下文压缩
from .compression.summarizer import Summarizer  # 异步摘要

# 导入搜索模块
from .search.file_watcher import FileWatcher  # 文件监视
from .search.file_store import FileStore  # 搜索引擎
from .search.bm25_index import BM25Index  # BM25索引

# 导入新接口
from .embedding import EmbeddingProvider, create_embedding_provider  # Embedding接口
from .llm import LLMProvider, create_llm_provider  # LLM接口


class FileMemory:
    """文件记忆系统主类 - 整合所有模块"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        初始化文件记忆系统

        Args:
            config: 配置字典，如果为None则使用集中配置
            embedding_provider: Embedding接口实例，如果为None则根据配置创建
            llm_provider: LLM接口实例，如果为None则根据配置创建
        """
        # 加载或创建配置
        if config:
            self.config = MemoryConfig(**config) if isinstance(config, dict) else config
        else:
            self.config = memory_settings  # 使用集中配置

        # 初始化 Embedding 接口
        if embedding_provider:
            self.embedding_provider = embedding_provider
        else:
            # 根据配置创建 Embedding 的实例，配置部分是 self.config.embedding （yaml的 Embedding 部分）
            self.embedding_provider = create_embedding_provider({
                "provider": self.config.embedding.provider,
                "model_name": self.config.embedding.model_name,
                "device": self.config.embedding.device,
                "batch_size": self.config.embedding.batch_size,
                "dimension": self.config.embedding.dimension,
            })

        # 初始化 LLM 接口
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            # 根据配置创建 LLM 的实例， 配置部分是 self.config.llm （yaml的 llm 部分）
            self.llm_provider = create_llm_provider({
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
                "api_key": self.config.llm.api_key,
                "base_url": self.config.llm.base_url,
                "max_tokens": self.config.llm.max_tokens,
                "top_p": self.config.llm.top_p,
                "temperature": self.config.llm.temperature,
                "presence_penalty": self.config.llm.presence_penalty,
                "extra_body": self.config.llm.extra_body
            })

        # 初始化各模块
        self.dir_manager = DirectoryManager(self.config.working_dir) # 目录管理器，管理记忆中的目录相关处理。
        self.dir_manager.ensure_directories() # 使用目录管理器， 创建所有必需的目录结构

        self.token_counter = TokenCounter()  # token counter 计数
        self.tool_compactor = ToolResultCompactor(self.config, self.dir_manager)  # tool result 的压缩
        self.compactor = Compactor(self.config, self.token_counter) # 上下文压缩器 - 压缩历史消息并生成摘要
        self.dialog_storage = DialogStorage(self.dir_manager) # 对话存储 - 将消息写入JSONL文件

        # 初始化摘要器（传入 LLM provider）
        self.summarizer = Summarizer(
            self.config,
            self.dir_manager,
            self.llm_provider
        )

        # 下面的文件监视器主要监控 memory文件夹是否变化，memory.md ，每日摘要和 长期记忆是否变化，如变化则更新索引
        self.file_watcher = FileWatcher(self.dir_manager, self._on_file_change)

        # 初始化向量存储（SQLite + FAISS）
        self.vector_store = SQLiteVectorStoreWithIndex(
            db_path=self.config.vector_db.db_path,
            table_name=self.config.vector_db.table_name,
            dimension=self.config.vector_db.dimension
        )

        # BM25索引路径（与向量数据库同目录，使用 .index 作为文件扩展名）
        bm25_index_path = str(Path(self.config.vector_db.db_path).parent / "bm25_index")

        # 初始化搜索引擎（混合检索，支持问答提取和分块）
        self.file_store = FileStore(
            self.dir_manager,
            self.vector_store,
            self.embedding_provider,
            llm_provider=self.llm_provider,
            enable_qa_extraction=self.config.search.enable_qa_extraction,
            chunk_size=self.config.search.chunk_size,
            overlap=self.config.search.overlap,
            qa_extract_prompt=self.config.qa_extract_prompt,
            bm25_index_path=bm25_index_path
        )

        # 会话消息缓冲区
        self.messages: List[Message] = [] # 内存中的消息列表，保存当前所有的消息
        self.current_summary: Optional[str] = None # 这个是当前会话的摘要，用于搜索时的上下文压缩

        # 构建搜索索引
        self.file_store.build_index()

        # 清理过期工具结果
        self.tool_compactor.cleanup_old_results()

        logger.info("File Memory System Initialization Complete!")

    def _on_file_change(self):
        """文件变化回调函数"""
        logger.info("File change detected, rebuilding index...")
        self.file_store.build_index()

    def _get_session_id_from_msg(self, msg: Union[Message, Dict]) -> str:
        """
        从消息中获取 session_id

        Args:
            msg: 消息对象或字典

        Returns:
            session_id

        Raises:
            ValueError: 消息不包含 session_id
        """
        if isinstance(msg, Message):
            metadata = msg.metadata or {}
        elif isinstance(msg, dict):
            metadata = msg.get("metadata", {})
        else:
            metadata = {}

        session_id = metadata.get("session_id", "")
        if not session_id:
            raise ValueError("消息必须包含 session_id")
        return session_id

    def add_message(
        self,
        role: Union[str, Message],
        content: Optional[str] = None,
        **kwargs
    ) -> Message:
        """
        添加消息到会话，并即时存储到文件

        Args:
            role: 角色 (user/assistant/system/tool/function) 或直接的 Message 对象
            content: 消息内容（如果 role 是字符串）
            **kwargs: 其他参数

        Returns:
            创建的消息对象

        Raises:
            ValueError: 当 role 为 "tool" 但未提供 tool_call_id 时，或消息缺少 session_id
        """
        metadata = kwargs.get("metadata", {})
        session_id = metadata.get("session_id", "")

        # 如果传入的是 Message 对象，提取 session_id
        if isinstance(role, Message):
            msg = role
            if not session_id:
                session_id = self._get_session_id_from_msg(msg)
                if msg.metadata is None:
                    msg.metadata = {}
                if not isinstance(msg.metadata, dict):
                    msg.metadata = {}
                # 确保 session_id 在 metadata 中
                msg.metadata["session_id"] = session_id
        else:
            # tool 角色的消息必须有 tool_call_id
            if role == "tool":
                tool_call_id = kwargs.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("tool_call_id is required when role is 'tool'")
                metadata["tool_call_id"] = tool_call_id

            if not session_id:
                raise ValueError("消息必须包含 session_id")

            # 创建新的 Message 对象
            msg = Message(
                role=role,
                content=content or "",
                name=kwargs.get("name"),
                metadata=metadata,
                tool_calls=kwargs.get("tool_calls"),
                extra=kwargs.get("extra"),
                tool_call_id=kwargs.get("tool_call_id"),
            )

        # 检查工具结果是否需要压缩
        if msg.role == "tool":
            logger.info(f"add_message msg role is tool, tool_call_id: {msg.tool_call_id} - content_len:{len(msg.content)}")
            if not msg.tool_call_id:
                raise ValueError("tool_call_id is required when role is 'tool'")

            text_content = msg.get_text_content()
            if text_content and self.tool_compactor.should_compact(text_content):
                # 压缩工具结果
                file_path = self.tool_compactor.compact(msg.tool_call_id, text_content)
                # 设置文件引用
                msg.file_path = file_path

        # 添加到消息列表
        self.messages.append(msg)

        # 即时存储到文件
        self.dialog_storage.append_message(msg.to_dict())

        return msg

    async def add_message_async(
        self,
        role: Union[str, Message],
        content: Optional[str] = None,
        **kwargs
    ) -> Message:
        """
        异步添加消息到会话，并即时存储到文件

        Args:
            role: 角色 (user/assistant/system/tool/function) 或直接的 Message 对象
            content: 消息内容（如果 role 是字符串）
            **kwargs: 其他参数

        Returns:
            创建的消息对象

        Raises:
            ValueError: 当 role 为 "tool" 但未提供 tool_call_id 时，或消息缺少 session_id
        """
        metadata = kwargs.get("metadata", {})
        session_id = metadata.get("session_id", "")

        # 如果传入的是 Message 对象，提取 session_id
        if isinstance(role, Message):
            msg = role
            if not session_id:
                session_id = self._get_session_id_from_msg(msg)
                if msg.metadata is None:
                    msg.metadata = {}
                if not isinstance(msg.metadata, dict):
                    msg.metadata = {}
                # 确保 session_id 在 metadata 中
                msg.metadata["session_id"] = session_id
        else:
            # tool 角色的消息必须有 tool_call_id
            if role == "tool":
                tool_call_id = kwargs.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("tool_call_id is required when role is 'tool'")
                metadata["tool_call_id"] = tool_call_id

            if not session_id:
                raise ValueError("消息必须包含 session_id")

            # 创建新的 Message 对象
            msg = Message(
                role=role,
                content=content or "",
                name=kwargs.get("name"),
                metadata=metadata,
                tool_calls=kwargs.get("tool_calls"),
                extra=kwargs.get("extra"),
                tool_call_id=kwargs.get("tool_call_id"),
            )

        # 检查工具结果是否需要压缩
        if msg.role == "tool":
            logger.info(f"add_message_async msg role is tool, tool_call_id: {msg.tool_call_id} - content_len:{len(msg.content)}")
            if not msg.tool_call_id:
                raise ValueError("tool_call_id is required when role is 'tool'")

            text_content = msg.get_text_content()
            if text_content and self.tool_compactor.should_compact(text_content):
                # 压缩工具结果
                file_path = self.tool_compactor.compact(msg.tool_call_id, text_content)
                # 设置文件引用
                msg.file_path = file_path

        # 添加到消息列表
        self.messages.append(msg)

        # 即时存储到文件
        await self.dialog_storage.append_message_async(msg.to_dict())

        return msg

    def _format_messages_for_llm(self, messages: List[Message]) -> str:
        """将消息列表格式化为文本"""
        messages_text = ""
        for msg in messages:
            text = msg.get_text_content() or ""
            messages_text += f"{msg.role}: {text}\n"
        return messages_text

    def _call_llm_for_compact(self, messages: List[Message]) -> Optional[str]:
        """调用 LLM 进行压缩"""
        if not self.config.compress_prompt or not self.llm_provider:
            return None

        messages_text = self._format_messages_for_llm(messages)
        prompt = self.config.compress_prompt.format(messages=messages_text)

        return self.llm_provider.chat([{"role": "user", "content": prompt}])

    def get_memory(self) -> List[Dict[str, Any]]:
        """
        获取当前会话的上下文记忆(从内存messages中获取对话记录)

        Returns:
            准备好的消息列表（用于传给LLM）
        """
        result = []

        # 添加摘要（如果存在）
        if self.current_summary:
            result.append({
                "role": "system",
                "content": f"之前对话的摘要:\n{self.current_summary}"
            })

        # 添加未压缩的消息
        for msg in self.messages:
            result.append(msg.to_dict())

        return result

    def check_and_compact(self) -> bool:
        """
        检查并执行压缩

        当 token 数量超过阈值时触发：
        1. 保存当前所有消息到 jsonl（追加模式）
        2. 使用 LLM 压缩消息为摘要
        3. 压缩后的摘要替代原有消息放在 self.messages 中

        Returns:
            是否执行了压缩
        """
        if not self.compactor.should_compact(self.messages):
            return False

        # 保存原始对话消息到 jsonl（用于生成每日摘要）
        original_messages = self.messages.copy()
        self.save_dialog()

        # 压缩消息为摘要（使用 LLM）
        self.messages, new_summary = self.compactor.compact_messages(
            self.messages,
            llm_compact_func=self._call_llm_for_compact
        )

        # 增量合并摘要
        if self.current_summary and new_summary:
            self.current_summary = f"{self.current_summary}\n\n{new_summary}"
        elif new_summary:
            self.current_summary = new_summary

        # 异步生成每日摘要（传入原始对话，用于生成完整摘要）
        self.summarizer.summarize_async(original_messages, self.current_summary or "")

        return True

    async def check_and_compact_async(self) -> bool:
        """
        异步检查并执行压缩

        当 token 数量超过阈值时触发：
        1. 保存当前所有消息到 jsonl（追加模式）
        2. 使用 LLM 压缩消息为摘要
        3. 压缩后的摘要替代原有消息放在 self.messages 中

        Returns:
            是否执行了压缩
        """
        if not self.compactor.should_compact(self.messages):
            return False

        # 保存原始对话消息到 jsonl（用于生成每日摘要）
        original_messages = self.messages.copy()
        await self.save_dialog_async()

        # 压缩消息为摘要（使用 LLM）
        self.messages, new_summary = self.compactor.compact_messages(
            self.messages,
            llm_compact_func=self._call_llm_for_compact
        )

        # 增量合并摘要
        if self.current_summary and new_summary:
            self.current_summary = f"{self.current_summary}\n\n{new_summary}"
        elif new_summary:
            self.current_summary = new_summary

        # 异步生成每日摘要（传入原始对话，用于生成完整摘要）
        self.summarizer.summarize_async(original_messages, self.current_summary or "")

        return True

    def memory_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索历史记忆

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            搜索结果列表，每项包含:
            - doc_id: chunk ID
            - content: chunk文本内容
            - source: 源文件ID
            - source_content: 源文件完整内容
            - type: chunk类型
            - score: 综合分数
        """
        if top_k is None:
            top_k = self.config.search.top_k

        # 混合搜索
        results = self.file_store.search(
            query,
            vector_weight=self.config.search.vector_weight,
            bm25_weight=self.config.search.bm25_weight,
            top_k=top_k
        )

        # 转换为结果列表
        search_results = []
        for item in results:
            search_results.append({
                "doc_id": item.get("chunk_id", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "source_content": item.get("source_content", ""),
                "type": item.get("type", ""),
                "score": item.get("score", 0.0)
            })

        return search_results

    async def memory_search_async(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        异步搜索历史记忆

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            搜索结果列表，每项包含:
            - doc_id: chunk ID
            - content: chunk文本内容
            - source: 源文件ID
            - source_content: 源文件完整内容
            - type: chunk类型
            - score: 综合分数
        """
        if top_k is None:
            top_k = self.config.search.top_k

        # 混合搜索
        results = self.file_store.search(
            query,
            vector_weight=self.config.search.vector_weight,
            bm25_weight=self.config.search.bm25_weight,
            top_k=top_k
        )

        # 转换为结果列表
        search_results = []
        for item in results:
            search_results.append({
                "doc_id": item.get("chunk_id", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "source_content": item.get("source_content", ""),
                "type": item.get("type", ""),
                "score": item.get("score", 0.0)
            })

        return search_results

    def get_long_term_memory(self) -> str:
        """获取长期记忆内容"""
        memory_file = self.dir_manager.get_long_term_memory_file()
        if memory_file.exists():
            return memory_file.read_text(encoding="utf-8")
        return ""

    async def get_long_term_memory_async(self) -> str:
        """异步获取长期记忆内容"""
        memory_file = self.dir_manager.get_long_term_memory_file()
        if memory_file.exists():
            return memory_file.read_text(encoding="utf-8")
        return ""

    async def get_recent_rounds_async(self, rounds: int = 20, session_id: str = "") -> List[Dict[str, Any]]:
        """
        异步获取最近 N 轮对话记录（过滤掉 tool 调用和 tool 结果）

        每轮 = 1 个 user 消息 + 1 个 assistant 消息
        过滤掉 role 为 "function"（tool 调用）和 "tool"（tool 结果）的消息

        Args:
            rounds: 轮数，默认 20 轮
            session_id: 会话ID，用于过滤特定会话的消息

        Returns:
            对话列表，格式为:
            - {'role': 'user', 'content': 'xxx'}
            - {'role': 'assistant', 'content': 'xxx'}
            - {'role': 'user', 'content': 'xxx'}
            - {'role': 'assistant', 'content': 'xxx'}
            ...
        """
        all_messages = await self.dialog_storage.load_recent_dialogs_async(days=0, session_id=session_id)
        logger.info(f"Retrieved {len(all_messages)} messages from the dialog storage.")

        result = []
        current_round = None

        for msg in all_messages:
            role = msg.get("role", "")

            if role == "user":
                if current_round is not None:
                    result.extend(current_round)
                current_round = [
                    {"role": "user", "content": msg.get("content", "")},
                    {"role": "assistant", "content": ""}
                ]
            elif role == "assistant" and current_round is not None:
                current_round[1]["content"] = msg.get("content", "")

        if current_round is not None:
            result.extend(current_round)

        total_rounds = len(result) // 2
        if total_rounds > rounds:
            result = result[-rounds * 2:]

        return result

    def get_recent_rounds(self, rounds: int = 20, session_id: str = "") -> List[Dict[str, Any]]:
        """
        获取最近 N 轮对话记录（过滤掉 tool 调用和 tool 结果）

        每轮 = 1 个 user 消息 + 1 个 assistant 消息
        过滤掉 role 为 "function"（tool 调用）和 "tool"（tool 结果）的消息

        Args:
            rounds: 轮数，默认 20 轮
            session_id: 会话ID，用于过滤特定会话的消息

        Returns:
            对话列表，格式为:
            - {'role': 'user', 'content': 'xxx'}
            - {'role': 'assistant', 'content': 'xxx'}
            - {'role': 'user', 'content': 'xxx'}
            - {'role': 'assistant', 'content': 'xxx'}
            ...
        """
        all_messages = self.dialog_storage.load_recent_dialogs(days=0, session_id=session_id)

        result = []
        current_round = None

        for msg in all_messages:
            role = msg.get("role", "")

            if role == "user":
                if current_round is not None:
                    result.extend(current_round)
                current_round = [
                    {"role": "user", "content": msg.get("content", "")},
                    {"role": "assistant", "content": ""}
                ]
            elif role == "assistant" and current_round is not None:
                current_round[1]["content"] = msg.get("content", "")

        if current_round is not None:
            result.extend(current_round)

        total_rounds = len(result) // 2
        if total_rounds > rounds:
            result = result[-rounds * 2:]

        logger.info(f"Retrieved the most recent {rounds} rounds of conversation, totaling {len(result)} entries.")
        return result

    def save_dialog(self):
        """保存当前对话到磁盘（追加模式）"""
        self.dialog_storage.append_messages([msg.to_dict() for msg in self.messages])

    async def save_dialog_async(self):
        """异步保存当前对话到磁盘（追加模式）"""
        await self.dialog_storage.append_messages_async([msg.to_dict() for msg in self.messages])

    def clear_session(self):
        """清除当前会话（保留长期记忆）"""
        self.save_dialog()

        if self.messages:
            self.summarizer.summarize_async(self.messages, self.current_summary or "")

        self.messages = []
        self.current_summary = None

        logger.info("Session cleared successfully and saved to disk")

    async def clear_session_async(self):
        """异步清除当前会话（保留长期记忆）"""
        await self.save_dialog_async()

        if self.messages:
            self.summarizer.summarize_async(self.messages, self.current_summary or "")

        self.messages = []
        self.current_summary = None

        logger.info("Session cleared successfully and saved to disk")

    def wait_for_async_tasks(self):
        """等待异步任务完成"""
        self.summarizer.wait_for_completion()

    def get_message_count(self) -> int:
        """获取当前会话消息数量"""
        return len(self.messages)

    def get_token_count(self) -> int:
        """获取当前会话token数量"""
        return self.token_counter.count_messages_tokens(self.messages)

    def rebuild_index(self):
        """重建搜索索引"""
        self.file_store.build_index()
        logger.info("Search index rebuilt successfully")

    def close(self):
        """关闭资源"""
        if hasattr(self.vector_store, 'close'):
            self.vector_store.close()
        logger.info("File Memory System Closed!")
