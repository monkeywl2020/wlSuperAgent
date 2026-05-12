# -*- coding: utf-8 -*-
"""
搜索引擎模块

维护双索引（向量+BM25）实现混合搜索
支持：
1. QA提取 + 分块
2. 纯分块
3. 增量索引同步（基于文件修改时间检测新增/删除/修改）
"""

# 导入标准库
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime

# 导入Path
from pathlib import Path

# 导入核心模块
from ..core.directory_manager import DirectoryManager
from ..vector_store import SQLiteVectorStoreWithIndex
from ..embedding import EmbeddingProvider
from .bm25_index import BM25Index
from .qa_extractor import QAExtractor, MemoryChunker

# 导入第三方库
from loguru import logger


class FileStore:
    """文件存储搜索引擎

    功能：
    1. 维护向量索引（SQLite + FAISS）
    2. 维护BM25索引
    3. 支持QA提取 + 分块
    4. 支持纯分块
    5. 增量索引同步（基于文件修改时间检测新增/删除/修改）
    """

    def __init__(
        self,
        dir_manager: DirectoryManager,
        vector_store: SQLiteVectorStoreWithIndex,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[Any] = None,
        enable_qa_extraction: bool = True,
        chunk_size: int = 500,
        overlap: int = 50,
        qa_extract_prompt: str = "",
        bm25_index_path: Optional[str] = None
    ):
        """
        初始化文件存储

        Args:
            dir_manager: 目录管理器
            vector_store: 向量存储
            embedding_provider: Embedding接口
            llm_provider: LLM接口（用于QA提取）
            enable_qa_extraction: 是否启用QA提取
            chunk_size: 分块大小
            overlap: 分块重叠大小
            qa_extract_prompt: QA提取提示词模板
            bm25_index_path: BM25索引文件路径
        """
        self.dir_manager = dir_manager
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.bm25_index = BM25Index(index_path=bm25_index_path)

        # QA提取器和分块器
        self.enable_qa_extraction = enable_qa_extraction
        if enable_qa_extraction:
            self.qa_extractor = QAExtractor(llm_provider, prompt_template=qa_extract_prompt)
            self.chunker = MemoryChunker(chunk_size=chunk_size, overlap=overlap)
        else:
            self.qa_extractor = None
            self.chunker = MemoryChunker(chunk_size=chunk_size, overlap=overlap)

        # 索引状态文件路径
        self._index_state_path = Path(vector_store.db_path).parent / "index_state.json"

        # 跟踪已索引的memory文件信息: {file_path: {"mtime": float, "mtime_str": str}}
        # - mtime: 文件修改时间戳（用于检测文件是否变化）
        # - mtime_str: 易读的修改时间格式（如 "2024-01-15 10:30:45"）
        self._indexed_files_info: Dict[str, Dict[str, Any]] = {}

    def _load_index_state(self):
        """从文件加载索引状态"""
        if not self._index_state_path.exists():
            return

        try:
            # 这个 index_state.json 文件 是一个 JSON 文件, 里面记录了建立了索引的 memory文件的 修改时间。
            with open(self._index_state_path, "r", encoding="utf-8") as f:
                self._indexed_files_info = json.load(f)
            logger.info(f"Loaded index state: {len(self._indexed_files_info)} files")
        except Exception as e:
            logger.error(f"Failed to load index state: {e}")
            self._indexed_files_info = {}

    def _save_index_state(self):
        """保存索引状态到文件"""
        try:
            self._index_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index_state_path, "w", encoding="utf-8") as f:
                json.dump(self._indexed_files_info, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save index state: {e}")

    def build_index(self):
        """构建索引（增量构建）

        检测逻辑：
        1. 获取当前memory目录中的文件（每日记忆 .md 和长期记忆 MEMORY.md）
        2. 对比memory文件的当前修改时间(mtime)和index_state.json中记录的mtime
        3. 判断文件是新增、修改还是删除：
           - 新增：memory文件存在，但index_state.json中没有记录
           - 修改：memory文件存在，且当前mtime > 记录的mtime
           - 删除：index_state.json中有记录，但memory文件已不存在
        """
        logger.info("Starting index build...")

        # 加载索引状态
        self._load_index_state()

        # 加载 BM25 索引
        self.bm25_index.load()

        # 获取当前memory目录中的memory文件（每日记忆 + 长期记忆）
        current_files = self._get_current_memory_files()

        # 分类处理memory文件
        files_to_add = []      # 新增的memory文件
        files_to_rebuild = []  # 修改的memory文件（需先删除再重建）
        files_to_remove = []   # 已删除的memory文件

        # 遍历当前memory文件，与记录的mtime对比
        for file_path in current_files:
            path = Path(file_path)
            current_mtime = path.stat().st_mtime
            
            # 如果文件路径 不在 已索引文件信息中，说明是新增的。
            if file_path not in self._indexed_files_info:
                # 情况1：memory文件是新增的
                files_to_add.append(file_path)
            else:
                saved_mtime = self._indexed_files_info[file_path].get("mtime", 0)
                if current_mtime > saved_mtime:#  如果文件当前修改时间大于保存的时间，说明文件已经修改
                    # 情况2：memory文件已修改
                    files_to_rebuild.append(file_path)
                # else: 文件未变化，跳过

        # 情况2：处理修改的memory文件（先删除旧索引，再重建）
        for file_path in files_to_rebuild:
            logger.info(f"Memory file modified, rebuilding index: {file_path}")
            self._remove_file_index(file_path)
            self._index_file(file_path)

        # 情况1：处理新增的memory文件
        for file_path in files_to_add:
            logger.info(f"Memory file added, indexing: {file_path}")
            self._index_file(file_path)

        # 情况3：处理已删除的memory文件
        for file_path in list(self._indexed_files_info.keys()): # 遍历已索引文件信息中的文件路径
            if file_path not in current_files: # 如果文件不存在，说明文件被删除了，但是索引文件信息中有，表示这个文件需要删除索引和相关的信息了
                files_to_remove.append(file_path)

        for file_path in files_to_remove:
            logger.info(f"Memory file deleted, removing index: {file_path}")
            self._remove_file_index(file_path)

        # 保存 BM25 索引
        self.bm25_index.save()

        # 保存索引状态
        self._save_index_state()

        logger.info(f"Index build completed: {len(self._indexed_files_info)} files indexed.")

    def _get_current_memory_files(self) -> set:
        """获取当前memory目录中的所有文件路径"""
        files = set()

        # 每日记忆文件
        memory_dir = self.dir_manager.get_memory_dir()
        if memory_dir.exists():
            for file_path in memory_dir.glob("*.md"):
                files.add(str(file_path))

        # 长期记忆文件
        memory_file = self.dir_manager.get_long_term_memory_file()
        if memory_file.exists():
            files.add(str(memory_file))

        return files

    def _index_file(self, file_path: str):
        """为单个文件建立索引

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return

        try:
            content = path.read_text(encoding="utf-8")
            file_id = path.stem

            # 添加到BM25索引
            self.bm25_index.add(file_id, content)

            # 根据配置选择索引方式
            if self.enable_qa_extraction and self.qa_extractor:
                self._index_with_qa(file_id, content, path)
            else:
                self._index_with_chunks(file_id, content, path)

            # 记录已索引的memory文件信息
            file_mtime = path.stat().st_mtime
            self._indexed_files_info[file_path] = {
                "mtime": file_mtime,
                "mtime_str": datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }

            logger.info(f"Indexed file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")

    def _index_with_qa(self, file_id: str, content: str, file_path: Path):
        """使用QA提取方式索引

        Args:
            file_id: 文件ID
            content: 文件内容
            file_path: 文件路径
        """
        # 提取QA对
        qa_pairs = self.qa_extractor.extract(content, source=file_id)
        logger.info(f"Extracted {len(qa_pairs)} QA pairs from {file_id}")

        # 保存QA对文件
        qa_file_path = self._get_qa_file_path(file_path)
        self._save_qa_pairs_to_file(qa_pairs, qa_file_path)

        if qa_pairs:
            # 使用QA分块（每个QA对独立成块）
            chunks = self.chunker.create_qa_chunks(qa_pairs)

            for chunk in chunks:
                try:
                    embedding = self.embedding_provider.get_embedding(chunk["text"])
                    chunk_id = f"{file_id}_qa_{chunk['index']}"

                    self.vector_store.add(
                        id=chunk_id,
                        content=chunk["text"],
                        embedding=embedding,
                        metadata={
                            "source": file_id,
                            "type": "qa_chunk",
                            "qa_pair": chunk.get("qa_pair"),
                            "chunk_index": chunk["index"]
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to add chunk {chunk_id}: {e}")
        else:
            # 没有QA对，回退到纯分块
            logger.warning(f"No QA pairs extracted from {file_id}, falling back to chunking")
            self._index_with_chunks(file_id, content, file_path)

    def _index_with_chunks(self, file_id: str, content: str, file_path: Path):
        """使用分块方式索引

        Args:
            file_id: 文件ID
            content: 文件内容
            file_path: 文件路径
        """
        chunks = self.chunker.chunk_text(content, source=file_id)

        for chunk in chunks:
            try:
                embedding = self.embedding_provider.get_embedding(chunk["text"])
                chunk_id = f"{file_id}_chunk_{chunk['index']}"

                self.vector_store.add(
                    id=chunk_id,
                    content=chunk["text"],
                    embedding=embedding,
                    metadata={
                        "source": file_id,
                        "type": "chunk",
                        "chunk_index": chunk["index"]
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to add chunk {chunk_id}: {e}")

        logger.info(f"Indexed {len(chunks)} chunks from {file_id}")

    def _save_qa_pairs_to_file(self, qa_pairs, file_path: Path):
        """保存QA对到文件

        Args:
            qa_pairs: QA对列表
            file_path: QA文件路径
        """
        if not qa_pairs:
            return

        lines = ["# QA Pairs\n\n"]
        for i, qa in enumerate(qa_pairs):
            lines.append(f"## QA Pair {i + 1}\n")
            lines.append(f"Q: {qa.question}\n")
            lines.append(f"A: {qa.answer}\n\n")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(''.join(lines), encoding="utf-8")
            logger.info(f"Saved QA pairs to: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save QA pairs to {file_path}: {e}")

    def _get_qa_file_path(self, memory_file_path: Path) -> Path:
        """获取QA文件路径

        QA文件存放在memory_qa目录下，不在memory目录下，
        以避免索引时扫描到QA文件（QA文件是索引过程的副产物）

        Args:
            memory_file_path: 原始记忆文件路径

        Returns:
            QA文件路径
        """
        if memory_file_path.name == "MEMORY.md":
            return self.dir_manager.get_long_term_memory_qa_file()
        else:
            # 每日记忆: 2024-01-15.md -> memory_qa/2024-01-15_qa.md
            return self.dir_manager.get_memory_qa_dir() / f"{memory_file_path.stem}_qa.md"

    def _remove_file_index(self, file_path: str):
        """移除文件的索引

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        file_id = path.stem

        try:
            # 从向量存储中删除该文件的所有chunk
            self._delete_chunks_by_source(file_id)

            # 从BM25索引中删除
            self.bm25_index.delete(file_id)

            # 删除QA文件（如果存在）
            qa_file_path = self._get_qa_file_path(path)
            if qa_file_path.exists():
                qa_file_path.unlink()
                logger.info(f"Deleted QA file: {qa_file_path}")

            # 从已索引文件中移除
            self._indexed_files_info.pop(file_path, None)
            logger.info(f"Removed index for file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to remove index for {file_path}: {e}")

    def _delete_chunks_by_source(self, source_id: str):
        """删除指定源的所有chunk

        Args:
            source_id: 源文件ID
        """
        self.vector_store.delete_by_prefix(f"{source_id}_qa_")
        self.vector_store.delete_by_prefix(f"{source_id}_chunk_")
        logger.info(f"Deleted chunks for source: {source_id}")

    def add_file(self, file_path: str):
        """添加单个文件的索引

        Args:
            file_path: 文件路径
        """
        if file_path not in self._indexed_files_info:
            self._index_file(file_path)
            self._save_index_state()

    def remove_file(self, file_path: str):
        """移除单个文件的索引

        Args:
            file_path: 文件路径
        """
        if file_path in self._indexed_files_info:
            self._remove_file_index(file_path)
            self._save_index_state()

    def update_file(self, file_path: str):
        """更新单个文件的索引（先删后加）

        Args:
            file_path: 文件路径
        """
        if file_path in self._indexed_files_info:
            self._remove_file_index(file_path)
        self._index_file(file_path)
        self._save_index_state()

    def search(
        self,
        query: str,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        混合搜索

        Args:
            query: 查询文本
            vector_weight: 向量搜索权重
            bm25_weight: BM25搜索权重
            top_k: 返回结果数

        Returns:
            搜索结果列表
        """
        # 向量搜索
        vector_results = []
        try:
            query_embedding = self.embedding_provider.get_embedding(query)
            raw_vector_results = self.vector_store.search(query_embedding, top_k * 2)
            for chunk_id, score in raw_vector_results:
                doc = self.vector_store.get(chunk_id)
                if doc:
                    vector_results.append({
                        "chunk_id": chunk_id,
                        "content": doc["content"],
                        "source": doc["metadata"].get("source", ""),
                        "type": doc["metadata"].get("type", "chunk"),
                        "score": score
                    })
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # BM25搜索（直接返回原文内容）
        raw_bm25_results = self.bm25_index.search(query, top_k * 2)
        bm25_results = []
        for doc_id, score in raw_bm25_results:
            file_path, content = self._get_file_content_by_id(doc_id)
            if content:
                bm25_results.append({
                    "chunk_id": f"{doc_id}_chunk_0",
                    "content": content,
                    "source": doc_id,
                    "type": "full_doc",
                    "score": score,
                    "source_file_path": file_path,
                    "source_content": content
                })

        # 合并结果
        combined_scores: Dict[str, Dict[str, Any]] = {}

        # 处理向量结果
        if vector_results:
            vector_max = max((r["score"] for r in vector_results), default=1.0)
            for r in vector_results:
                chunk_id = r["chunk_id"]
                normalized_score = r["score"] / vector_max if vector_max > 0 else 0
                weighted_score = normalized_score * vector_weight

                # 获取源文件内容
                file_path, source_content = self._get_file_content_by_id(r["source"])

                combined_scores[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": r["content"],
                    "source": r["source"],
                    "type": r["type"],
                    "score": weighted_score,
                    "match_score": weighted_score,
                    "source_file_path": file_path,
                    "source_content": source_content
                }

        # 处理BM25结果
        if bm25_results:
            bm25_max = max((r["score"] for r in bm25_results), default=1.0)
            for r in bm25_results:
                chunk_id = r["chunk_id"]
                normalized_score = r["score"] / bm25_max if bm25_max > 0 else 0
                weighted_score = normalized_score * bm25_weight

                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["score"] += weighted_score
                    combined_scores[chunk_id]["match_score"] += weighted_score
                else:
                    combined_scores[chunk_id] = {
                        "chunk_id": chunk_id,
                        "content": r["content"],
                        "source": r["source"],
                        "type": r["type"],
                        "score": weighted_score,
                        "match_score": weighted_score,
                        "source_file_path": r.get("source_file_path"),
                        "source_content": r.get("source_content")
                    }

        # 排序并返回
        results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _get_file_content_by_id(self, file_id: str):
        """根据文件ID获取文件内容

        Args:
            file_id: 文件ID (如 "2024-01-15" 或 "MEMORY")

        Returns:
            (文件路径, 文件内容)
        """
        # 尝试每日记忆目录
        memory_dir = self.dir_manager.get_memory_dir()
        file_path = memory_dir / f"{file_id}.md"
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
                return str(file_path), content
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")

        # 尝试长期记忆文件
        if file_id == "MEMORY":
            memory_file = self.dir_manager.get_long_term_memory_file()
            if memory_file.exists():
                try:
                    content = memory_file.read_text(encoding="utf-8")
                    return str(memory_file), content
                except Exception as e:
                    logger.warning(f"Failed to read file {memory_file}: {e}")

        return None, None

    def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """获取chunk内容

        Args:
            chunk_id: chunk ID

        Returns:
            chunk内容
        """
        doc = self.vector_store.get(chunk_id)
        if doc:
            return doc["content"]
        return None

    def get_source_chunks(self, source_id: str) -> List[Dict[str, Any]]:
        """获取指定源的所有chunks

        Args:
            source_id: 源文件ID

        Returns:
            chunk列表
        """
        chunks = []
        all_ids = self.vector_store.get_all_ids()

        for chunk_id in all_ids:
            if chunk_id.startswith(f"{source_id}_"):
                doc = self.vector_store.get(chunk_id)
                if doc:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {})
                    })

        return chunks

    def clear_index(self):
        """清空所有索引"""
        self.vector_store.clear()
        self.bm25_index.clear()
        self._indexed_files_info.clear()
        self._save_index_state()
        logger.info("All indexes cleared")

    def get_index_stats(self) -> Dict[str, int]:
        """获取索引统计"""
        return {
            "indexed_files": len(self._indexed_files_info),
            "vector_count": self.vector_store.count(),
            "bm25_count": self.bm25_index.size(),
        }
