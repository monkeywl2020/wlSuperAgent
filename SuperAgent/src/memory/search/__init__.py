# -*- coding: utf-8 -*-
"""
搜索模块

包含文件监视器、向量存储、BM25索引、问答提取和搜索引擎等功能
"""

# 导出搜索组件
from .file_watcher import FileWatcher  # 文件监视
from .bm25_index import BM25Index  # BM25索引
from .file_store import FileStore  # 搜索引擎
from .qa_extractor import QAExtractor, MemoryChunker, QAPair  # 问答提取和分块

# 定义模块导出列表
__all__ = [
    "FileWatcher",  # 文件监视器
    "BM25Index",  # BM25索引
    "FileStore",  # 搜索引擎
    "QAExtractor",  # 问答提取器
    "MemoryChunker",  # 分块器
    "QAPair",  # 问答对
]
