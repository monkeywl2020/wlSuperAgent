# -*- coding: utf-8 -*-
"""
BM25索引模块

使用 bm25s 库实现 BM25 索引，支持中文分词（jieba）和停用词过滤
"""

# 导入标准库
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

# 导入第三方库
import bm25s
import jieba
from loguru import logger


STOPWORDS_CN = [
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
    "她", "它", "们", "这个", "那个", "什么", "怎么", "为什么",
    "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "啦", "喔", "哈",
    "为", "与", "或", "而", "且", "但", "如果", "因为", "所以",
    "可以", "没", "能", "让", "把", "被", "对", "给", "跟", "从",
    "向", "往", "于", "以", "及", "等", "多", "少", "大", "小",
    "新", "旧", "前", "后", "里", "外", "中", "内",
    "上面", "下面", "左边", "右边", "这里", "那里", "这么", "那么",
]

STOPWORDS_EN = [
    "the", "is", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "that", "this",
    "it", "its", "be", "are", "was", "were", "will", "would", "could",
    "should", "can", "may", "might", "must", "have", "has", "had",
    "do", "does", "did", "done", "not", "no", "so", "if", "then",
    "there", "their", "they", "them", "we", "us", "our", "you", "your",
]


class BM25Index:
    """BM25索引（基于 bm25s 库实现）

    支持：
    - 中文分词（jieba）
    - 停用词过滤
    - 索引持久化
    - 增量添加/删除文档
    """

    def __init__(
        self,
        index_path: Optional[str] = None
    ):
        """
        初始化 BM25 索引

        Args:
            index_path: 索引文件路径（可选）
        """
        self.index_path = index_path

        # 内置停用词（中文 + 英文）
        self.stopwords = set(STOPWORDS_CN + STOPWORDS_EN)

        # 文档存储 (doc_id -> text)
        self.documents: Dict[str, str] = {}

        # doc_ids 列表（用于 bm25s 内部）
        self._doc_ids: List[str] = []
        self._corpus: List[str] = []

        # BM25 模型
        self._retriever: Optional[bm25s.BM25] = None
        self._is_indexed: bool = False

        # 分词结果缓存 (doc_id -> tokens)
        self._tokenization_cache: Dict[str, List[str]] = {}

    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 分词并过滤停用词"""
        if not text:
            return []
        tokens = jieba.lcut(text)
        # 下面是过滤停用词
        if self.stopwords:
            tokens = [t for t in tokens if t.strip() and t not in self.stopwords]
        return tokens

    def _ensure_index_dir(self):
        """确保索引目录存在"""
        if self.index_path:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

    def add(self, id: str, text: str):
        """
        添加文档到索引

        Args:
            id: 唯一标识
            text: 文档文本
        """
        if not text or not text.strip():
            return

        # 如果文档已存在，先删除
        if id in self.documents:
            self.delete(id)

        # 添加到文档存储
        self.documents[id] = text
        self._doc_ids.append(id)
        self._corpus.append(text)

        # 标记需要重新索引
        self._is_indexed = False

    def _rebuild_index(self):
        """重新构建 BM25 索引"""
        if not self._corpus:
            self._retriever = None
            self._is_indexed = False
            return

        try:
            # 使用 jieba 分词
            corpus_tokens = [self._tokenize(text) for text in self._corpus]

            # 保存分词结果到缓存
            self._tokenization_cache.clear()
            for doc_id, tokens in zip(self._doc_ids, corpus_tokens):
                self._tokenization_cache[doc_id] = tokens

            # 创建 BM25 模型
            self._retriever = bm25s.BM25()

            # 索引
            self._retriever.index(corpus_tokens)

            self._is_indexed = True
            logger.info(f"BM25 index built successfully: {len(self._corpus)} documents")

        except Exception as e:
            logger.error(f"BM25 index build failed: {e}")
            self._retriever = None
            self._is_indexed = False

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        搜索文档

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            [(doc_id, score), ...] 列表
        """
        if not self._corpus or not query.strip():
            return []

        # 确保索引已构建
        if not self._is_indexed:
            self._rebuild_index()

        if self._retriever is None:
            return []

        try:
            # 分词查询
            query_tokens = [self._tokenize(query)]

            # 确保 k 不大于语料库大小
            actual_k = min(top_k, len(self._corpus))

            # 检索
            doc_ids, scores = self._retriever.retrieve(query_tokens, k=actual_k)

            # 转换结果
            results = []
            for i in range(len(doc_ids[0])):
                doc_idx = int(doc_ids[0, i])
                score = float(scores[0, i])
                if doc_idx < len(self._doc_ids):
                    actual_id = self._doc_ids[doc_idx]
                    results.append((actual_id, score))

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def get_document(self, id: str) -> str:
        """
        获取文档内容

        Args:
            id: 文档ID

        Returns:
            文档文本
        """
        return self.documents.get(id, "")

    def get_tokenization(self, id: str) -> List[str]:
        """
        获取文档的分词结果

        Args:
            id: 文档ID

        Returns:
            分词列表
        """
        return self._tokenization_cache.get(id, [])

    def get_all_tokenizations(self) -> Dict[str, List[str]]:
        """
        获取所有文档的分词结果

        Returns:
            {doc_id: tokens} 字典
        """
        return self._tokenization_cache.copy()

    def get_all_ids(self) -> List[str]:
        """
        获取所有文档ID

        Returns:
            文档ID列表
        """
        return list(self.documents.keys())

    def delete(self, id: str):
        """
        删除文档

        Args:
            id: 要删除的文档ID
        """
        if id not in self.documents:
            return

        # 从文档存储中删除
        del self.documents[id]

        # 删除分词缓存
        self._tokenization_cache.pop(id, None)

        # 找到并删除对应的索引
        idx = None
        for i, doc_id in enumerate(self._doc_ids):
            if doc_id == id:
                idx = i
                break

        if idx is not None:
            del self._doc_ids[idx]
            del self._corpus[idx]

        # 标记需要重新索引
        # 注意：这里不删除索引文件，因为删除索引文件会影响所有文档
        # 下次 save() 时会调用 _rebuild_index() 重建整个索引
        self._is_indexed = False

    def clear(self):
        """清空索引"""
        self.documents.clear()
        self._doc_ids.clear()
        self._corpus.clear()
        self._tokenization_cache.clear()
        self._retriever = None
        self._is_indexed = False

        # 删除索引文件
        if self.index_path:
            index_path_obj = Path(self.index_path)
            if index_path_obj.exists() and index_path_obj.is_file():
                index_path_obj.unlink()
            meta_path = str(self.index_path) + ".meta"
            meta_path_obj = Path(meta_path)
            if meta_path_obj.exists() and meta_path_obj.is_file():
                meta_path_obj.unlink()

        logger.info("BM25 index cleared")

    def save(self):
        """保存索引到文件"""
        if not self.index_path:
            logger.warning(f"Index path not set, cannot save")
            return

        if not self._corpus:
            logger.warning("Index is empty, nothing to save")
            return

        # 确保索引已构建
        if not self._is_indexed:
            self._rebuild_index()

        if self._retriever is None:
            logger.error("BM25 model not initialized, cannot save")
            return

        try:
            self._ensure_index_dir()

            # 保存索引
            self._retriever.save(self.index_path)

            # 保存元数据（不包含文档内容）
            meta_path = str(self.index_path) + ".meta"
            meta_data = {
                "doc_ids": self._doc_ids,
                "tokenization_cache": self._tokenization_cache, # 缓存 分词结果，这个内容会保存起来
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False)

            logger.info(f"BM25 index saved to: {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def load(self) -> bool:
        """
        从文件加载索引

        Returns:
            是否加载成功
        """
        if not self.index_path:
            logger.warning("Index path not set, cannot load")
            return False

        index_path = Path(self.index_path)
        if not index_path.exists():
            logger.info(f"Index file not found: {self.index_path}")
            return False

        try:
            # 加载元数据
            meta_path = str(self.index_path) + ".meta"
            if Path(meta_path).exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    self._doc_ids = meta_data.get("doc_ids", [])
                    self._tokenization_cache = meta_data.get("tokenization_cache", {})
            else:
                self._doc_ids = []
                self._tokenization_cache = {}

            # 加载索引
            self._retriever = bm25s.BM25.load(self.index_path, load_corpus=False)
            self._is_indexed = True

            logger.info(f"BM25 index loaded: {len(self._doc_ids)} documents, {len(self._tokenization_cache)} tokenization records")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def size(self) -> int:
        """
        获取文档数量

        Returns:
            文档数量
        """
        return len(self.documents)

    def __len__(self) -> int:
        """获取文档数量"""
        return len(self.documents)
