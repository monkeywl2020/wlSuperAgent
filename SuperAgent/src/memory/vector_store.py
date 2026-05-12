# -*- coding: utf-8 -*-
"""
向量存储模块（SQLite 实现）

使用 SQLite 存储向量数据，支持高效的向量检索
"""

# 导入标准库
import sqlite3  # SQLite 数据库
import json  # JSON 序列化
from typing import List, Tuple, Optional, Dict, Any  # 类型注解
from pathlib import Path  # 路径操作
import struct
import faiss
import numpy as np

# 导入第三方库
from loguru import logger  # 日志记录

# 导入 Embedding 接口
from .embedding import EmbeddingProvider


class VectorStore:
    """向量存储（SQLite 实现）"""

    def __init__(
        self,
        db_path: str,
        table_name: str = "embeddings",
        dimension: int = 1024
    ):
        """
        初始化向量存储

        Args:
            db_path: 数据库路径
            table_name: 表名
            dimension: 向量维度
        """
        self.db_path = db_path
        self.table_name = table_name
        self.dimension = dimension
        self.conn = None
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        # 确保目录存在
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # 连接数据库
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # 创建表
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id TEXT PRIMARY KEY,           -- 文档ID
            content TEXT NOT NULL,         -- 文档内容
            metadata TEXT,                  -- 元数据（JSON）
            vector BLOB,                    -- 向量数据（BLOB）
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 更新时间
        );
        """
        self.conn.execute(sql)

        # 创建索引（用于相似度搜索）
        sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at 
        ON {self.table_name}(created_at);
        """
        self.conn.execute(sql)

        self.conn.commit()
        logger.info(f"Vector database initialized: {self.db_path}")

    def add(
        self,
        id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加向量

        Args:
            id: 文档唯一ID
            content: 文档内容
            embedding: 向量
            metadata: 元数据
        """
        # 将向量转换为 bytes
        import struct
        vector_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        # 元数据转为 JSON
        metadata_json = json.dumps(metadata) if metadata else None

        sql = f"""
        INSERT OR REPLACE INTO {self.table_name} 
        (id, content, metadata, vector, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        self.conn.execute(sql, (id, content, metadata_json, vector_bytes))
        self.conn.commit()

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取向量

        Args:
            id: 文档ID

        Returns:
            包含向量和内容的字典
        """
        sql = f"SELECT * FROM {self.table_name} WHERE id = ?"
        cursor = self.conn.execute(sql, (id,))
        row = cursor.fetchone()

        if row is None:
            return None

        # 解析向量
        import struct
        vector = struct.unpack(f"{self.dimension}f", row["vector"])

        # 解析元数据
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": metadata,
            "vector": list(vector),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete(self, id: str):
        """
        删除向量

        Args:
            id: 文档ID
        """
        sql = f"DELETE FROM {self.table_name} WHERE id = ?"
        self.conn.execute(sql, (id,))
        self.conn.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        搜索相似向量（暴力搜索，适用于小规模数据）

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数
            filter_metadata: 元数据过滤条件

        Returns:
            (id, 相似度分数) 列表
        """
        # 获取所有向量
        sql = f"SELECT id, vector FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            return []

        # 计算相似度
        results = []
        import struct

        for row in rows:
            doc_id = row["id"]
            vector = struct.unpack(f"{self.dimension}f", row["vector"])
            similarity = self._cosine_similarity(query_embedding, list(vector))
            results.append((doc_id, similarity))

        # 排序并返回 top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def get_all_ids(self) -> List[str]:
        """
        获取所有文档ID

        Returns:
            文档ID列表
        """
        sql = f"SELECT id FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        return [row["id"] for row in cursor.fetchall()]

    def count(self) -> int:
        """
        获取向量数量

        Returns:
            向量数量
        """
        sql = f"SELECT COUNT(*) FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        return cursor.fetchone()[0]

    def clear(self):
        """清空所有向量"""
        sql = f"DELETE FROM {self.table_name}"
        self.conn.execute(sql)
        self.conn.commit()
        logger.info("Vector database cleared")

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """析构函数"""
        self.close()


class SQLiteVectorStoreWithIndex:
    """带索引的 SQLite 向量存储（使用 FAISS 加速）"""

    def __init__(
        self,
        db_path: str,
        table_name: str = "embeddings",
        dimension: int = 384,
        index_type: str = "flat",
        index_path: Optional[str] = None
    ):
        """
        初始化带索引的向量存储

        Args:
            db_path: 数据库路径
            table_name: 表名
            dimension: 向量维度
            index_type: 索引类型
            index_path: FAISS 索引文件路径（可选，默认与数据库同目录）
        """
        self.db_path = db_path
        self.table_name = table_name
        self.dimension = dimension
        self.index_type = index_type

        if index_path:
            self.index_path = index_path
        else:
            db_path_obj = Path(db_path)
            self.index_path = str(db_path_obj.parent / f"{db_path_obj.stem}.index")

        self.conn = None
        self.index = None
        self.id_map = {}

        self._init_db()
        self._init_index()

    def _init_db(self):
        """初始化数据库"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            vector BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(sql)
        self.conn.commit()
        logger.info(f"Vector database initialization complete: {self.db_path}")

    def _init_index(self):
        """初始化索引"""
        try:
            self.faiss = faiss
            self.np = np

            # 尝试从文件加载已持久化的 FAISS 索引
            if self.index_path and Path(self.index_path).exists(): # 如果faiss的index文件存在，则直接从这个文件恢复
                try:
                    self.index = faiss.read_index(self.index_path)
                    self._load_id_map()
                    logger.info(f"Loading FAISS index from file: {self.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index file: {e}. Reloading from the database.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self._load_vectors_from_db()
            else: # 否则从sqlite数据加载数据重建索引
                # 创建新索引并从 SQLite 加载
                self.index = faiss.IndexFlatL2(self.dimension)
                self._load_vectors_from_db()

            logger.info(f"FAISS index initialization complete: {self.index.ntotal} vectors")
        except ImportError:
            logger.warning("faiss is not installed. Using simple search.")
            self.faiss = None
            self.index = None

    def _load_id_map(self):
        """从数据库加载 ID 映射"""
        if self.conn is None:
            return

        sql = f"SELECT id FROM {self.table_name} ORDER BY created_at"
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        self.id_map = {i: row["id"] for i, row in enumerate(rows)}

    def save_index(self):
        """保存 FAISS 索引到文件"""
        if self.index is None or not self.index_path:
            return

        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, self.index_path)
        logger.info(f"FAISS Index saved to: {self.index_path}")

    def _load_vectors_from_db(self):
        """从 SQLite 加载历史向量到 FAISS 索引"""
        if self.conn is None:
            return

        sql = f"SELECT id, vector FROM {self.table_name} ORDER BY created_at"
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            return

        vectors = []
        for row in rows:# 
            doc_id = row["id"]
            # 看下面的 add 函数的 struct.pack() 的解释
            vector = struct.unpack(f"{self.dimension}f", row["vector"])
            vectors.append(vector)
            self.id_map[len(vectors) - 1] = doc_id

        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            self.index.add(vectors_np) 
            logger.info(f"Loading {len(vectors)} vectors to FAISS index")

    def add(
        self,
        id: str, # 这个是添加向量索引的文档 表示内容，用这个id来唯一标识每个向量
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加向量"""
        # 保存到数据库
        
        # vector_bytes 是 embedding （List[float]）的二进制表示 ：
        # - struct.pack() 将 Python 的浮点数列表转换为原始二进制字节
        # - "f" 表示单精度浮点数（4字节） 1024f 等价于 'f' * 1024，即告诉 pack 要处理 1024 个 f 格式的数据。 Python 的 float 是双精度（8 字节），但 pack 会将其转换为单精度存储。
        # - 如果 dimension=1024，转换后是 1024 × 4 = 4096 字节
        vector_bytes = struct.pack(f"{self.dimension}f", *embedding)
        metadata_json = json.dumps(metadata) if metadata else None

        sql = f"""
        INSERT OR REPLACE INTO {self.table_name}
        (id, content, metadata, vector, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        # vector_bytes: sqlite中存储的vector是 二进制的bytes 格式。是将list[float] 转换为原始二进制字节
        self.conn.execute(sql, (id, content, metadata_json, vector_bytes))
        self.conn.commit()

        # 添加到索引
        if self.index is not None:
            import numpy as np
            vector_np = np.array([embedding], dtype=np.float32)
            # faiss的add函数的参数必须是 numpy.ndarray，dtype=np.float32 形状： (n, dimension) ，n >= 1 即可
            # # 单条向量
            # embedding = [0.1, 0.2, ..., 0.768]  # List[float], 长度=dimension
            # vector_np = np.array([embedding], dtype=np.float32)  # shape: (1, 768)
            # index.add(vector_np)
            # 批量向量
            # vectors = [[...], [...], [...]]  # 多个 List[float]]
            # vectors_np = np.array(vectors, dtype=np.float32)  # shape: (n, 768)
            # index.add(vectors_np)

            self.index.add(vector_np)
            self.id_map[self.index.ntotal - 1] = id
            self.save_index()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        if self.index is None:
            # 使用简单搜索
            return self._simple_search(query_embedding, top_k)

        import numpy as np
        query_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                # 转换距离为相似度（L2距离越小越相似）
                similarity = 1.0 / (1.0 + dist)
                results.append((self.id_map[idx], similarity))

        return results

    def _simple_search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Tuple[str, float]]:
        """简单搜索（无 FAISS）"""
        sql = f"SELECT id, vector FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            return []

        import struct
        results = []
        for row in rows:
            vector = struct.unpack(f"{self.dimension}f", row["vector"])
            similarity = self._cosine_similarity(query_embedding, list(vector))
            results.append((row["id"], similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def clear(self):
        """清空"""
        if self.index is not None:
            self.index.reset()
        self.id_map.clear()
        sql = f"DELETE FROM {self.table_name}"
        self.conn.execute(sql)
        self.conn.commit()

        # 删除索引文件
        if self.index_path and Path(self.index_path).exists():
            Path(self.index_path).unlink()
            logger.info(f"Index file deleted from: {self.index_path}")

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取向量"""
        sql = f"SELECT * FROM {self.table_name} WHERE id = ?"
        cursor = self.conn.execute(sql, (id,))
        row = cursor.fetchone()

        if row is None:
            return None

        import struct
        vector = struct.unpack(f"{self.dimension}f", row["vector"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": metadata,
            "vector": list(vector),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_all_ids(self) -> List[str]:
        """获取所有向量ID"""
        sql = f"SELECT id FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        return [row["id"] for row in cursor.fetchall()]

    def count(self) -> int:
        """获取向量数量"""
        sql = f"SELECT COUNT(*) FROM {self.table_name}"
        cursor = self.conn.execute(sql)
        return cursor.fetchone()[0]

    def delete(self, id: str):
        """根据ID删除向量"""
        sql = f"DELETE FROM {self.table_name} WHERE id = ?"
        self.conn.execute(sql, (id,))
        self.conn.commit()
        self._rebuild_index()

    def delete_by_ids(self, ids: List[str]):
        """批量删除向量"""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        self.conn.execute(sql, ids)
        self.conn.commit()
        self._rebuild_index()

    def delete_by_prefix(self, prefix: str):
        """根据ID前缀删除向量"""
        sql = f"DELETE FROM {self.table_name} WHERE id LIKE ?"
        self.conn.execute(sql, (f"{prefix}%",))
        self.conn.commit()
        self._rebuild_index()
        logger.info(f"Deleted vectors with id prefix: {prefix}")

    def _rebuild_index(self):
        """重建 FAISS 索引"""
        if self.index is None:
            return

        self.index.reset()
        self.id_map.clear()

        sql = f"SELECT id, vector FROM {self.table_name} ORDER BY created_at"
        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            self.save_index()
            return

        vectors = []
        for row in rows:
            doc_id = row["id"]
            vector = struct.unpack(f"{self.dimension}f", row["vector"])
            vectors.append(vector)
            self.id_map[len(vectors) - 1] = doc_id

        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            self.index.add(vectors_np)

        self.save_index()
        logger.info(f"FAISS index rebuilt: {self.index.ntotal} vectors")
