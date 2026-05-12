# -*- coding: utf-8 -*-
"""
Embedding 接口模块

提供统一的 Embedding 接口，支持多种提供商
"""

# 导入标准库
from abc import ABC, abstractmethod  # 抽象基类
from typing import List, Optional, Dict, Any  # 类型注解
import numpy as np  # 数值计算

# 导入第三方库
from loguru import logger  # 日志记录


class EmbeddingProvider(ABC):
    """Embedding 提供商抽象基类"""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的 Embedding

        Args:
            text: 输入文本

        Returns:
            Embedding 向量
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本的 Embedding

        Args:
            texts: 输入文本列表

        Returns:
            Embedding 向量列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """获取向量维度"""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地 Embedding 提供商（使用 sentence-transformers）"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        初始化本地 Embedding 提供商

        Args:
            model_name: 模型名称
            device: 设备 (cpu/cuda)
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Local embedding model loaded: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using random vectors")
            self.model = None

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 Embedding"""
        if self.model is None:
            return self._random_embedding()
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 Embedding"""
        if self.model is None:
            return [self._random_embedding() for _ in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def _random_embedding(self) -> List[float]:
        """生成随机向量（降级方案）"""
        import random
        return [random.random() for _ in range(self.dimension)]

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self.model is None:
            return 384
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI Embedding 提供商"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        初始化 OpenAI Embedding 提供商

        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 地址
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"OpenAI embedding client initialized: {self.model_name}")
        except ImportError:
            logger.warning("openai library not installed")
            self.client = None

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 Embedding"""
        if self.client is None:
            return self._random_embedding()
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 Embedding"""
        if self.client is None:
            return [self._random_embedding() for _ in texts]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        # OpenAI embedding 模型维度
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model_name, 1536)

    def _random_embedding(self) -> List[float]:
        """生成随机向量（降级方案）"""
        import random
        return [random.random() for _ in range(self.dimension)]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace Embedding 提供商（使用 transformers）"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        初始化 HuggingFace Embedding 提供商

        Args:
            model_name: 模型名称
            device: 设备
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info(f"HuggingFace model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"HuggingFace model load failed: {e}")
            self.model = None

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 Embedding"""
        if self.model is None:
            return self._random_embedding()
        
        import torch
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 Embedding"""
        if self.model is None:
            return [self._random_embedding() for _ in texts]
        
        import torch
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self.model is None:
            return 384
        return self.model.config.hidden_size

    def _random_embedding(self) -> List[float]:
        """生成随机向量（降级方案）"""
        import random
        return [random.random() for _ in range(self.dimension)]


class TEIEmbeddingProvider(EmbeddingProvider):
    """TEI (Text Embeddings Inference) Embedding 提供商

    TEI 是 HuggingFace 提供的推理服务，通过 HTTP API 提供 embedding 功能。
    Embedding模型下载在/models/Qwen3-Embedding-0.6B 目录中，text-embeddings-inference:cuda-1.9 是TEI的版本。
    启动 TEI 服务示例:
        docker run --gpus '"device=0"' -p 60010:60010 -v /models/Qwen3-Embedding-0.6B:/models/Qwen3-Embedding-0.6B --pull never ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id /models/Qwen3-Embedding-0.6B --port 60010 
    """

    def __init__(
        self,
        base_url: str = "http://localhost:60010",
        dimension: int = 1024
    ):
        """
        初始化 TEI Embedding 提供商

        Args:
            base_url: TEI 服务地址
            dimension: 向量维度
        """
        self.base_url = base_url.rstrip("/")
        self.dimension_value = dimension
        self.session = None

    def _get_session(self):
        """获取 HTTP 会话"""
        if self.session is None:
            import requests
            self.session = requests.Session()
        return self.session

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 Embedding"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else self._random_embedding()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 Embedding"""
        import requests

        try:
            session = self._get_session()
            response = session.post(
                f"{self.base_url}/embed",
                json={"inputs": texts},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result
                else:
                    return [result]

            logger.warning(f"TEI response format error: {result}")
            return []

        except ImportError:
            logger.warning("requests library not installed, cannot use TEI")
            return []

        except requests.RequestException as e:
            logger.error(f"TEI request failed: {e}")
            return []

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.dimension_value

    def _random_embedding(self) -> List[float]:
        """生成随机向量（降级方案）"""
        import random
        return [random.random() for _ in range(self.dimension)]


def create_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    """
    根据配置创建 Embedding 提供商

    Args:
        config: 配置字典

    Returns:
        EmbeddingProvider 实例
    """
    provider = config.get("provider", "local")
    
    if provider == "local":
        logger.info(f"create_embedding_provider type: local, config: {provider}")
        return LocalEmbeddingProvider(
            model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=config.get("device", "cpu"),
            batch_size=config.get("batch_size", 32)
        )
    elif provider == "openai":
        logger.info(f"create_embedding_provider type: openai, config: {provider}")
        return OpenAIEmbeddingProvider(
            model_name=config.get("model_name", "text-embedding-3-small"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
    elif provider == "huggingface":
        logger.info(f"create_embedding_provider type: huggingface, config: {provider}")
        return HuggingFaceEmbeddingProvider(
            model_name=config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=config.get("device", "cpu")
        )
    elif provider == "tei":# text-embeddings-inference  这个是hunggingface的 TEI 服务
        logger.info(f"create_embedding_provider type: tei, config: {provider}")
        return TEIEmbeddingProvider(
            base_url=config.get("base_url", "http://localhost:60010"),
            dimension=config.get("dimension", 1024)
        )
    else:
        logger.warning(f"Unknown embedding provider: {provider}, using local default")
        return LocalEmbeddingProvider()
