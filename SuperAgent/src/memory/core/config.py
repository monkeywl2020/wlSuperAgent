# -*- coding: utf-8 -*-
"""
核心模块 - 配置管理

提供 YAML 配置加载和管理功能
"""

# 导入标准库
from dataclasses import dataclass, field  # 数据类装饰器
from typing import Any, Dict, Optional, Union  # 类型注解
from pathlib import Path  # 路径操作
import os  # 环境变量

# 导入第三方库
from loguru import logger  # 日志记录

# 尝试导入 yaml
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed, using JSON config")


@dataclass
class ToolResultConfig:
    """工具结果压缩配置"""
    threshold: int = 10000  # 工具结果压缩阈值
    retention_days: int = 30  # 保留天数


@dataclass
class SearchConfig:
    """搜索配置"""
    vector_weight: float = 0.7  # 向量权重
    bm25_weight: float = 0.3  # BM25权重
    top_k: int = 5  # 返回结果数
    enable_qa_extraction: bool = True  # 是否启用QA提取
    chunk_size: int = 500  # 分块大小（字符数）
    overlap: int = 50  # 分块重叠大小


@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    db_path: str = "./memory_data/vectors.db"  # 数据库路径
    table_name: str = "embeddings"  # 表名
    dimension: int = 384  # 向量维度


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    provider: str = "local"  # 提供商
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 模型名称
    device: str = "cpu"  # 设备
    batch_size: int = 32  # 批处理大小
    dimension: int = 1024  # 向量维度


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "openai"  # 提供商
    model: str = "gpt-4"  # 模型
    api_key: str = ""  # API密钥
    base_url: str = ""  # API地址
    max_tokens: int = 4000  # 最大token
    top_p: float = 0.9  # 最大输出token概率
    temperature: float = 0.7  # 温度
    presence_penalty: float = 0.0  # 存在惩罚
    extra_body: Dict[str, Any] = field(default_factory=dict)  # 额外请求体

@dataclass
class MemoryConfig:
    """记忆系统配置"""
    working_dir: str = "./workspace/memory_data"  # 工作目录
    max_input_length: int = 128000  # 最大输入token数
    compact_ratio: float = 0.8  # 压缩比例
    tool_result: ToolResultConfig = field(default_factory=ToolResultConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    compress_prompt: str = ""  # 压缩提示词
    summarize_prompt: str = ""  # 摘要提示词
    longterm_prompt: str = ""  # 长期记忆提示词
    qa_extract_prompt: str = ""  # QA提取提示词


def load_yaml_config(config_path: str = "") -> MemoryConfig:
    """
    从 YAML 文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        MemoryConfig 配置对象
    """
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed, cannot read YAML config")
        return MemoryConfig()

    # 如果没有提供 config_path，使用相对于项目根目录的路径
    if not config_path:
        # 获取当前文件的绝对路径
        current_file = Path(__file__)
        # 向上三级到达 SuperAgent 目录，再向上一级到达项目根目录
        project_root = current_file.parent.parent.parent.parent.parent
        logger.info(f"===============project_root: {project_root}")
        # 构建配置文件路径
        config_path = str(project_root / "superagentconfig" / "memory_config.yaml")
        logger.info(f"===============config_path: {config_path}")

    config_file = Path(config_path)
    if not config_file.exists():
        logger.info(f"Config file not found: {config_path}, using default config")
        return MemoryConfig()

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 构建嵌套配置对象
        tool_result = ToolResultConfig(**config_dict.get("tool_result", {}))
        search = SearchConfig(**config_dict.get("search", {}))
        vector_db = VectorDBConfig(**config_dict.get("vector_db", {}))
        embedding = EmbeddingConfig(**config_dict.get("embedding", {}))
        llm = LLMConfig(**config_dict.get("llm", {}))

        config = MemoryConfig(
            working_dir=config_dict.get("working_dir", "./memory_data"),
            max_input_length=config_dict.get("max_input_length", 128000),
            compact_ratio=config_dict.get("compact_ratio", 0.8),
            tool_result=tool_result,
            search=search,
            vector_db=vector_db,
            embedding=embedding,
            llm=llm,
            compress_prompt=config_dict.get("compress_prompt", ""),
            summarize_prompt=config_dict.get("summarize_prompt", ""),
            longterm_prompt=config_dict.get("longterm_prompt", ""),
            qa_extract_prompt=config_dict.get("qa_extract_prompt", ""),
        )

        logger.info(f"Config loaded: {config_path}")
        return config

    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using default config")
        return MemoryConfig()


def save_yaml_config(config: MemoryConfig, config_path: str = ""):
    """
    保存配置到 YAML 文件

    Args:
        config: MemoryConfig 配置对象
        config_path: 配置文件路径
    """
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not installed, cannot save YAML config")
        return

    # 如果没有提供 config_path，使用相对于项目根目录的路径
    if not config_path:
        # 获取当前文件的绝对路径
        current_file = Path(__file__)
        # 向上三级到达 SuperAgent 目录，再向上一级到达项目根目录
        project_root = current_file.parent.parent.parent.parent.parent
        # 构建配置文件路径
        config_path = str(project_root / "superagentconfig" / "memory_config.yaml")

    config_dict = {
        "working_dir": config.working_dir,
        "max_input_length": config.max_input_length,
        "compact_ratio": config.compact_ratio,
        "tool_result": {
            "threshold": config.tool_result.threshold,
            "retention_days": config.tool_result.retention_days,
        },
        "search": {
            "vector_weight": config.search.vector_weight,
            "bm25_weight": config.search.bm25_weight,
            "top_k": config.search.top_k,
            "enable_qa_extraction": config.search.enable_qa_extraction,
            "chunk_size": config.search.chunk_size,
            "overlap": config.search.overlap,
        },
        "vector_db": {
            "db_path": config.vector_db.db_path,
            "table_name": config.vector_db.table_name,
            "dimension": config.vector_db.dimension,
        },
        "embedding": {
            "provider": config.embedding.provider,
            "model_name": config.embedding.model_name,
            "device": config.embedding.device,
            "batch_size": config.embedding.batch_size,
        },
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
        },
        "compress_prompt": config.compress_prompt,
        "summarize_prompt": config.summarize_prompt,
        "longterm_prompt": config.longterm_prompt,
        "qa_extract_prompt": config.qa_extract_prompt,
    }

    # 确保目录存在
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"Config saved: {config_path}")


# 兼容旧接口
def load_config(config_path: str = "") -> MemoryConfig:
    """加载配置（兼容旧接口）"""
    return load_yaml_config(config_path)


def save_config(config: MemoryConfig, config_path: str = ""):
    """保存配置（兼容旧接口）"""
    save_yaml_config(config, config_path)
