# -*- coding: utf-8 -*-
"""
集中配置管理模块

功能说明：
    统一管理所有配置文件的加载和访问
    - 主配置 (config.yaml): Agent核心配置
    - Memory配置 (memory_config.yaml): 记忆系统配置
    - Sandbox配置 (sandbox.yaml): 沙箱容器配置
    - Scheduler配置 (scheduler_config.yaml): 调度服务配置

使用方式：
    from src.settings import settings, memory_settings, sandbox_settings, scheduler_settings
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from loguru import logger

# 配置目录路径
CONFIG_DIR = Path(__file__).parent.parent.parent / "superagentconfig"

# ================== 配置文件路径定义 ==================
# 所有配置文件都放在 superagentconfig 目录下

CONFIG_FILE = CONFIG_DIR / "config.yaml"                    # 主配置文件
MEMORY_CONFIG_FILE = CONFIG_DIR / "memory_config.yaml"     # Memory记忆系统配置
SANDBOX_CONFIG_FILE = CONFIG_DIR / "sandbox.yaml"         # Sandbox沙箱容器配置
SCHEDULER_CONFIG_FILE = CONFIG_DIR / "scheduler_config.yaml"  # Scheduler调度服务配置


# ================== 嵌套配置模型定义 ==================
# 这些是配置文件中嵌套对象的结构定义

class LLMConfig(BaseModel):
    """LLM大模型配置"""
    model_type: str           # 模型类型
    model: str               # 模型名称
    base_url: str            # API地址
    api_key: str             # API密钥
    top_p: Optional[float] = None           # Top-p采样参数
    temperature: Optional[float] = None      # 温度参数
    presence_penalty: Optional[float] = None # 存在惩罚参数
    extra_body: Optional[Dict[str, Any]] = None  # 额外请求体

class AgentConfig(BaseModel):
    """Agent智能体配置"""
    agent_type: str                          # Agent类型
    agent_name: str                          # Agent名称
    sys_prompt_cn: str                      # 中文系统提示词
    sys_prompt_en: str                      # 英文系统提示词
    tools_list: List[str] = []              # 工具列表
    description: str                         # Agent描述
    extra: Optional[str] = None              # 额外配置
    llmcfg: LLMConfig                       # LLM配置

class NuwaAgentCfg(BaseModel):
    """NuwaAgent配置容器"""
    agentcfg: List[AgentConfig]              # Agent配置列表

class ChatTemplateAction(BaseModel):
    """聊天模板动作配置"""
    skill_context_perfix: Optional[str] = None     # 技能上下文前缀
    skill_select_perfix: Optional[str] = None      # 技能选择前缀
    workspace_perfix: Optional[str] = None         # 工作空间前缀
    sandbox_perfix: Optional[str] = None            # 沙箱前缀
    timezone_perfix: Optional[str] = None          # 时区前缀
    project_context_perfix: Optional[str] = None   # 项目上下文前缀

class ChatTemplate(BaseModel):
    """聊天模板配置"""
    cn: ChatTemplateAction                   # 中文模板
    en: ChatTemplateAction                   # 英文模板

class ModelPoolSrv(BaseModel):
    """模型服务池配置"""
    primary: str                            # 主服务地址
    secondary: str                          # 备用服务地址


# ================== 主 Settings 模型 ==================
# 对应 config.yaml 配置文件

class Settings(BaseModel):
    """主配置模型 - 对应 config.yaml"""
    nuwaagentcfg: NuwaAgentCfg              # Agent配置列表
    chat_template: ChatTemplate             # 聊天模板
    file_log_enable: bool                   # 是否启用文件日志
    agent_listen_port: int                  # Agent监听端口
    gradio_test_port: int                   # Gradio测试端口
    sandbox_container_name: str             # 沙箱容器名称

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        从YAML文件加载主配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            Settings配置实例
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"配置文件为空: {yaml_path}")
        return cls(**config_data)

    class Config:
        extra = "ignore"


# ================== Memory 记忆系统配置模型 ==================
# 对应 memory_config.yaml 配置文件

class ToolResultConfig(BaseModel):
    """工具结果压缩配置"""
    threshold: int = 1000        # 压缩阈值（字符数），超过该长度则压缩
    retention_days: int = 30    # 保留天数

class SearchConfig(BaseModel):
    """搜索功能配置"""
    vector_weight: float = 0.7           # 向量搜索权重
    bm25_weight: float = 0.3             # BM25关键词搜索权重
    top_k: int = 5                       # 返回前k个结果
    enable_qa_extraction: bool = True    # 是否启用QA提取
    chunk_size: int = 500                 # 分块大小（字符数）
    overlap: int = 50                     # 分块重叠大小

class VectorDBConfig(BaseModel):
    """向量数据库配置"""
    db_path: str = "./workspace/memory_data/vectors.db"  # 数据库路径
    table_name: str = "embeddings"        # 表名
    dimension: int = 1024                 # 向量维度

class EmbeddingConfig(BaseModel):
    """Embedding向量模型配置"""
    provider: str = "tei"                 # 提供商
    model_name: str = "Qwen3-Embedding-0.6B"  # 模型名称
    device: str = "gpu"                   # 运行设备
    batch_size: int = 32                  # 批处理大小
    dimension: int = 1024                 # 向量维度

class LLMConfigMemory(BaseModel):
    """Memory使用的LLM配置"""
    provider: str = "openai"              # 提供商
    model: str = "nuwa"                   # 模型名称
    api_key: str = "EMPTY"                # API密钥
    base_url: str = ""                    # API地址
    max_tokens: int = 40960                # 最大token数
    top_p: float = 0.95                   # Top-p采样
    temperature: float = 0.6              # 温度参数
    presence_penalty: float = 0.0         # 存在惩罚
    extra_body: Optional[Dict[str, Any]] = None  # 额外请求体

class MemoryConfig(BaseModel):
    """Memory记忆系统主配置"""
    working_dir: str = "./workspace/memory_data"  # 工作目录
    max_input_length: int = 128000        # 最大输入token数
    compact_ratio: float = 0.8             # 压缩比例
    tool_result: ToolResultConfig = ToolResultConfig()       # 工具结果配置
    search: SearchConfig = SearchConfig()  # 搜索配置
    vector_db: VectorDBConfig = VectorDBConfig()  # 向量数据库配置
    embedding: EmbeddingConfig = EmbeddingConfig()  # Embedding配置
    llm: LLMConfigMemory = LLMConfigMemory()  # LLM配置
    compress_prompt: str = ""              # 压缩提示词
    summarize_prompt: str = ""            # 摘要提示词
    longterm_prompt: str = ""             # 长期记忆提示词
    qa_extract_prompt: str = ""           # QA提取提示词

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        从YAML文件加载Memory配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            MemoryConfig配置实例
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"配置文件为空: {yaml_path}")
        return cls(**config_data)


# ================== Sandbox 沙箱配置模型 ==================
# 对应 sandbox.yaml 配置文件

class SandboxConfig(BaseModel):
    """Sandbox沙箱容器配置"""
    container_name: str = "agent-sandbox"     # Docker容器名称
    image_name: str = "python:3.12-slim"     # Docker镜像名称
    container_workspace: str = "/workspace"   # 容器内工作目录
    memory_limit: str = "1024m"               # 内存限制
    cpus: int = 1                            # CPU核心数
    network: str = "bridge"                  # 网络模式
    workspace_subdirs: List[str] = []        # 工作目录子文件夹列表

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        从YAML文件加载Sandbox配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            SandboxConfig配置实例
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"配置文件为空: {yaml_path}")
        return cls(**config_data)


# ================== Scheduler 调度服务配置模型 ==================
# 对应 scheduler_config.yaml 配置文件

class SchedulerConfig(BaseModel):
    """Scheduler调度服务配置"""
    enabled: bool = True                     # 是否启用调度器
    standalone: bool = True                 # 是否独立启动（True=独立进程，False=作为子模块）
    port: int = 7702                         # 服务端口
    host: str = "0.0.0.0"                   # 服务地址
    file_log_enable: bool = False           # 是否启用文件日志
    log_dir: str = "./schedulerlog"         # 日志目录
    log_file: str = "scheduler.log"        # 日志文件名
    max_log_size: int = 10485760            # 单个日志文件最大大小（10MB）
    log_retention: int = 30                 # 日志保留天数
    default_superagent_url: str = "http://localhost:7701"  # 默认SuperAgent地址
    default_jobs: List[Dict[str, Any]] = []  # 默认定时任务列表

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        从YAML文件加载Scheduler配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            SchedulerConfig配置实例
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            raise ValueError(f"配置文件为空: {yaml_path}")
        return cls(**config_data.get("scheduler", {}))


# ================== 加载所有配置 ==================

def _load_all_configs():
    """
    加载所有配置文件

    功能说明：
        - 主配置加载失败会抛出异常终止程序
        - 其他配置加载失败只会记录警告，不会终止程序

    Returns:
        dict: 包含所有配置实例的字典
    """
    configs = {}

    # 加载主配置（必须成功）
    try:
        configs["main"] = Settings.from_yaml(CONFIG_FILE)
        logger.info(f"主配置加载成功: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"主配置加载失败: {e}")
        raise

    # 加载Memory配置（可选）
    try:
        configs["memory"] = MemoryConfig.from_yaml(MEMORY_CONFIG_FILE)
        logger.info(f"Memory配置加载成功: {MEMORY_CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Memory配置加载失败: {e}")

    # 加载Sandbox配置（可选）
    try:
        configs["sandbox"] = SandboxConfig.from_yaml(SANDBOX_CONFIG_FILE)
        logger.info(f"Sandbox配置加载成功: {SANDBOX_CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Sandbox配置加载失败: {e}")

    # 加载Scheduler配置（可选）
    try:
        configs["scheduler"] = SchedulerConfig.from_yaml(SCHEDULER_CONFIG_FILE)
        logger.info(f"Scheduler配置加载成功: {SCHEDULER_CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Scheduler配置加载失败: {e}")

    return configs


# 模块加载时自动加载所有配置
_all_configs = _load_all_configs()


# ================== 全局配置单例 ==================
# 供其他模块直接导入使用

settings:Settings = _all_configs.get("main")           # 主配置实例
memory_settings:MemoryConfig = _all_configs.get("memory")   # Memory配置实例
sandbox_settings:SandboxConfig = _all_configs.get("sandbox") # Sandbox配置实例
scheduler_settings:SchedulerConfig = _all_configs.get("scheduler")  # Scheduler配置实例


# ================== 路径定义 ==================
# 供其他模块使用

CURRENT_DIR = Path(__file__).parent                  # 当前文件所在目录
PARENT_DIR = CURRENT_DIR.parent                       # 父目录
PROJECT_ROOT = PARENT_DIR.parent                      # 项目根目录

CURRENT_FILE_ABSOLUTE_PATH = Path(__file__).absolute().parent     # 当前文件绝对路径
PARENT_DIR_ABSOLUTE_PATH = CURRENT_FILE_ABSOLUTE_PATH.parent      # 父目录绝对路径
PROJECT_ROOT_ABSOLUTE_PATH = PARENT_DIR_ABSOLUTE_PATH.parent     # 项目根目录绝对路径
