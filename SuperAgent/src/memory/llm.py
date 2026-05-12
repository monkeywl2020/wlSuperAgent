# -*- coding: utf-8 -*-
"""
LLM 接口模块

提供统一的 LLM 接口，用于压缩和摘要生成
"""

# 导入标准库
from abc import ABC, abstractmethod  # 抽象基类
from typing import List, Dict, Any, Optional, Callable  # 类型注解

# 导入第三方库
from loguru import logger  # 日志记录


class LLMProvider(ABC):
    """LLM 提供商抽象基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            **kwargs: 其他参数

        Returns:
            模型回复内容
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM 提供商"""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any # llm支持其他参数
    ):
        """
        初始化 OpenAI LLM 提供商

        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 地址
            temperature: 默认温度参数
            max_tokens: 默认最大 token 数
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.max_tokens = kwargs.get("max_tokens", 4000)
        self.top_p = kwargs.get("top_p", 0.9)
        self.temperature = kwargs.get("temperature", 0.7)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)
        self.extra_body = kwargs.get("extra_body", dict())

        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"OpenAI client initialized: {self.model}")
        except ImportError:
            logger.warning("openai library not installed")
            self.client = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """发送聊天请求"""
        if self.client is None:
            return "[LLM 未可用]"

        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        logger.info(f"OpenAIProvider::chat start, model: {self.model}, temperature: {temperature}, max_tokens: {max_tokens}, messages: {messages}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream = False,
            **kwargs
        )

        logger.info(f"OpenAIProvider::chat response: {response}")

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic LLM 提供商"""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        初始化 Anthropic LLM 提供商

        Args:
            model: 模型名称
            api_key: API 密钥
            temperature: 默认温度参数
            max_tokens: 默认最大 token 数
        """
        self.model = model
        self.api_key = api_key
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 Anthropic 客户端"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized: {self.model}")
        except ImportError:
            logger.warning("anthropic library not installed")
            self.client = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """发送聊天请求"""
        if self.client is None:
            return "[LLM 未可用]"

        # 转换消息格式
        system = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=filtered_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.content[0].text


class LocalLLMProvider(LLMProvider):
    """本地 LLM 提供商（支持 Ollama 等）"""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        初始化本地 LLM 提供商

        Args:
            model: 模型名称
            base_url: API 地址
            api_key: API 密钥
            temperature: 默认温度参数
            max_tokens: 默认最大 token 数
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Local LLM client initialized: {self.model}")
        except ImportError:
            logger.warning("openai library not installed")
            self.client = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """发送聊天请求"""
        if self.client is None:
            return "[LLM 未可用]"

        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            return "[LLM 调用失败]"


class CallableLLMProvider(LLMProvider):
    """可调用 LLM 提供商（使用自定义函数）"""

    def __init__(
        self,
        callable_func: Callable,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4000
    ):
        """
        初始化可调用 LLM 提供商

        Args:
            callable_func: 自定义函数，签名: func(messages: List[Dict]) -> str
            default_temperature: 默认温度参数
            default_max_tokens: 默认最大 token 数
        """
        self.callable_func = callable_func
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """调用自定义函数"""
        try:
            return self.callable_func(messages, **kwargs)
        except Exception as e:
            logger.error(f"Custom LLM function call failed: {e}")
            return "[LLM 调用失败]"


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    根据配置创建 LLM 提供商

    Args:
        config: 配置字典

    Returns:
        LLMProvider 实例
    """
    provider = config.get("provider", "openai")

    if provider == "openai":
        logger.info(f"create_llm_provider type: openai, config: {provider}")
        return OpenAIProvider(
            model=config.get("model", "gpt-4"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            max_tokens=config.get("max_tokens", 4000),
            top_p=config.get("top_p", 0.9),
            temperature=config.get("temperature", 0.7),
            presence_penalty=config.get("presence_penalty", 0.0),
            extra_body=config.get("extra_body", dict())
        )
    elif provider == "anthropic":
        logger.info(f"create_llm_provider type: anthropic, config: {provider}")
        return AnthropicProvider(
            model=config.get("model", "claude-3-sonnet-20240229"),
            api_key=config.get("api_key", ""),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4000)
        )
    elif provider == "local":
        logger.info(f"create_llm_provider type: local, config: {provider}")
        return LocalLLMProvider(
            model=config.get("model", "llama2"),
            base_url=config.get("base_url", "http://localhost:11434/v1"),
            api_key=config.get("api_key", "ollama"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4000)
        )
    elif provider == "callable":
        logger.info(f"create_llm_provider type: callable, config: {provider}")
        return create_callable_llm_provider(
            func=config.get("func"),
            default_temperature=config.get("temperature", 0.7),
            default_max_tokens=config.get("max_tokens", 4000)
        )
    else:
        logger.warning(f"Unsupported LLM Provider: {provider}, switching to OpenAI defaults")
        return OpenAIProvider(
            model=config.get("model", "gpt-4"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            max_tokens=config.get("max_tokens", 4000),
            top_p=config.get("top_p", 0.9),
            temperature=config.get("temperature", 0.7),
            presence_penalty=config.get("presence_penalty", 0.0),
            extra_body=config.get("extra_body", dict())
        )


def create_callable_llm_provider(
    func: Callable,
    default_temperature: float = 0.7,
    default_max_tokens: int = 4000
) -> CallableLLMProvider:
    """
    创建可调用的 LLM 提供商

    Args:
        func: 自定义函数
        default_temperature: 默认温度
        default_max_tokens: 默认最大 token

    Returns:
        CallableLLMProvider 实例
    """
    return CallableLLMProvider(
        callable_func=func,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens
    )
