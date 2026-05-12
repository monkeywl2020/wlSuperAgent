# -*- coding: utf-8 -*-
""" agent 导入的统一的模型交互接口 """
from typing import Dict, Any, Type

from .core.base import BaseChatModelClient, LLM_REGISTRY
from .openai_model_client import OpenAiChatModelClient
from .post_model_client import PostAPIChatModelClient

# 获取agent 类型
def get_chat_model_client(llmcfg: Dict[str, Any]) -> Type[BaseChatModelClient]:
    """根据 llmcfg 中的 model_type 从 LLM_REGISTRY 中获取对应的 LLM 客户端类
    
    Args:
        llmcfg (Dict[str, Any]): LLM 配置字典，需包含 'model_type' 键
        
    Returns:
        Type[BaseChatModelClient]: 对应的 LLM 客户端类
        
    Raises:
        KeyError: 如果 model_type 未在 LLM_REGISTRY 中注册
        ValueError: 如果 llmcfg 缺少 model_type 键
    """
    model_type = llmcfg.get('model_type')
    if not model_type:
        raise ValueError("llmcfg must contain 'model_type' to identify the LLM client")
    
    if model_type not in LLM_REGISTRY:
        raise KeyError(f"Model type '{model_type}' is not registered in LLM_REGISTRY. "
                       f"Available models: {list(LLM_REGISTRY.keys())}")
    
    return LLM_REGISTRY[model_type]

__all__ = [
    "OpenAiChatModelClient",
    "PostAPIChatModelClient",
]
