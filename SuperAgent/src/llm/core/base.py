from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, AsyncIterator, Union

from src.msg import Msg as Message
from src.tools.base import BaseTool
#from .modelpool_client import ModelPoolClient 

from loguru import logger

#这是一个全局字典，用于存储注册的大模型客户端类，类的名称作为键，类的实例作为值。 LLM_REGISTRY 用来保存已注册的所有大模型客户端。
LLM_REGISTRY = {}

# 注册大模型客户端类
def register_llm(model_type):
    def decorator(cls):
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator


class ModelServiceError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message

#这个基类，所有的 chat 类的 llm 适配的基类
class BaseChatModelClient(ABC):
    """The base class of LLM"""
    
    # 类级重试配置
    AUTO_RETRY_CONFIG = {
        'max_retries': 3,
        'base_delay': 0.5,
        'retryable_errors': (Exception,)
    }

    # llm 客户端适配
    def __init__(self, 
                 config_list: Dict[str, Any], 
                 **kwargs: Any):
        """初始化 LLM 客户端
        
        Args:
            config_list (Optional[List[Dict[str, Any]]]): LLM 配置列表
            **kwargs: 任意关键字参数，将转换为实例属性
        """
        # 存储 config_list
        self.llm_cfg = config_list
        
        # 将 kwargs 中的所有参数转换为实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)  # 无条件覆盖

        logger.info(f"BaseChatModelClient::__init__ config_list: {config_list} model_type:{self.llm_cfg['model_type']} model_name:{self.llm_cfg['model']} base_url:{self.llm_cfg['base_url']}")

    @abstractmethod
    async def a_generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            tools: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """生成响应（抽象方法，由子类实现）
        
        Args:
            messages: 输入消息列表
            tools: 可选的功能列表
            stream: 是否使用流式输出
            delta_stream: 是否使用增量流式输出
            **kwargs: 其他参数
        
        Returns:
            Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]: 响应内容
        """
        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        raise NotImplementedError
        
    @abstractmethod
    async def parse_stream_response_async(self, chunks: Any) -> AsyncIterator[List[Message]]:
        """解析 OpenAI 非流式响应的结果"""
        raise NotImplementedError

    ##############################################
    # 同步接口，生成响应
    #
    ##############################################
    @abstractmethod
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            tools: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """生成响应（抽象方法，由子类实现）
        
        Args:
            messages: 输入消息列表
            tools: 可选的功能列表
            stream: 是否使用流式输出
            delta_stream: 是否使用增量流式输出
            **kwargs: 其他参数
        
        Returns:
            Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]: 响应内容
        """
        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        raise NotImplementedError

    #-----------------------------------------------
    # 解析模型的响应
    #   1：parse_stream_response 解析流式响应（同步接口）
    #   2：parse_stream_response_async  解析流式响应（异步接口）
    #   3：parse_non_stream_response 解析非流式响应（同步接口）
    #   流式分成同步和异步是因为迭代器分别是 Iterator 和 AsyncIterator，非流式没这个要求
    #-----------------------------------------------
    @abstractmethod
    def parse_stream_response(self, chunks: Any) -> Iterator[List[Message]]:
        """解析 OpenAI 非流式响应的结果"""
        raise NotImplementedError
       
    @abstractmethod
    def parse_non_stream_response(self, chunk: Any) -> List[Message]:
        raise NotImplementedError

    #-----------------------------------------------
    # 将 NuwaBaseTool 转换为 OpenAI 支持的工具格式
    #-----------------------------------------------
    @abstractmethod
    def convert_nuwa_tool_to_model_support(self, nuwa_tools: Union[Dict, List[BaseTool]]) -> List[Dict]:
        """将 NuwaBaseTool 格式转换为 OpenAI 支持的工具格式"""
        raise NotImplementedError