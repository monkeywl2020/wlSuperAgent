import copy
import json
import os
import random
import time
from abc import ABC, abstractmethod
from pprint import pformat
import asyncio
import traceback
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, AsyncIterator
from src.llm import get_chat_model_client
from src.utils.common import print_traceback
from src.tools.base import BaseTool,TOOL_REGISTRY


from src.settings import AgentConfig
from src.msg import Msg as Message
from src.msg import CustomBlock,TextBlock
from src.sandbox.sandbox import Sandbox
from loguru import logger

from .operator import Operator

#这是一个全局字典，用于存储注册的 agent 类，类的名称作为键，类的实例作为值。 AGENT_REGISTRY 用来保存已注册的所有agent。
AGENT_REGISTRY = {}

#注册agent，修饰符
def register_agent(agent_type):
    #agent不会太多，所以不用管冲突问题
    def decorator(cls):
        AGENT_REGISTRY[agent_type] = cls
        return cls

    return decorator

#这个基类，所有的 chat 类的 llm 适配的基类
class BaseAgent(Operator):
    """The base class of LLM"""

    #初始化
    def __init__(
        self,
        agent_config: AgentConfig, # 这个就是Agent 的配置
        sandbox: Sandbox,
        **kwargs: Any,
    ):
        logger.info("BaseAgent::__init__    --------- begin!")
        self.sandbox = sandbox # 挂载沙箱
        self.name = agent_config.agent_name
        self.sys_prompt_cn = agent_config.sys_prompt_cn # 中文提示词模版
        self.sys_prompt_en = agent_config.sys_prompt_en # 英文提示词模版
        self.llm_config = agent_config.llmcfg.model_dump() or {} # 转成dict
        self.description = agent_config.description
        self.tools_list = agent_config.tools_list or []  # 记录工具
        self.llm_client = None # 挂载模型客户端
        logger.info(f"BaseAgent::__init__    --------- sys_prompt_cn:{self.sys_prompt_cn}!")
        logger.info(f"BaseAgent::__init__    --------- sys_prompt_en:{self.sys_prompt_en}!")
        logger.info(f"BaseAgent::__init__    --------- tools_list:{self.tools_list}!")
        # 初始化传入的所有 tools
        self.function_map = {} # 存储工具实例的字典
        if self.tools_list:
            for tool in self.tools_list:
                self._init_tool(tool) # 初始化工具，agent的function_map下面挂着所有注册过的工具

        # 将 kwargs 中的所有参数转换为实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)  # 无条件覆盖
            
        # 初始化 LLM 客户端
        logger.info(f"BaseAgent::__init__    ---------llm_config type:{type(self.llm_config)}  llm_config:{self.llm_config}!")
        # 模型客户端的配置
        if self.llm_config is not False and isinstance(self.llm_config, dict):
            try:
                # 获取 LLM 客户端类并实例化
                llm_client_cls = get_chat_model_client(self.llm_config) 
                self.llm_client = llm_client_cls(config_list=self.llm_config)# 传入的是 llmcfg 模型客户端的配置
                logger.info(f"Initialized LLM client for {self.name} with model_type: {self.llm_config.get('model_type')}") 
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing LLM client: {e}")
                traceback.print_exc()
                raise

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    @abstractmethod
    async def a_generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any
        ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        raise NotImplementedError

    # 这个是访问大模型，获取模型的响应接口。
    @abstractmethod
    async def get_llm_reply(
        self,
        messages: List[Union[Message, Dict]],
        num_llm_calls_available: int ,
        tools: List[Dict],
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        raise NotImplementedError

    # 这个是将 大模型调用 tool call时候的消息进行规范化，发现qwq 调用tool的时候如果没有参数 function 字典会设置为none，正常应该是 {}
    def Normalize_tool_calls(self, tool_calls: List[Dict]) -> List[Any]:
        """原地修改 tool_calls，将 arguments 为 None 的替换为空字典"""
        for tc in tool_calls:
            function = tc.setdefault("function", {})  # 确保 function 存在
            if function.get("arguments") is None:
                function["arguments"] = {}

        return tool_calls

    #-----------------------------------------------------------------------------------------
    #                   ---** tool call的通用处理逻辑 **---
    #
    # 处理tool call的函数，解析tool call消息，这个消息是模型客户端返回的
    # 检测到后调用 tool ，并且将结果转成 Message 添加到传入的 messages 中去
     #-----------------------------------------------------------------------------------------
    async def _handle_tool_calls(
        self,
        responses: List[Message],
        **kwargs: Any
    ) -> Tuple[bool, Optional[Dict], List[Dict]]:
        """处理工具调用并返回工具调用消息和结果消息"""
        if not responses or responses[0].role != "function":
            return False, None, []

        use_tool, tools_info = self._detect_tool(responses[0])
        if not use_tool:
            return False, None, []

        logger.info(f"_handle_tool_calls=====================responses[0].tool_calls:{responses[0].tool_calls}")
        tool_call_normalized = self.Normalize_tool_calls(responses[0].tool_calls)# 规范化 tool call，将 arguments 为 None 的替换为空字典
        logger.info(f"_handle_tool_calls=====================tool_call_normalized:{tool_call_normalized}")
        tool_call_msg = {"role":"assistant","content":None,"tool_calls":tool_call_normalized} # 这个工具调用的信息
        logger.info(f"_handle_tool_calls=====================tool_call_msg:{tool_call_msg}")

        # 执行工具调用并生成结果消息
        tool_results = []
        for tool in tools_info:
            func_id, func_name, func_args, _ = tool
            # 函数调用都换成了异步调用
            tool_result = await self._a_call_tool(func_name, func_args, **kwargs)
            # 如果返回的是dict，要放到extra中。这个是切换agent function call设置的特殊格式，dict格式返回的
            if isinstance(tool_result, dict): # ppt_controller 的tool结果是dict
                fn_msg = {"role":"tool","content":'','id':func_id,"extra":tool_result}
            else:
                fn_msg = {"role":"tool","content":tool_result,"id":func_id}
            tool_results.append(fn_msg)

        #logger.info(f"BaseAgent::_handle_tool_calls    ---------tool_results:{tool_results}!")
        #===============================================================================================
        # 返回的 tool_call_msg 是{"role":"assistant","content":None,"tool_calls":tool_call_normalized}格式
        # tool_results 也是{"role":"tool","content":tool_result,"id":func_id}格式
        #  这两个返回的值都是 dict，可以直接添加到 用户输入的 messages 中去作为接下来的输入，让llm来处理tool调用的消息和tool结果消息
        #===============================================================================================
        return True, tool_call_msg, tool_results

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        rsp = self.genrate_rsp(*args, **kwargs)
        return rsp

    #------------------------------------------------
    # 同步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    @abstractmethod
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:

        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        raise NotImplementedError
    
    # 将消息转成对应的格式 message 或者 dict
    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    # 初始化工具，将工具实例化，挂到  function_map 中去，作为字段的方式存储，根据名称可以直接找到该工具
    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        logger.info(f"BaseAgent::_init_tool    --------- tool:【{tool}】")
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                # 默认我们在 使用qwen-agent的时候，调用传入的是一个字符串
                tool_name = tool #这个函数名字
                tool_cfg = None # 这个配置是空的
            
            logger.info(f'BaseAgent::_init_tool NUWA_TOOL_REGISTRY:{TOOL_REGISTRY}')
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            #这里是获取每个工具的类，并且将这个工具的类实例化，参数就是工具的配置 
            # 可以参考 retrieval.py中的 retrieval函数
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg) # 这里实际上就是对工具类进行实例化

    #检测当前的消息是否含有工具调用，并返回 函数名，参数和 调函数的时候描述(openAI没有，有些模型是有的)
    def _detect_tool(self, message: Message) -> Tuple[bool, List[Tuple[str, str, str]]]:
        """检测消息是否为工具调用，并返回工具调用的详细信息。

        Args:
            message: 由 LLM 生成的一条消息。

        Returns:
            Tuple[bool, List[List[Union[str, None]]]]:
                - 是否需要调用工具（bool）。
                - 工具调用详情列表，每个元素为 [func_name, func_args, text]。
                如果没有工具调用，返回空列表。
        """
        # 通过 role 判断是否为工具调用
        if message.role != "function":
            return False, []

        # 有些时候调用tool的时候会有对本次调用tool的描述。
        text = message.content if message.content else ''

        # 如果没有 tool_calls，返回 False 和空列表，获取 block_type 为 tool_use 的block 提取出来
        tool_calls = message.get_content_blocks(block_type = "tool_use")
        if not tool_calls:
            return False, []
        
        # 确保 tool_calls 是列表格式
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        tool_calls_info = []

        # 遍历每个 tool_call，提取详细信息
        for tool_call in tool_calls:
            # 假设 tool_call 是字典，包含 'function' 键
            if isinstance(tool_call, dict):
                # 这个是
                func_id = tool_call.get('id', None)
                #function_info = tool_call.get('function', {})
                func_name = tool_call.get('name')
                func_args = tool_call.get('input') # 这个 tool的dict转成 了msg的 ToolUseBlock 后的格式
            else:
                func_id =  getattr(tool_call, 'id', None)
                # 如果 tool_call 是对象，使用 getattr 获取属性
                func_name = getattr(tool_call, 'name', None)
                func_args = getattr(tool_call, 'input', None)

            # 将提取的信息添加到列表
            tool_calls_info.append([func_id, func_name, func_args, text])

        logger.info(f"BaseAgent::_detect_tool    ---------tool_calls_info:{tool_calls_info}!")
        # 返回是否有工具调用和工具调用信息列表
        return True, tool_calls_info

    # 调用函数，这个agent需要有的功能
    async def _a_call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, dict]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        # 检查tool是否为agent挂载的默认工具里面的
        if tool_name in self.function_map:
            logger.info(f"BaseAgent::_a_call_tool    ---------tool_name:{tool_name} is in self.function_map!")
            tool = self.function_map[tool_name]
            try:
                tool_result = await tool.call(tool_args, **kwargs)
            except Exception as ex:
                exception_type = type(ex).__name__
                exception_message = str(ex)
                traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
                error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                                f'{exception_type}: {exception_message}\n' \
                                f'Traceback:\n{traceback_info}'
                logger.warning(error_message)
                return error_message
        else:
            # 本地工具没有找到
            return f'Tool {tool_name} does not exists.'

        # 字符串 和 dict 都直接返回
        if isinstance(tool_result, str) or isinstance(tool_result, dict):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, dict) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    # 将迭代器中的消息转成对应的格式 message 或者 dict
    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)

    #------------------------------------------------
    # 暂不支持多模态输入
    #
    #------------------------------------------------
    @property
    def support_multimodal_input(self) -> bool:
        # Does the model support multimodal input natively? It affects how we preprocess the input.
        return False

    #------------------------------------------------
    # 暂不支持多模态输出
    #
    #------------------------------------------------
    @property
    def support_multimodal_output(self) -> bool:
        # Does the model generate multimodal outputs beyond texts? It affects how we post-process the output.
        return False

# 获取agent 类型， 这个是根据agent 的类型名称获取 agent类的类型，默认是 chat_agent 类型，获取出去后，要进行实例化（）
def get_chat_agent_cls(agent_type_name: Optional[str] = 'chat_agent') -> BaseAgent:
    """
    根据 agent_type_name 从 AGENT_REGISTRY 中获取对应的 Agent 类。
    
    Args:
        agent_type_name (Optional[str]): Agent 类型名称，默认值为 'chat_agent'。
    
    Returns:
        Type[BaseAgent]: 对应的 Agent 类。
    
    Raises:
        KeyError: 如果指定的 agent_type_name 未在 AGENT_REGISTRY 中注册。
    """
    if agent_type_name not in AGENT_REGISTRY:
        raise KeyError(f"Agent type '{agent_type_name}' is not registered in AGENT_REGISTRY. "
                       f"Available agents: {list(AGENT_REGISTRY.keys())}")
    # 返回类名
    return AGENT_REGISTRY[agent_type_name]