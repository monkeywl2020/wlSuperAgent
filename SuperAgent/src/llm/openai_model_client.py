# -*- coding: utf-8 -*-
import openai
from openai import APIConnectionError, APITimeoutError
import json
import asyncio
import time
import re
from loguru import logger

from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence,Iterator,AsyncIterator
from .core.base import BaseChatModelClient, register_llm
from ..tools.base import BaseTool
from ..msg import  Msg as Message
from ..msg._message_block import ThinkingBlock,TextBlock,ToolUseBlock
from ..utils.retry_utils import auto_retry 

USE_MOCK = 0
#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
@register_llm('openai')
class OpenAiChatModelClient(BaseChatModelClient):

    # 自定义重试策略
    AUTO_RETRY_CONFIG = {
        **BaseChatModelClient.AUTO_RETRY_CONFIG,
        'retryable_errors': (APIConnectionError, APITimeoutError)# 连接错误和超时才重试
    }
    # llm 客户端适配
    def __init__(self, 
                 config_list: Optional[Dict[str, Any]] = None, 
                 **kwargs: Any):
        logger.info("OpenAiChatModelClient-----------__init__")
        if config_list:
            # 如果有内容，直接展开作为参数传入 BaseChatModelClient 作为关键字保存
            super().__init__(config_list=config_list, **config_list, **kwargs)
        else:
            super().__init__(config_list=config_list, **kwargs)

        logger.info(f"OpenAiChatModelClient-----------__init__  api_key:{self.api_key} base_url:{self.base_url}")
 
        #初始化openAI 异步客户端 这个放到 BaseChatModelClient 中 初始化 OAIModelPoolClientManager 的时候初始化了，所以这个不需要了
        self.async_client = openai.AsyncOpenAI(
            api_key= self.api_key,
            organization=None,
            base_url = self.base_url
        )

        # 设置 max_tokens，默认值 4096
        self.max_tokens = kwargs.get('max_tokens', 40960)

    async def _mock_stream_response(self, openAIkwargs: Dict) -> AsyncIterator:
        """模拟 OpenAI 流式响应的桩函数，使用分词逐词输出"""
        logger.info("Using MOCK response generator (token-based)")
        import datetime
        #import jieba
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")# 记录调用模型的开始时间

        # # 默认内容 - 使用了特殊格式包含websearch动作
        # content = "<think>用户要求生成一段500-600字的中文文字内容，用于游戏参与。Action:Normal</think>" \
        #         "<emo>专业</emo>数字人技术在展厅讲解中发挥重要作用，通过多模态交互提升访客体验。Nuwa平台创建的数字人" \
        #         "具备情商脑和智商脑协同工作能力，突破传统局限，提供情绪和实用价值。应用包括网页嵌入、终端设备和定" \
        #         "制终端如柜机大屏，支持声音处理。管理功能涵盖流量监控、对话提取、用户画像分析和体验反馈。创建过程" \
        #         "需角色设计、形象克隆和实时生成，步骤包括性格、职位、技能选择，以及容貌和声音定制。平台集成企业知" \
        #         "识问答、数据库搜索、Web检索和内容创作，支持文字、语音和视觉交互。数字人应用广泛，从教育到医疗，" \
        #         "从文化到制造业，都能快速构建专属数字员工或助手。"       
        # # 使用jieba分词
        # words = list(jieba.cut(content, cut_all=False))
        wl_current_time = f"当前时间：{current_time}"

        words = ['<think>', '用户', '要求', '生成', '一段', '500', '-', '600', '字', '的', '中文', '文字', '内容', '，', '用于', '游戏', '参与', '。', 
        'Action', ':', 'Normal', '</think>', '<emo>', '专业', '</emo>', '数字', '人', '技术', '在', '展厅', '讲解', '中', '发挥', '重要', '作用', '，', 
        '通过', '多', '模态', '交互', '提升', '访客', '体验', '。', 'Nuwa', '平台', '创建', '的', '数字', '人', '具备', '情商', '脑', '和', '智商', '脑', 
        '协同工作', '能力', '，', '突破', '传统', '局限', '，', '提供', '情绪', '和', '实用价值', '。', '应用', '包括', '网页', '嵌入', '、', '终端设备', 
        '和', '定制', '终端', '如', '柜机', '大屏', '，', '支持', '声音', '处理', '。', '管理', '功能', '涵盖', '流量', '监控', '、', '对话', '提取', '、', 
        '用户', '画像', '分析', '和', '体验', '反馈', '。', '创建', '过程', '需', '角色', '设计', '、', '形象', '克隆', '和', '实时', '生成', '，', '步骤', 
        '包括', '性格', '、', '职位', '、', '技能', '选择', '，', '以及', '容貌', '和', '声音', '定制', '。', '平台', '集成', '企业', '知识', '问答', '、', 
        '数据库', '搜索', '、', 'Web', '检索', '和', '内容', '创作', '，', '支持', '文字', '、', '语音', '和', '视觉', '交互', '。', '数字', '人', '应用', 
        '广泛', '，', '从', '教育', '到', '医疗', '，', '从', '文化', '到', '制造业', '，', '都', '能', '快速', '构建', '专属', '数字', '员工', '或', '助手', '。']
 
        words.append(wl_current_time)
        logger.info(f"Tokenized content into {len(words)} words.:\n{words}")
        
        chunks = [
            {"choices": [{"delta": {"role": "assistant","content": ""}}]},
        ]
        
        for word in words:
            chunks.append({"choices": [{"delta": {"content": word}}]})
        
       # chunks.append({"choices": [{"finish_reason": "stop"}]})
        
        logger.info(f"Generating {len(chunks)} chunks of mock response")
        
        for chunk in chunks:
            await asyncio.sleep(0.001)
            #logger.info(f"Yielding mock chunk: {chunk}")
            yield chunk

    # 异步生成响应
    # 自动重试装饰器，可以自动重试，失败达到了次数后会调用 大模型客户端管理的切换到备份的大模型客户端函数,也就是 OAIModelPoolClientManager的 switch_to_backup()
    @auto_retry(**AUTO_RETRY_CONFIG)
    async def a_generate_rsp(
        self,
        messages: List[Union[Message, Dict]], # 实际处理都是dict，不是message
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any,
    ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info("OpenAiChatModelClient::a_generate_rsp --- begin!")
        #logger.info(f"get_llm_reply::get_llm_reply ---- kwargs:{kwargs}")

        # 动态获取最新的异步客户端
        #client, model_name = await self.client_manager.get_client()

        """生成响应，支持流式和非流式输出（异步版本）"""
        openAIkwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
            # "top_p":self.top_p,
            # "temperature":self.temperature,
            # "frequency_penalty":self.frequency_penalty,
            # "presence_penalty":self.presence_penalty,
            # "extra_body":self.extra_body,
            #"repetition_penalty":self.repetition_penalty
            #"logit_bias": self.logit_bias,
            #"presence_penalty": self.presence_penalty,
            #"frequency_penalty":self.frequency_penalty,
            #"stop":self.stop # 新增的参数用来防止 工具调用吐出<|im start|>
        }
        
        # 动态添加可选参数（仅当属性存在时）
        optional_params = {
            "top_p": "top_p",
            "temperature": "temperature",
            #"frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "extra_body": "extra_body",
            # "repetition_penalty": "repetition_penalty",
            # "logit_bias": "logit_bias",
            # "stop": "stop",
        }

        # 如果参数在 optional_params 中，且属性存在，添加到 openAIkwargs
        for key, attr_name in optional_params.items():
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                # 可选：过滤掉 None 值，避免传无效参数（根据 OpenAI API 要求）
                if value is not None:
                    openAIkwargs[key] = value

        # websearch_enable = kwargs.pop('websearch_enable',False) 
        # logger.info(f"OpenAiChatModelClient::a_generate_rsp: websearch_enable:{websearch_enable}")

        if stream:
            openAIkwargs["stream_options"] = {"include_usage": True}

        if tools:
            openAIkwargs["tools"] = tools

        # openAIkwargs.update(kwargs) # 将kwargs的键和值更新到 openAIkwargs 中去，如果有相同的键直接kwargs 覆盖掉，没有的就添加

        logger.info(f"OpenAiChatModelClient::a_generate_rsp: openAIkwargs:\n{openAIkwargs}")
        # 记录调用开始时间
        start_time = time.perf_counter()

        try:
            if USE_MOCK:
                # 使用模拟流式响应
                response = self._mock_stream_response(openAIkwargs)
            else:           
                #response = await client.chat.completions.create(**openAIkwargs)
                response = await self.async_client.chat.completions.create(**openAIkwargs)
            #response = self.client.chat.completions.create(**openAIkwargs)
            logger.info(f"OpenAiChatModelClient::a_generate_rsp: response: {response}")

            if stream:
                # 当 stream=True 时，返回异步生成器以支持流式处理
                return self.parse_stream_response_async(response, start_time=start_time)
                #logger.info(f"OpenAiChatModelClient::a_generate_rsp: parse_stream_response_async abc: {abc}")
                #return abc
            
            else:
                # 当 stream=False 时，返回完整的消息列表
                return self.parse_non_stream_response(response, start_time=start_time)
            
        except Exception as e:
            logger.info(f"Caught exception type: {type(e).__name__}, full exception: {repr(e)}")
            # 检查异常是否为可重试的类型,如果是重试类型则触发重试
            if isinstance(e, self.AUTO_RETRY_CONFIG['retryable_errors']):
                logger.info("Exception matched retryable_errors, raising for retry")
                raise  # 重新抛出异常，让装饰器处理重试
            else:
                # 如果其他错误，则直接返回错误
                logger.error(f"Error in a_generate_rsp: {str(e)}", exc_info=True)
                error_msg = f"[系统异常] 请求处理中断: {str(e)}"
                
                # 返回一个立即结束的异步生成器，包含错误信息
                async def error_generator():
                    yield [Message(role="assistant", content=error_msg)]
                
                return error_generator()

    # 异步解析流式响应
    async def parse_stream_response_async(self, chunks: Any, start_time: float) -> AsyncIterator[List[Message]]:
        """解析 OpenAI 流式响应的结果（异步版本）"""
        logger.info("OpenAiChatModelClient::parse_stream_response_async --- begin!")
        tool_calls = []
        first_packet_logged = False
        # first_think_packet = False  # 第一个think包
        # first_content_packet = False # 第一个content包
        #full_content = ''  # 存放 给用户回复的内容
        think_content = '' # 存放 think 内容
        is_think_content = True # 是否是think内容
        think_start_stripped = False   # 标记是否已处理过 <think> 开始标签

        try:
            # 使用异步迭代器遍历流式响应
            async for chunk in chunks:
                try:
                    if not USE_MOCK:
                        chunk = chunk.model_dump() # 这个是openAI的接口
                    else:
                        chunk = chunk
                    #logger.info(f"OpenAiChatModelClient rsp:==>{chunk}")
                    
                    # 记录第一个有效报文的时间
                    if not first_packet_logged:
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        logger.info(f"OpenAiChatModelClient::parse_stream_response_async =========>Time to first packet: {elapsed_time:.3f} seconds")
                        first_packet_logged = True

                    # 检查 choices 是否为空
                    if not chunk['choices']:  # 如果 choices 是空的，跳过或处理
                        continue
                    #logger.info(f"OpenAiChatModelClient::parse_stream_response_async ---ddd----chunk: {chunk} ")
                    delta = chunk['choices'][0]['delta']
                    # 如果reason有内容或者 content有内容
                    content_text = delta.get('content',None)
                    reasoning_text = '' # 存放 思考内容 内容
                    if self.model == 'nuwa':# 这个是qwen3.5 397b的模型
                        reasoning_text = delta.get('reasoning',None)  # 思考内容是放在reasoning 字段里面的

                    #==================思考内容 和 content 内容 分开===========
                    elif self.model == 'minimax' and is_think_content:
                        reasoning_text = content_text # 刚开始的时候，content 内容为think内容
                        # ---------- 1. 剥离 <think> 开始标签 ----------
                        if not think_start_stripped:
                            if '<think>' in reasoning_text:
                                reasoning_text = reasoning_text.split('<think>', 1)[1]   # 取标签后的部分
                                think_start_stripped = True
                            else:
                                # 第一个 chunk 就没有 <think>，视为异常，直接当作思考内容处理
                                think_start_stripped = True

                        content_text = None # think内容结束，content设置为 None
                        if "</think>" in reasoning_text:
                            is_think_content = False # think内容结束
                            # 按结束标签拆分：前面是思考，后面可能是正式回复的开头
                            parts = reasoning_text.split('</think>', 1)
                            think_part = parts[0].strip()
                            after_end = parts[1].lstrip() if len(parts) > 1 else ''
                            reasoning_text = None
                            content_text = after_end # content 为think标签后的内容
                    #========================================================
                    else:
                        reasoning_text = delta.get('reasoning_content',None)

                    if content_text is not None or reasoning_text is not None:
                        #============================================================================================
                        #  20250825 换成oss模型后，没有think标签。think部分放到了reasoning_content。
                        #  使用oss模型的话，think部分内容是放到 reasoning_content 里面的，并且这个里面没有 think标签内容
                        #============================================================================================
                        if content_text == '' and reasoning_text is None:# 如果 reasoning_content 为 none， content 为空字符串，跳过 
                            continue # content 为空字符串，直接跳过

                        if content_text is None and reasoning_text =='':# 如果 content 为 none， reasoning_text 为空字符串，跳过 
                            continue # content 为空字符串，直接跳过

                        reasoning_content = None # 如果 think 有字段，并且有内容
                        if reasoning_text is not None and reasoning_text:
                            #logger.info(f"OpenAiChatModelClient::parse_stream_response_async ---reasoning_content: {reasoning_text} ")
                            reasoning_content = reasoning_text  # oss模型的时候，响应reason部分会放到 reasoning_content 里面
                            content = None # 如果有 reasoning  内容则 content设置为 None
                            yield [Message(role="assistant", content=[ThinkingBlock(type="thinking", thinking=reasoning_content)])]
                        
                        if content_text is not None and content_text:
                            reasoning_content = None # 如果 有content，设置think 为None
                            content = content_text 
                            yield [Message(role="assistant", content=[TextBlock(type="text", text=content)])]
 
                        #logger.info(f"OpenAiChatModelClient::parse_stream_response_async ---yield----content: {content} ")
                        # 如果是 openAI的 reasoning_content 就直接放到 extra里面发送
                        

                    # 下面处理是 tool call部分的内容，是一次性收完再yield出去的
                    elif delta.get('tool_calls'):
                        tool_call_data = delta['tool_calls'][0]
                        index = tool_call_data.get('index', 0)

                        while len(tool_calls) <= index:
                            tool_calls.append(tool_call_data)

                        if tool_call_data['function']['arguments'] == "" or tool_call_data['function']['arguments'] == None:
                            continue
                        else:
                            if tool_calls[index]['function']['arguments'] is None:
                                # 第一个消息是空的，直接赋值覆盖原有的None
                                tool_calls[index]['function']['arguments'] = tool_call_data['function']['arguments']
                            else:
                                tool_calls[index]['function']['arguments'] += tool_call_data['function']['arguments']

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing stream response: {e}, rsp: {chunk}")
                    continue

            if tool_calls:
                # 如果 role 是  FUNCTION 表示是 tool call
                logger.info(f"OpenAiChatModelClient::parse_stream_response_async ---fff----tool_calls: {tool_calls} ")
                # 注意这里的  tool_calls 返回的是整个所有被调用的 tool 集合，不是只有一个tool call调用的
                content = []
                for tool_call in tool_calls:
                    tool_use_block = ToolUseBlock(type="tool_use", id=tool_call['id'], name=tool_call['function']['name'], input=tool_call['function']['arguments'])
                    content.append(tool_use_block)

                # 如果是tool call 返回tool call list
                yield [Message(role="function", content=content, tool_calls=tool_calls)] # 如果是tool调用，这么消息出去就是 function 消息

            logger.info("OpenAiChatModelClient::parse_stream_response_async --- end!")

        except openai.APIConnectionError as e:
            logger.error(f"Connection dropped: {e}")
            yield [Message(role="assistant", content="[ERROR] 网络中断，请检查连接后重试")]
        except openai.APIError as e:
            logger.error(f"OpenAI server error: {e}")
            yield [Message(role="assistant", content="[ERROR] 服务暂时不可用")]
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            yield [Message(role="assistant", content="[ERROR] 系统异常")]
        
        #logger.info("OpenAiChatModelClient::parse_stream_response_async --- end!")

    # 根据用户输入的messages 和 Functions 进行处理了
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            tools: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """生成响应，支持流式和非流式输出
        
        Args:
            messages: 输入消息列表
            functions: 可选的功能列表（tools）
            stream: 是否使用流式输出
            delta_stream: 流式输出时是否仅返回增量内容
            **kwargs: 其他参数
        
        Returns:
            Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]: 响应内容
        """
        # 整理 OpenAI 参数
        openAIkwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
        }

        # 其他参数
        if stream:
            openAIkwargs["stream_options"] = {"include_usage": True}

        # 如果有 Function 
        if tools:
            openAIkwargs["tools"] = tools

        # 其他参数从 kwargs 中更新
        openAIkwargs.update(kwargs)

        #-------------------------------------------------------------
        #   调用 OpenAI 接口
        #-------------------------------------------------------------
        logger.info(f"OpenAiChatWarpperClient::genrate_rsp:-------cccc---- openAIkwargs:\n{openAIkwargs}")
        #response = self.client.chat.completions.create(**openAIkwargs)
        response = self.async_client.chat.completions.create(**openAIkwargs)
        logger.info(f"OpenAiChatWarpperClient::genrate_rsp:-------cccc---- response:\n{response}")

        if stream:
            # 返回流式 迭代器
            abc = self.parse_stream_response(response)
            logger.info(f"OpenAiChatWarpperClient::genrate_rsp:-------parse_stream_response -- response:\n{response}")
            return abc
        else:
            # 返回 Message 消息
            return self.parse_non_stream_response(response)

    # 解析模型的响应
    def parse_stream_response(self, chunks: Any) -> Iterator[List[Message]]:
        logger.info("OpenAiChatModelClient::parse_stream_response --- begin!")
        """解析 OpenAI 非流式响应的结果"""
        tool_calls = []  # 存储多个工具调用

        # 遍历每一个chunk 
        for chunk in chunks:
            try:
                #转成dict
                chunk = chunk.model_dump()
                delta = chunk['choices'][0]['delta']
                if delta.get('content') is not None:
                    content = delta['content']
                    # 有用户内容，将内容返回给用户
                    yield [Message(role="assistant", content=content)]
                        
                # 收集工具调用信息
                elif delta.get('tool_calls'):
                    tool_call_data = delta['tool_calls'][0]
                    index = tool_call_data.get('index', 0)

                    # 扩展 tool_calls 列表以匹配 index
                    while len(tool_calls) <= index:
                        tool_calls.append(tool_call_data)

                    # 更新工具调用字段
                    if tool_call_data['function']['arguments'] == "":
                        continue
                    else:
                        tool_calls[index]['function']['arguments']  += tool_call_data['function']['arguments']

            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing stream response: {e}, rsp: {chunk}")
                continue
        
        # 流结束后返回工具调用
        if tool_calls:
            yield [Message(role="tool", content='', tool_calls=tool_calls)]

    def parse_non_stream_response(self, chunk: Any, start_time: float) -> List[Message]:
        logger.info("OpenAiChatModelClient::parse_non_stream_response --- begin!")
        """解析 OpenAI 非流式响应的结果"""
        try:
            #转成dict
            chunk = chunk.model_dump()

            # 获取消息体
            message = chunk['choices'][0]['message']

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info(f"OpenAiChatModelClient::parse_non_stream_response =========>Time to first packet: {elapsed_time:.3f} seconds")

            if message.get('role') == 'assistant' and message.get('content'):
                return [Message(role="assistant", content=message['content'])]
            # 如果是tool call
            elif message.get('tool_calls'):
                # 如果 role 是  FUNCTION 表示是 tool call
                return [Message(role="tool", content='', tool_calls = message['tool_calls'])]
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing non-stream response: {e}, rsp: {chunk}")
            return [Message(role="assistant", content="Error processing response")]

    # 将 NuwaBaseTool 转换为 OpenAI 支持的工具格式
    def convert_nuwa_tool_to_model_support(self, nuwa_tools: Union[Dict, List[BaseTool]]) -> List[Dict]:
        """将 NuwaBaseTool 格式转换为 OpenAI 支持的工具格式"""
        '''
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia"
                            }
                        },
                        "required": [
                            "location"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },{.....},{...}]
        '''
        openai_tools =[]
        # 没有tool 返回空
        if not nuwa_tools:
            return openai_tools
        
        # 如果dict说明是openAI格式，直接返回
        if isinstance(nuwa_tools, dict):
            nuwa_tools = [nuwa_tools]

        # 遍历传入的 nuwa_tools
        for tool in nuwa_tools:
            tool_json = tool.openAI_format()
            # 包装成 OpenAI 标准格式
            openai_tool = {
                "type": "function",
                "function": tool_json  # 直接使用 tool.openAI_format() 的返回值作为 function 内容
            }
            openai_tools.append(openai_tool)

        return openai_tools
        
