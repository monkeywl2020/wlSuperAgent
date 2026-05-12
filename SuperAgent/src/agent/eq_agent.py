# -*- coding: utf-8 -*-
""" Base class for Agent """
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Union,AsyncIterator,Set
import re
import os
import json
from datetime import datetime
import httpx
import openai

from src.msg import Msg as Message
from src.agent.core.base_agent import BaseAgent,register_agent
from src.utils import semaphore_gather
from src.settings import settings,AgentConfig,PROJECT_ROOT_ABSOLUTE_PATH
from src.llm import get_chat_model_client
from src.sandbox.sandbox import Sandbox
from src.memory.file_memory import FileMemory
import src.tools.agent_tools  # 这个导入会注册tool内容
import traceback

from loguru import logger

'''
情商模型，专门负责情商问答，纯问答模型，基本不含tool call
'''

# BaseAgent
@register_agent('chat_agent')
class ChatAgent(BaseAgent):

    #初始化
    def __init__(
        self,
        agent_config: AgentConfig,
        sandbox: Sandbox,
        **kwargs: Any,
    ):
        logger.info("BaseAgent-----------__init__:{}",agent_config.agent_name)
        super().__init__(agent_config=agent_config, sandbox=sandbox, **kwargs)
        # 添加其他的初始化
        # 初始化的时候读取配置文件，从配置中读取system prompt template 的参数内容。
        #self.sys_prompt_param = self.load_system_prompt_param()
        # 初始化 FileMemory 实例
        self.memory = FileMemory()
        logger.info("FileMemory initialized successfully")
    
    # 从message中提取content
    def _get_content(self, msg: Union[Message, Dict]) -> str:
        """安全提取消息的 content 字段，兼容 Message 对象和 dict"""
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        
        # 如果 content 是 list（如多模态输入），尝试拼接为字符串
        if isinstance(content, list):
            # 假设每个 item 有 text 字段，或可转为字符串
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    texts.append(getattr(item, "text", ""))
                else:
                    texts.append(str(item))
            content = "".join(texts)
        elif not isinstance(content, str):
            content = str(content)
        
        return content

    # 这个通常情况都是str，不是list也就是会自己返回后的str
    def _get_role(self, msg: Union[Message, Dict]) -> str:
        if isinstance(msg, dict):
            return msg.get("role", "")
        else:
            return getattr(msg, "role", "")

    def get_filter_history_str_with_count(self, history_msg: List[Union[Message, Dict]]) -> tuple[str, int]:
        """
        将历史消息列表转换为字符串格式，用于模型输入。
        """
        # 如果历史消息为空，直接返回空字符串
        if not history_msg:
            return "", 0

        rounds = []  # 存储每轮的字符串
        i = 0
        round_num = 1

        while i < len(history_msg):
            msg = history_msg[i]

            # 跳过开头或中间的 assistant（防御性处理）
            role = self._get_role(msg)
            if role != "user":
                i += 1
                continue

            # 开启新一轮：以 user 消息开始
            user_content = self._get_content(msg)

            # 尝试获取紧随其后的 assistant 消息
            assistant_content = None
            if i + 1 < len(history_msg):
                next_msg = history_msg[i + 1]
                next_role = self._get_role(next_msg)
                if next_role == "assistant":
                    assistant_content = self._get_content(next_msg)
                    i += 2  # 消费两条消息
                else:
                    # 下一条不是 assistant，本轮只有 user
                    i += 1
            else:
                # 已到末尾，本轮只有 user
                i += 1

            # 构建本轮字符串（不含时间戳，如需可扩展）
            if assistant_content is not None:
                round_str = f"\n{round_num}：user: {user_content}\nassistant: {assistant_content}"
            else:
                round_str = f"{round_num}：user: {user_content}"

            rounds.append(round_str)
            round_num += 1

        abc = "\n".join(rounds)
        dialogue_count = len(rounds)
        logger.info(f"EmotionalChatAgent::get_filter_history_str --- \ndialogue_count:{dialogue_count}, abc:\n{abc}")
        return abc, dialogue_count

    # 按轮次过滤，输入history_messges 列表，和 要排除的轮次编号列表
    def filter_history_by_excluded_rounds(
        self,
        history_msg: List[Union[Message, Dict]],
        excluded_rounds: List[int]
    ) -> List[Union[Message, Dict]]:
        """
        根据大模型返回的要排除的轮次编号，过滤掉对应的历史对话轮次。

        Args:
            history_msg: 历史消息列表（不含当前用户最新输入）
            excluded_rounds: 要排除的轮次编号列表，如 [1, 2, 8]

        Returns:
            过滤后的新历史消息列表（保持原始顺序）
        """
        if not history_msg:
            return []

        excluded_set: Set[int] = set(excluded_rounds)
        # 记录每个轮次包含的原始消息索引
        round_to_indices = []  # list of list of indices, e.g., [[0,1], [2], [3,4]]
        
        i = 0
        while i < len(history_msg):
            msg = history_msg[i]
            role = self._get_role(msg)
            
            # 跳过非 user 消息（防御性处理）
            if role != "user":
                i += 1
                continue

            # 开启一个新轮次
            current_round_indices = [i]  # 至少包含 user 消息

            # 检查下一条是否是 assistant
            if i + 1 < len(history_msg):
                next_msg = history_msg[i + 1]
                next_role = self._get_role(next_msg)
                if next_role == "assistant":
                    current_round_indices.append(i + 1)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

            round_to_indices.append(current_round_indices)

        # 收集所有要保留的消息索引
        keep_indices = set()
        for round_num, indices in enumerate(round_to_indices, start=1):
            if round_num not in excluded_set:
                keep_indices.update(indices)

        # 按原始顺序构建新列表
        new_history_msg = [
            history_msg[i] for i in range(len(history_msg))
            if i in keep_indices
        ]

        logger.info(f"EmotionalChatAgent::filter_history_by_excluded_rounds --- \nnew_history_msg:{new_history_msg}")
        return new_history_msg

    # 解析模型响应，获取要排除的轮次
    async def parse_excluded_rounds(self, rsp: str) -> List[int]:
        """
        从模型响应中提取要排除的轮次编号列表。

        Args:
            rsp: 模型的原始响应字符串，例如 "[1,2,8]"

        Returns:
            解析后的轮次编号列表，如 [1, 2, 8]
        """
        # 移除可能的首尾空格和方括号
        rsp = rsp.strip().strip('[]')
        # 使用正则表达式提取所有数字
        round_nums = re.findall(r'\d+', rsp)
        # 转换为整数并返回
        return [int(num) for num in round_nums]
        
    # 动态注入角色提示词，根据用户提供的参数，构建角色的提示词内容
    def constructe_role_system_prompt(self,  metadata_data: Dict)-> str:
	    #-----------------------------------
        # 1. 获取 metadata_data 所有参数
        #-----------------------------------
        language = metadata_data.get('language', '中文')

        #logger.info(f"EmotionalChatAgent::constructe_role_system_prompt ---- metadata_data:{metadata_data}")
        #-----------------------------------
        # 2. 将所有的参数填充到 sys_prompt 中
        #-----------------------------------
        if '中文' in language:
            final_system_prompt = self.sys_prompt_cn
        else:
            final_system_prompt = self.sys_prompt_en

        #logger.debug(f"EmotionalChatAgent::constructe_role_system_prompt ---- final_system_prompt:{final_system_prompt}")
        return final_system_prompt

    #----------------------------------------------------
    # 构造 skill select 这个工具的提示词
    #----------------------------------------------------
    async def constructe_skill_select_tool_system_prompt(self, avatar_data: Dict, skill_select_tool_name ='skill_select') -> str:
        # 接着这段内容下面的，所以加2个回车换行 ## **历史记录**：{history}

        function_switch = avatar_data.get('function_switch', {})
        web_search_enabled = function_switch.get('web_search', False)
        ppt_play_enabled = function_switch.get('ppt_play', False)
        talent_play_enabled = function_switch.get('talent_play', False)
        generate_ppt_enabled = function_switch.get('generate_ppt', False)
        add_knowledge_enabled = function_switch.get('add_knowledge', False)

        # 从 function_map 获取 skill_select 的实例
        skill_select_instance = self.function_map.get(skill_select_tool_name,None)
        if isinstance(skill_select_instance,skill_select):
            skills_info_list = await skill_select_instance.get_available_skills(
                                                    web_search = web_search_enabled,
                                                    ppt_play_switch = ppt_play_enabled,
                                                    talent_play = talent_play_enabled,
                                                    generate_ppt = generate_ppt_enabled,
                                                    add_knowledge = add_knowledge_enabled
                                                    )
        else:
            logger.info(f"EmotionalChatAgent::constructe_skill_select_tool_system_prompt ---- skill_system_prompt is null!")
            return '' # 没有工具
        
        # XML 特殊字符转义函数
        def escape_xml(text):
            if not isinstance(text, str):
                return str(text)
            return (text.replace("<", "“")
                        .replace(">", "”"))

        # 构建 XML 内容
        xml_lines = ['<available_skills>']

        # 遍历 skill ，按照开关组成对应的提示词        
        for skill in skills_info_list:
            # 添加技能块（带缩进）
            xml_lines.append('  <skill>')
            xml_lines.append(f'    <skill_type>{escape_xml(skill["router_token"])}</skill_type>')
            xml_lines.append(f'    <description>{escape_xml(skill["description"])}</description>')
            #xml_lines.append(f'    <name>{escape_xml(skill["name"])}</name>')
            xml_lines.append('  </skill>')
        
        xml_lines.append('</available_skills>')
        final_skill_content = "\n".join(xml_lines)
        intent_routing_perfix = settings.chat_template.cn.intent_routing_perfix
        intent_routing_subfix = settings.chat_template.cn.intent_routing_subfix

        skill_system_prompt = intent_routing_perfix + '\n\n' + final_skill_content + '\n' + intent_routing_subfix

        #logger.info(f"EmotionalChatAgent::constructe_skill_select_tool_system_prompt ---- skill_system_prompt:【{skill_system_prompt}】")
        return skill_system_prompt

    #----------------------------------------------------
    # 构造 tool 提示词，这个是二次选择后的提示词
    #----------------------------------------------------
    async def constructe_select_tool_system_prompt(self, skill_name:str, avatar_data: Dict, skill_select_tool_name ='skill_select') -> str:

        # 从 function_map 获取 skill_select 的实例
        skill_detail_info = ''
        skill_select_instance = self.function_map.get(skill_select_tool_name,None)
        if isinstance(skill_select_instance,skill_select):
            skill_detail_info = await skill_select_instance.get_skill_by_name(skill_name)

        final_skill_detail_info = skill_detail_info
        # 处理有文件列表的技能
        if skill_name == 'singing_performance':
            final_skill_detail_info = skill_detail_info.format(song_file_list = avatar_data.get('song_file_list', []))
        elif skill_name == 'dancing_performance':
            final_skill_detail_info = skill_detail_info.format(dancing_file_list = avatar_data.get('dancing_file_list', []))
        elif skill_name == 'poetry_recitation':
            final_skill_detail_info = skill_detail_info.format(poetry_file_list = avatar_data.get('poetry_file_list', []))
        elif skill_name == 'drama_performance':
            final_skill_detail_info = skill_detail_info.format(drama_file_list = avatar_data.get('drama_file_list', []))
        elif skill_name == 'speech_performance':
            final_skill_detail_info = skill_detail_info.format(speech_file_list = avatar_data.get('speech_file_list', []))
        elif skill_name == 'ppt_controller':
            final_skill_detail_info = skill_detail_info.format(
                ppt_file_list = avatar_data.get('ppt_file_list', []),
                ppt_status = avatar_data.get('ppt_status', False),
                video_status = avatar_data.get('video_status', False),
            )
        else:
            pass
        
        select_tool_perfix = settings.chat_template.cn.select_tool_perfix
        select_tool_subfix = settings.chat_template.cn.select_tool_subfix
        skill_detail_info_rst = select_tool_perfix + '\n\n' + final_skill_detail_info + '\n' + select_tool_subfix
        logger.info(f"EmotionalChatAgent::constructe_select_tool_system_prompt ---- skill_detail_info:【{skill_detail_info_rst}】")
        return skill_detail_info_rst

    #----------------------------------------------------
    # 构造 skill select 的执行协议 sys_excution_protocol 提示词
    #----------------------------------------------------
    def constructe_skill_select_execution_protocol_prompt(self, avatar_data: Dict) -> str:
        #----------------------------
        # 设置 Execution_Protocol 其中的 core_instructions核心指令 会因为 talent_play 开关 而变化里面的 core_instructions
        #----------------------------
        core_instructions = settings.chat_template.cn.core_instructions_skill_select # 使用 技能选择的 核心指令

        default_language = avatar_data.get('default_language', '中文')
        sys_excution_protocol = '\n\n'+ settings.chat_template.cn.sys_excution_protocol.format(
            core_instructions = core_instructions, #核心指令
            default_language = default_language
        )

        return sys_excution_protocol

    #----------------------------------------------------
    # 构造 select_tool 的执行协议 sys_excution_protocol 提示词
    #----------------------------------------------------
    def constructe_select_tool_execution_protocol_prompt(self, avatar_data: Dict) -> str:
        #----------------------------
        # 设置 Execution_Protocol 其中的 core_instructions核心指令 会因为 talent_play 开关 而变化里面的 core_instructions
        #----------------------------
        core_instructions = settings.chat_template.cn.core_instructions_select_tool # 使用 选择工具的 核心指令

        default_language = avatar_data.get('default_language', '中文')
        sys_excution_protocol = '\n\n'+ settings.chat_template.cn.sys_excution_protocol.format(
            core_instructions = core_instructions, #核心指令
            default_language = default_language
        )

        return sys_excution_protocol

    # 将历史记录转换成 Markdown 格式
    def format_history_as_markdown(self, history: List[Dict[str, str]], language:str) -> str:
        # 接着这段内容下面的，所以加2个回车换行 ## **历史记录**：{history}
        markdown = ''
        if "中文" in language:
            markdown = settings.chat_template.cn.history_perfix # 替换成配置文件中的配置
        else:
            markdown = settings.chat_template.en.history_perfix # 替换成配置文件中的配置
        
        #=======================================
        # 将历史记录过滤掉 Assistant部分的，只保留最近3轮模型的回复，其他的都是只有user部分的内容
        #=======================================
        # 找出所有 user 消息的索引
        user_indices = [i for i, msg in enumerate(history) if msg['role'] == 'user']
        
        # 取最后 3 个 user 的索引（如果有的话）
        last_three_user_indices = set(user_indices[-5:]) if user_indices else set()

        for i, message in enumerate(history):
            role = message['role']
            content = message['content']

            # 总是显示 user 消息
            if role == 'user':
                markdown += f"\n{role}: {content}\n"
            # assistant 消息仅在它紧跟的前一个 user 属于“最后3个 user”时才显示
            elif role == 'assistant':
                # 检查前一条是否是 user 且属于最后三个
                if i > 0 and history[i - 1]['role'] == 'user' and (i - 1) in last_three_user_indices:
                    markdown += f"\n{role}: {content}\n"
            # 其他角色（如 system）可选择忽略或处理，这里暂不处理

        # 结尾添加上标签的闭合
        markdown += settings.chat_template.cn.history_subfix

        if 0: # 下面这个是原始的代码
            for message in history:
                role = message['role']
                content = message['content']
                #time = message['created_at'] # 给每条消息添加一个 时间记录
                #markdown += f"\n{time} | {role}: {content}\n"
                markdown += f"\n{role}: {content}\n"
                
        return markdown

    #----------------------------------------------------
    # 搜索用户的画像信息和相关的记忆信息
    #----------------------------------------------------
    def constructe_system_prompt(self, 
        new_role_system_prompt, # 角色提示词部分内容(含安全防角色越狱部分的描述)
        skill_select_prompt, # 技能选择的提示词
        md_prompt # md文件内容的提示词
    ):

        # logger.debug(f"EmotionalChatAgent::constructe_system_prompt ---- new_role_system_prompt:{new_role_system_prompt}")
        # logger.debug(f"EmotionalChatAgent::constructe_system_prompt ---- skill_select_prompt:{skill_select_prompt}")
        # logger.debug(f"EmotionalChatAgent::constructe_system_prompt ---- md_prompt:{md_prompt}")
        final_system_prompt = ''

        # 技能选择的提示词 后续再添加
        # # skill 的提示词。需要根据 skill的内容来填写提示词
        # skill_context_prompt = settings.chat_template.cn.skill_context_perfix
        # # skill_select 提示词
        # skill_select_prompt_tmp = settings.chat_template.cn.skill_select_perfix
        # # 构造skill提示词
        # skill_select_prompt = self.constructe_skill_select_tool_system_prompt(skill_select_prompt_tmp)
        # skill_select_prompt = skill_select_prompt_tmp.format(skill_select_prompt)
        
        # workspace提示词
        workspace_prompt = settings.chat_template.cn.workspace_perfix

        # sandbox 提示词
        sandbox_prompt =  settings.chat_template.cn.sandbox_perfix

        #时区 提示词
        timezone_prompt_tmp = settings.chat_template.cn.timezone_perfix
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timezone_prompt = timezone_prompt_tmp.format(current_time = current_time)

        # 项目上下文提示词
        project_context_prompt =  settings.chat_template.cn.project_context_perfix

        # 组合提示词：1 系统提示词 2 知识库提示词 3 websearch rst提示词（没开websearch开关这个为空） 4 历史记录提示词
        final_system_prompt = (new_role_system_prompt 
            + workspace_prompt
            + sandbox_prompt # 知识库
            + timezone_prompt
            + md_prompt
            + project_context_prompt 
        )

        # 组合所有部分
        logger.debug(f"EmotionalChatAgent::constructe_system_prompt ---- final_system_prompt:{final_system_prompt}")
        return final_system_prompt

    def constructe_user_prompt(self, 
        memory_text, # 历史记录
        knowledge_rag, # 知识库动态RAG查询回来的内容
        user_input # 角色提示词部分内容(含安全防角色越狱部分的描述)
    ):
        final_user_prompt = ''
        user_input_prompt = "\n用户输入:"+ user_input.strip()

        # 组合提示词：1 系统提示词 2 知识库提示词 3 websearch rst提示词（没开websearch开关这个为空） 4 历史记录提示词
        final_user_prompt = (memory_text 
            + knowledge_rag
            + user_input_prompt # 知识库
        )

        # 组合所有部分
        logger.debug(f"EmotionalChatAgent::constructe_user_prompt ---- final_user_prompt:{final_user_prompt}")
        return final_user_prompt

    # 根据skill name 获取 这个skill的 tool
    def get_tool_by_skill_name(self, skill_name:str):

        skill_tools = [
            tool for name, tool in self.function_map.items()
            if name == skill_name
        ]

        if len(skill_tools) == 0:
            logger.info(f"EmotionalChatAgent::get_tool_by_skill_name ---- skill_name:{skill_name} is not found!")
            return []

        logger.info(f"EmotionalChatAgent::get_tool_by_skill_name ---- skill_name:{skill_name} is found! skill_tools:{skill_tools}")
        return skill_tools

    def a_getRagknowledge(self, user_input: str) -> str:
        return ''

    async def search_memory_info(self, user_input:str) -> str:
        logger.info(f"ChatAgent::search_memory_info  begin!---- user_input:{user_input}")
        return ''

    async def task_parallel_process(self, user_input: str, max_coroutines: int = 2) -> List[Any]:
        tasks = [] # 并发的任务
        task_names = [] # 任务的名称

        #-----------------------------------------------------------
        # 任务1)： RAG 知识库 
        #-----------------------------------------------------------
        knowledge_content = '' 
        #  简单问候不查询，直接过滤掉不查知识库
        if self.is_simple_greeting(user_input):
            logger.warning(f"EmotionalChatAgent =========> Simple greeting detected, skipping knowledge base query")
            knowledge_content = '\n\n' # 不需要查知识库 ，直接返回空
        else:
            tasks.append(
               self.a_getRagknowledge(user_input)
            )
            task_names.append("knowledge")

        #-----------------------------------------------------------
		# 任务2)：个性化记忆检索 记忆相关参数：获取画像用户id，userid是agent对应的用户，可能会有多个人跟这个 agent进行交流。这个portrait_id就是前端识别出来的具体的某个用户
        #-----------------------------------------------------------
        tasks.append(
                self.search_memory_info(user_input)
        )
        task_names.append("memory")
    
        #----------------------------------------------
        #      ***** 2个任务并发执行 *****
        #----------------------------------------------
        if tasks:
            wlstart = time.perf_counter()

            results = await semaphore_gather(*tasks, max_coroutines=2)
            for name, result in zip(task_names, results):
                if name == "knowledge":
                    knowledge_content = result or ''
                elif name == "memory":
                    portrait_text = result or ''

            wlend = time.perf_counter()
            logger.info(f"EmotionalChatAgent::a_generate_rsp  ---- rag-search and memory-search cost time is:{wlend - wlstart}s!")
            return knowledge_content, portrait_text
        else:
            return '', ''

    # 处理用户输入的消息，提取 user_id 和 session_id，整理为标准格式。
    def prepare_user_message(self, messages: List[Dict]):
        if not messages:
            raise ValueError("消息列表为空")

        # 假设所有消息都有相同的 user_id 和 session_id，取第一条消息提取
        first_msg = messages[0]
        #------------------------
        metadata = first_msg.get("metadata", {})
        session_id = metadata.get("session_id","")
        #------------------------
        if not session_id:
            raise ValueError("消息必须包含 session_id")

        # 整理消息，仅保留 role 和 content
        prepared_messages = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
            if msg.get("role") == "user"  # 只保留 role 为 user 的消息
        ]
        return prepared_messages, metadata

    #================================================
    # 从 workspace 中提取 *.md 格式的系统提示词。
    # 1. 从 workspace 中读取 *.md 文件
    # 2. 提取 *.md 文件中的系统提示词
    # 3. 按照指定顺序组合md文件内容
    # 4. 返回md提示词
    #================================================
    async def get_md_prompt(
        self,
        sandbox: Sandbox,                      # sandbox 类实例，用于调用 read_file 方法
        relative_dir: str = 'md/cn',           # 相对于 HOST_WORKSPACE 的目录路径，如 "docs"
        file_order: Optional[List[str]] = ["AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md","USER.md", "HEARTBEAT.md", "BOOTSTRAP.md"] # 文件读取顺序，如 ["AGENT.md","SOUL.md","TOOLS.md",...]
    ) -> str:
        # 遍历md文件，提取 *.md 格式的系统提示词
        # ============ 步骤 1: 构建目录的绝对路径 ============
        # 将用户传入的相对路径拼接为宿主机的绝对路径，用于文件系统遍历
        md_dir_in_workspace = 'SuperAgent/example/workspace/md/cn'
        md_absolute_dir = os.path.join(PROJECT_ROOT_ABSOLUTE_PATH, md_dir_in_workspace)
        logger.info(f"ChatAgent::get_md_prompt  ---- md_absolute_dir:{md_absolute_dir}")

        # ============ 步骤 2: 校验目录有效性 ============
        # 如果路径不存在，记录警告并返回空字符串
        if not os.path.exists(md_absolute_dir):
            logger.error(f"⚠️  Directory does not exist: {md_absolute_dir}")
            return ""

        # 如果路径存在但不是目录（比如是文件），也返回空
        if not os.path.isdir(md_absolute_dir):
            logger.error(f"⚠️  Path is not a directory: {md_absolute_dir}")
            return ""

        # ============ 步骤 3: 初始化结果容器 ============
        # 使用列表暂存每个文件的"路径+内容"，最后统一拼接，效率高于字符串累加
        result_parts = []

        # ============ 步骤 4: 按照指定顺序读取文件 ============
        # 如果没有指定顺序，默认按文件名排序
        if file_order is None:
            file_names = sorted(os.listdir(md_absolute_dir))
        else:
            file_names = file_order

        # ============ 步骤 5: 遍历指定顺序的文件列表 ============
        for entry_name in file_names:
            logger.info(f"ChatAgent::get_md_prompt  ---- entry_name:{entry_name}")

            # ============ 步骤 5.1: 构建文件的绝对路径 ============
            # 拼接得到文件在宿主机上的完整路径
            file_abs_path = os.path.join(md_absolute_dir, entry_name)
            logger.info(f"ChatAgent::get_md_prompt  ---- file_abs_path:{file_abs_path}")

            # ============ 步骤 5.2: 跳过非文件项 ============
            # 只处理普通文件，跳过子目录、符号链接等
            if not os.path.isfile(file_abs_path):
                continue

            # ============ 步骤 5.3: 跳过非 .md 文件 ============
            file_ext = os.path.splitext(entry_name)[1].lower()
            if file_ext != '.md':
                continue

            # ============ 步骤 5.4: 构建相对路径（供 sandbox.read_file 使用） ============
            # sandbox 的 read_file 需要相对于 HOST_WORKSPACE 的路径
            logger.info(f"ChatAgent::get_md_prompt  ---- relative_dir:{relative_dir} entry_name:{entry_name}")
            file_relative_path = os.path.join(relative_dir, entry_name)
            logger.info(f"ChatAgent::get_md_prompt  ---- file_relative_path:{file_relative_path}")

            # ============ 步骤 5.5: 调用 sandbox 方法读取文件内容 ============
            try:
                # 通过 sandbox 实例读取文件，复用其编码处理和挂载逻辑
                content = await sandbox.read_file(file_relative_path)
                #logger.info(f"ChatAgent::get_md_prompt  ---- content:{content}")

                # ============ 步骤 5.6: 使用标签包裹文件内容 ============
                # 根据文件名生成标签名，如 AGENTS.md -> <agents>...</agents>
                tag_name = entry_name.replace('.md', '').lower()
                md_content = f"<{tag_name}>\n## 文件在workspace下路径为【{file_relative_path}】（更新此文件注意路径要正确）\n\n{content}\n\n</{tag_name}>\n"
                result_parts.append(md_content)

            except FileNotFoundError:
                # 文件不存在或已被删除，标记为 [MISSING]
                tag_name = entry_name.replace('.md', '').lower()
                missing_content = f"<{tag_name}>\n## 文件在workspace下路径为 {file_relative_path}\n\n[MISSING]\n\n</{tag_name}>\n"
                result_parts.append(missing_content)
                logger.warning(f"⚠️  file not found, marked as [MISSING]: {file_relative_path}")
            except UnicodeDecodeError:
                # 文件不是 utf-8 编码时跳过
                logger.error(f"⚠️  Encoding error, skipping:{file_relative_path}")
            except Exception as e:
                # 捕获其他异常，避免单个文件失败导致整个函数崩溃
                logger.error(f"❌  Read failed {file_relative_path}: {type(e).__name__} - {e}")

        # ============ 步骤 6: 拼接所有文件内容并返回 ============
        # 用分隔线连接不同文件，便于阅读区分（可根据需要调整或移除）
        separator = "\n"
        final_result = separator.join(result_parts)
        final_result = "\n"+ final_result  # 在开头增加一个换行

        # 可选：打印统计信息
        if result_parts:
            logger.info(f"✅ 已读取 {len(result_parts)} 个文件")

        #logger.info(f"ChatAgent::get_md_prompt  ---- final_result:{final_result}")
        return final_result

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来自用户的消息，并且给用户回响应
    #------------------------------------------------
    async def a_generate_rsp(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any
    ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info("ChatAgent::a_generate_rsp begin!")

        new_messages = [] # 存放新的消息
        #========================================================================
        #1：解析用户消息，提取 user_id 和 session_id，整理为标准格式。
        #========================================================================
        prepared_messages, metadata = self.prepare_user_message(messages)

        #========================================================================
        #2：获取用户的对话历史记录（根据 session_id 过滤）
        #========================================================================
        # 从 metadata 中提取 session_id
        session_id = metadata.get("session_id", "") if metadata else ""
        user_history = await self.memory.get_recent_rounds_async(session_id=session_id)
        logger.info(f"=====>>>> ChatAgent::a_generate_rsp ---- user_history:{user_history}")

        #========================================================================
        #3：将用户消息添加到记忆中
        #========================================================================
        msg = prepared_messages[-1]
        await self.memory.add_message_async(msg.get('role', 'role'), msg.get('content', ''), metadata=metadata)
        logger.info(f"ChatAgent::a_generate_rsp ---- Added user message to memory: {msg}!")

        #========================================================================
        #4：组合系统提示词。1 最开始的系统提示词。
        #========================================================================
        new_role_system_prompt = self.constructe_role_system_prompt(metadata)
        logger.info(f"ChatAgent::a_generate_rsp ---- new_role_system_prompt:{new_role_system_prompt}")

        #========================================================================
        #5：组合技能上下文提示词。---- 暂时不用
        #========================================================================
        #skill_select_prompt = settings.chat_template.cn.skill_context_prefix.format(read_tool="read_file")

        #========================================================================
        #6：添加 *.md 格式的系统提示词,将所有的 md文件内容作为提示词加入
        #========================================================================
        # md 文件读取顺序，按照指定顺序组合
        md_file_order = [
            "AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md",
            "USER.md", "HEARTBEAT.md", "BOOTSTRAP.md"
        ]
        md_prompt = await self.get_md_prompt(self.sandbox, file_order=md_file_order)
        
        #========================================================================
        #7：整合所有的系统提示词
        #========================================================================
        new_system_prompt = self.constructe_system_prompt(
            new_role_system_prompt, 
            "",  # skill_select_prompt
            md_prompt,  # history_text
        )

        sys_prompt_message = {"role":"system","content":new_system_prompt}

        #========================================================================
        # 8：搜索相关记忆
        #========================================================================
        user_input = prepared_messages[-1].get('content', '') if prepared_messages else ''
        memory_results = await self.memory.memory_search_async(user_input, 5)
        memory_text = """\n## 相关记忆\n"""
        if len(memory_results) > 0:
            for result in memory_results:  # 只取前5个最相关的记忆
                memory_text += f"- {result.get('content', '')}\n"
        else:
            memory_text = "" # 如果没有相关记忆，就空字符串

        #========================================================================
        # 9：知识库动态RAG查询回来的内容
        #========================================================================
        # knowledge_content = await self.knowledge_base.search(user_input)
        # knowledge_content = knowledge_content.replace("\n", " ")
        # knowledge_rag = settings.chat_template.cn.knowledge_perfix.format(knowledge_content=knowledge_content)
        knowledge_rag = "\n" # 这里面存放知识库动态RAG查询回来的内容 

        #========================================================================
        # 10：整合构建用户提示词
        #========================================================================
        user_prompt = self.constructe_user_prompt(memory_text, knowledge_rag, user_input)
        user_prompt_message = {"role":"user","content":user_prompt}
        ###################################################################################
        # 2：挂载tool工具
        # 构建 agent_tools：后续我准备将这个tool放到 模型回复里面去可以修改。也就是模型可以决定挂载哪些工具
        ###################################################################################
        agent_tools = [
            tool for name, tool in self.function_map.items()
            #if name not in exclude_tool_names
        ]
        #logger.info(f"ChatAgent::a_generate_rsp ---- agent_tools:{agent_tools}!")

        tools = self.llm_client.convert_nuwa_tool_to_model_support(agent_tools)
        #logger.info(f"ChatAgent::a_generate_rsp ---- new format tools:{tools}!")

        ###################################################################################
        #   3:            ***** 收集需要并发执行的任务 *******
        ###################################################################################
        #knowledge_content, portrait_text = await self.task_parallel_process(user_input)

        ###################################################################################
        # 4：重新整合提示词
        # 获取用户avatar参数。构建 role 角色部分的提示词
        ###################################################################################
        #new_role_system_prompt = self.constructe_role_system_prompt()

        #--------------------重新组装消息，openAI格式qwen支持不行-----------------------------
        new_messages = []

        #-----------------------------------------
        #   1：添加用户历史记录
        #   2：添加用户消息
        #   3：添加系统提示词（加到最前面）
        #-----------------------------------------
        new_messages.extend(user_history) # 添加用户历史记录
        new_messages.append(user_prompt_message) # 添加用户消息
        new_messages.insert(0,sys_prompt_message) # 添加系统提示词,加到最前面

        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        num_llm_calls_available = kwargs.pop('num_llm_calls', 50) # 最大执行轮次。
        logger.info(f"EmotionalChatAgent::a_generate_rsp  ---- num_llm_calls_available:{num_llm_calls_available}!")
        
        #logger.info(f"EmotionalChatAgent::a_generate_rsp  ---- new_messages:{new_messages}")

        # 返回生成器对象
        return self.get_llm_reply(
            messages=new_messages,
            tools=tools,
            num_llm_calls_available=num_llm_calls_available,
            stream=stream,
            sys_prompt_message = sys_prompt_message,
            metadata = metadata, # 将metadata 参数传下去
            **kwargs
        )

    #-----------------------------------------------------------------------------------------
    #                 ---** get_llm_reply 定制处理逻辑，重写了 BaseAgent 里面的Function call逻辑 **---
    # 这个是访问大模型，获取模型的响应接口。
    # 由于这个函数是 eq_agent 和 iq_agent定制的，所以需要重写 base_agent 的逻辑
    #   **为什么是定制的： 因为 这两个agent有个 切换agent的逻辑在里面，通过tool call来切换agent**
    #   所以 这个tool call调用不是通用逻辑，需要定制
    #-----------------------------------------------------------------------------------------
    async def get_llm_reply(
        self,
        messages: List[Union[Message, Dict]],
        tools: List[Dict],
        num_llm_calls_available: int,
        stream: bool = True,
        sys_prompt_message: Dict = {},
        metadata:Dict = {},
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        """异步生成器函数，统一处理流式和非流式响应"""
        logger.info("get_llm_reply::get_llm_reply ---- begin!")

        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        #org_sys_prompt = messages[0] # 保存一份原始的system prompt，因为在skill_select的tool调用的时候可能会替换掉这个system prompt
        # 并且在skill_select的tool调用结束后，可能需要再一次使用这个原始的system prompt来请求模型获取最终的回复。3次模型交互
        used_tools = tools

        while num_llm_calls_available > 0:  # 这个控制对话轮次
            num_llm_calls_available -= 1

            logger.info(f"get_llm_reply::get_llm_reply ---- num_llm_calls_available【{num_llm_calls_available}】")
            #logger.info(f"get_llm_reply::get_llm_reply ---- kwargs:{kwargs}")
            responses = await self.llm_client.a_generate_rsp(
                messages=messages,
                stream=stream,
                tools=used_tools,
                **kwargs
            )

            if isinstance(responses, AsyncIterator):
                collected_responses_message = []# 这个是 list[Message] ，是收集 llm 回来的 工具调用的role为 function 消息
                assistant_reply = ""  # 累积助手的回复内容
                # 遍历 迭代器中的每个消息
                #logger.info(f"get_llm_reply::get_llm_reply ---- a_generate_rsp --- AsyncIterator begin !")
                async for response in responses:
                    #logger.info(f"get_llm_reply::get_llm_reply ---ddd--- response:{response}!")
                    message: Message = response[0] 
                    # 如果是用户消息，直接yield出去
                    #if message.role == ASSISTANT and message.content:
                    if message.role == "assistant":
                        yield [message]  # 实时返回用户消息
                        # 累积助手的回复内容
                        text_blocks = message.get_content_blocks(block_type="text")
                        if text_blocks and len(text_blocks) > 0:
                            assistant_reply += text_blocks[0].get("text", "")
                    # 如果是 Function 消息，将所有的消息收完（模型客户端侧之后返回一个消息）
                    elif message.role == "function" and message.has_content_blocks("tool_use"):
                        collected_responses_message.extend(response)# 获取的是response 这个是llm返回的[Message] 是list[Message]
                    else:
                        # do nothing
                        pass
                
                # 将助手的回复添加到记忆中
                if assistant_reply:
                    assistant_message = Message(role='assistant',content=assistant_reply,metadata=metadata)
                    await self.memory.add_message_async(assistant_message)
                    logger.info(f"Added assistant reply to memory: {assistant_reply}...")

                # 这个是在迭代器遍历完成后。如果有 tool的话，这个不为空，标识有 tool调用
                if collected_responses_message:
                    logger.info(f"===============collected_responses_message:{collected_responses_message}")
                    # tool_call_msg：调用消息 
                    # tool_results： 调用的结果消息
                    has_tool_call, tool_call_msg, tool_results = await self._handle_tool_calls(collected_responses_message, 
                                                                                                sandbox=self.sandbox,
                                                                                                **kwargs)
                    logger.info(f"BaseAgent::get_llm_reply ---fff--- 【has_tool_call:{has_tool_call}】 【tool_call_msg:{tool_call_msg}】 【tool_results:{tool_results}】 !")
                    if has_tool_call:
                        # tool_call_msg 是dict，tool_results 是list[dict]。 combined_results 是list[dict]
                        combined_results = [tool_call_msg] + tool_results # 组合两个消息
                        #####################################################################
                        #   combined_results 里面是 function call 的信息和
                        #       调用了tool的信息，返回的是tool的结果  
                        #       这两个消息，一个是 {role:assistant，content:function call} 
                        #                  一个是 {role:tool，content:function call result}
                        ####################################################################
                        # 将 tool 调用消息记录到 memory（role 为 function）
                        tool_calls_in_msg = tool_call_msg.get("tool_calls",None)
                        toolcallMsg= Message(role="function",content="", tool_calls=tool_calls_in_msg, metadata=metadata)
                        await self.memory.add_message_async(toolcallMsg)
                        logger.info(f"Added function message to memory: {toolcallMsg} ...")
                        yield [toolcallMsg]
                        
                        # 将 tool 结果消息记录到 memory（role 为 tool）
                        for tool_result in tool_results:
                            result_content = tool_result.get("content",None)
                            tool_call_id = tool_result.get("id",None)
                            extra = tool_result.get("extra",None)
                            toolresultMsg = Message(role="tool", content=result_content, tool_call_id =tool_call_id, extra=extra, metadata=metadata)    
                            await self.memory.add_message_async(toolresultMsg)

                            logger.info(f"Added tool result to memory, extra:{extra}, result_content:{result_content[:50] if result_content else 'No content'}...")
                        
                        # 这个地方，要修改下发送给前端显示的内容，这里只需要显示tool调用结果就行了。 并且需要将格式修改为用户可读的内容。
                        #yield combined_results  # 返回工具调用结果   前端显示调用结果

                        # 如果还可以调用大模型
                        if num_llm_calls_available > 0:
                            messages.extend(combined_results)   # 将 tool_call_msg 消息加入messages

                            #used_tools = tools # 更新used_tools工具
                            continue
                
                # 结束流式迭代器       
                return

            # 下面是非流式的消息，就是普通的消息不是异步迭代器
            elif isinstance(responses, list) and responses:
                if responses[0].role == "assistent":
                    yield responses  # 返回非流式用户消息，转成 异步迭代器返回，外部统一处理

                elif responses[0].role == "tool":
                    has_tool_call, tool_call_msg, tool_results = await self._handle_tool_calls(responses, **kwargs)
                    if has_tool_call:
                        combined_results = [tool_call_msg] + tool_results

                        # 采用skill 模式后，原有的tool 被替换成skill模式的调用了。
                        # 增加技能选择工具调用结果处理
                        if (len(tool_results) == 1 
                            and tool_results[0].content == '' 
                            and isinstance(tool_results[0].extra, dict) 
                            and "skill_type" in tool_results[0].extra
                        ):
                            # 提取技能类型和新系统提示词
                            skill_data = tool_results[0].extra
                            skill_type = skill_data["skill_type"]
                            new_skill_system_prompt = skill_data["new_system_prompt"]
                            need_call_llm = skill_data["need_call_llm"]
                            logger.info(f"=== ---fff--- 【Skill selection tool detected, skill type:: {skill_type}】")
                            # 替换系统提示词
                            if new_skill_system_prompt:
                                # 定位并替换系统提示词，不能用新 的提示词，如果没有调用函数直接回复那就错误了，所以还是用原始提示词函数。
                                # 使用新的 技能提示词
                                new_tool_topic_content = self.constructe_select_tool_system_prompt(skill_name = skill_type)

                                new_skill_sys_prompt = self.constructe_system_prompt(
                                    new_role_system_prompt = role_system_prompt, 
                                    knowledge_system_prompt = '',
                                    history_text = history_text, 
                                    portrait_text ='', 
                                    tool_topic_content = new_tool_topic_content, 
                                    sys_excution_protocol = sys_excution_protocol
                                )
                                
                                messages[0].content = new_skill_sys_prompt
                                logger.info(f"The system prompt has been updated to a skill-specific prompt.\n{messages[0]}")

                        if (len(tool_results) == 1 
                            and tool_results[0].content == '' 
                            and isinstance(tool_results[0].extra, dict) 
                            and "skill_controller" in tool_results[0].extra
                        ): # 这个是skil 控制类，所有的 skill 的tool 返回的结果都是这个标识
                            logger.info(f"BaseAgent::get_llm_reply --- talent_controller_rst:{tool_results[0].extra}")

                            # 判断是否 需要再次调用 llm，如果不调用直接返回结果给前端，如果要调用，需要再次调用模型进行最后的回复
                            if "reasoning_content" in tool_results[0].extra:
                                abc = [Message(role="assistant", content='', extra = tool_results[0].extra)] 
                                yield abc  # 返回工具调用结果 
                            else:
                                # 如果不是tool 调用 返回空
                                yield []  # 返回工具调用结果 
                            return # 结束生成器  不继续执行下面的内容了 ---  ppt不需要模型处理回响应

                        if num_llm_calls_available > 0:
                            messages.append(tool_call_msg.model_dump())
                            #将调用的结果加入到message中去
                            for tool in tool_results:
                                messages.append(tool.model_dump())    # 将 tool_results 消息加入messages
                            continue
                        yield combined_results  # 返回工具调用结果，异步迭代器返回

                return

            # 如果没有响应，默认返回空列表
            yield responses if responses else [] # 异步迭代器返回
            return


    #------------------------------------------------
    # 同步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------        
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:

        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        # 先将  self.tools_list["abc","cdd","ddfe"] 对应的工具获取出来,里面是工具名字
        pass