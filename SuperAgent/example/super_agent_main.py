from fastapi import Query, FastAPI, Request, HTTPException
from typing import AsyncGenerator, Any, Dict, Optional,Tuple,List
import asyncio
import uvicorn
import os
import sys
import json
import re
import httpx
import socket
import time
import traceback

# from pathlib import Path

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
print(a,flush=True)
b = os.path.dirname(a)  #返回上一级目录部分，去掉文件名
print(b,flush=True)
#sys.path.append(b)
c = os.path.dirname(b) #返回上一级目录部分
print(c,flush=True)

# 将上一级目录加入 添加到搜索路径中也 就是examples的上级目录
sys.path.append(c)
print(sys.path,flush=True)

#from src.msg import Message
from src.agent import get_chat_agent_cls,ChatAgent
from starlette.responses import StreamingResponse
import traceback

from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.settings import settings, scheduler_settings
from src.msg import Msg as Message
from src.msg import TextBlock,ThinkingBlock
from src.sandbox.sandbox import Sandbox, init_workspace_and_sandbox, copy_folder_advanced
import subprocess
import signal
from multiprocessing import Process
from src.settings import settings, scheduler_settings

from loguru import logger

# Scheduler 配置
scheduler_enabled = scheduler_settings.enabled if scheduler_settings else True
scheduler_standalone = scheduler_settings.standalone if scheduler_settings else True
scheduler_port = scheduler_settings.port if scheduler_settings else 7702

logger.info(f"Scheduler config: enabled={scheduler_enabled}, standalone={scheduler_standalone}, port={scheduler_port}")

# Scheduler 子进程
scheduler_process: Optional[Process] = None

#-----------------------------------------------------------------
#      配置日志处理，设置为 20M每个文件，一共5个。循环覆盖
#-----------------------------------------------------------------
abc = settings.file_log_enable
print(f"===========file_log_enable:{abc}",flush=True)
# 从yaml配置中获取 日志开关配置，启动的时候 false设置的时候，打印到控制台，true打印的文件
if abc:
    log_dir = os.path.join("./", "log")
    log_path = os.path.join(log_dir, "nuwa_main_log.log")

    # 移除默认控制台输出
    logger.remove()

    # 异步写入日志文件（enqueue=True）
    #logger.add(log_path, level="INFO", enqueue=True)
    # 添加异步日志文件写入
    logger.add(log_path, rotation="10 MB", retention=30, enqueue=True) 
     
#-----------------------------------------------------------------

def start_scheduler_process():
    """
    启动 Scheduler 作为独立子进程

    工作模式：
        - Scheduler 运行在独立进程中
        - 通过配置的端口（如7702）提供服务
        - 与 SuperAgent 互不影响，可独立停止/重启
    """
    global scheduler_process

    if scheduler_process is not None and scheduler_process.is_alive():
        logger.warning("The Scheduler process is already running.")
        return

    logger.info(f"Starting Scheduler process on port: {scheduler_port}")

    # 获取 scheduler_main.py 的路径
    scheduler_main_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scheduler",
        "scheduler_main.py"
    )

    # 启动独立进程
    scheduler_process = subprocess.Popen(
        [sys.executable, scheduler_main_path],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    logger.info(f"Scheduler process started with PID: {scheduler_process.pid}")


def stop_scheduler_process():
    """
    停止 Scheduler 子进程
    """
    global scheduler_process

    if scheduler_process is None:
        return

    if scheduler_process.is_alive():
        logger.info(f"Stopping Scheduler process with PID: {scheduler_process.pid}")
        scheduler_process.terminate()
        try:
            scheduler_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            scheduler_process.kill()
            scheduler_process.wait()
        logger.info("Scheduler process stopped successfully.")
    else:
        logger.info("Scheduler process already stopped.")

    scheduler_process = None

# fastapi的生命周期推荐写法。
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑
    try:
        # 1.初始化沙箱和workspace，同时将md文件拷贝到workspace目录
        sandbox = init_workspace()

        # 2. 初始化agent实例,将沙箱挂载到agent实例中
        logger.info("Initializing ChatAgent...")
        ChatAgentCls = get_chat_agent_cls("chat_agent") # 获取 ChatAgent 类
        app.state.ChatAgent = ChatAgentCls(settings.nuwaagentcfg.agentcfg[0], sandbox) # 默认第一个 agentcfg 是主要配置
        logger.info("ChatAgent initialized successfully")

        # 3. 如果 Scheduler 已启用且不是独立启动模式，则作为子进程启动
        if scheduler_enabled and not scheduler_standalone:
            start_scheduler_process()

        yield  # FastAPI 应用在此运行

    finally:
        # 4. 关闭 Scheduler 子进程
        if scheduler_enabled and not scheduler_standalone:
            stop_scheduler_process()

        # 关闭逻辑
        logger.info("Starting cleanup...")
        logger.info("Cleanup completed")

# fastAPI
app = FastAPI(lifespan=lifespan)

register = None

def init_workspace():
    # 初始化沙箱和workspace
    sandbox = init_workspace_and_sandbox()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../superagentconfig/md"))
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./workspace/md"))
    logger.info(f"copy_folder_advanced src_dir:{src_dir} dest_dir:{dest_dir}") # md文件拷贝到workspace目录
    cp_count = copy_folder_advanced(src_dir, dest_dir, recursive=True)
    return sandbox # 返回沙箱实例

# 根据新消息格式修改
async def parse_user_message(request_content):
    #logger.info(f"parse_user_message--->begin!! request_content:{request_content}")
    # session id 会话id 字段
    data = request_content.get("data", {})
    #从data中获取用户输入的消息
    user_msg = data.get("content", '')   # 用户输入的消息
    # 从data中获取用户输入的消息的扩展字段
    metadata = data.get("metadata", {})
    
    logger.info(f"======》process_input user_id：{user_msg}")

    if not isinstance(user_msg, str) or not user_msg.strip():
        response = {"code":1005, "msg":"user_message None", "type":"chat", "version":"1", "data":{}}
        logger.info(response)
        return False, response

    messages = [
        {
            'role': 'user',
            'content':user_msg,
            'metadata': metadata
        },
    ]

    #logger.info(f"parse_user_message============>message:{messages}")
    return True, messages

async def agent_generate_response(mychatagent:ChatAgent, request_content:dict):
    logger.info(f"agent_generate_response--->begin!!!!!!!!")
    try:
        parse_status, new_messages =  await parse_user_message(request_content)
        if parse_status == False:
            new_messages_json =  (json.dumps(new_messages, ensure_ascii=False) + "\n")
            yield new_messages_json # 直接返回错误了
        else:
            logger.info(f"agent_generate_response--->new_messages:{new_messages}")

            #-----------------------------
            # 将扩展字段取出来，正在响应中带回去给请求端
            #-----------------------------
            #ext_info = new_messages[0]["ext_info"]
            #logger.info(f"muwa_generate_response--->ext_info:{ext_info}")

            # 记录调用开始时间
            start_time = time.perf_counter()
            #logger.info(f"receive_message--->Time to begin!!!!!!!!")
            first_packet_logged = False  # 标记是否已记录第一个报文时间
            
            # 用户正常的消息，调用推理模型
            responses = await mychatagent.a_generate_rsp(new_messages)
            #--------------------------------------------------------------
            text = '' # 累积所有响应内容，不做判断等处理
            think_text = '' # 累积所有思考内容，不做判断等处理
            tool_text = '' # 累积所有工具调用内容，不做判断等处理
            text_blocks = [] # 累积所有文本内容块，不做判断等处理
            thinking_blocks = [] # 累积所有思考内容块，不做判断等处理
            #--------------------------------------------------------------
            async for response in responses:
                #logger.info(f"text Response: =======>{response}")
                # 如果是think内容
                if response is None or len(response) == 0:
                    continue

                if not first_packet_logged:
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logger.info(f"Time to first packet: {elapsed_time:.3f} seconds")
                    first_packet_logged = True
                
                response_msg:Message = response[0]
                text_blocks:List[TextBlock] = response_msg.get_content_blocks(block_type="text") # 获取文本内容块
                thinking_blocks:List[ThinkingBlock] = response_msg.get_content_blocks(block_type="thinking") # 获取思考内容块
                response_data = response_msg.to_dict()
                #logger.info(f"agent_generate_response--->response_msg:{response_data}")

                # 累积所有think内容，text内容和tool调用的内容供调试信息使用
                if len(text_blocks) > 0:
                    text += text_blocks[0].get("text", "") # 获取文本内容块
                if len(thinking_blocks) > 0:
                    think_text += thinking_blocks[0].get("thinking", "") # 获取思考内容块
                if response_msg.role == "function":
                    tool_calls = response_msg.tool_calls or None # 如果有工具调用
                    for tool_call in tool_calls:
                        function = tool_call.get("function", None) 
                        function_name = function.get("name", None) # 获取工具调用的函数名
                        function_args = function.get("arguments", None) # 获取工具调用的参数
                        tool_call_text = f"\n工具调用：【{function_name}】【{function_args}】"
                        tool_text += tool_call_text # 获取工具调用的内容
                
                agent_response = {"code": 200, "type": "chat", "version": "1", "data": response_data}
                #logger.info(f"agent_generate_response--->agent_response:{agent_response}")
                yield json.dumps(agent_response, ensure_ascii=False) + "\n"
                    
            
            logger.debug(f"agent_generate_response: text Response: =======>reasoning_content:{think_text}\n\ntool_call_content:{tool_text}\n\ncontent:\n{text}\n\n")

    except asyncio.CancelledError:
        # 增加用户 cancel的异常处理
        logger.warning("=====>>>【The task was canceled by user！！!】")
        raise
    except Exception as e:
        logger.error(f"❌An anomaly occurred: {type(e).__name__}: {e}")
        raise

# 异步调用，并且返回响应
async def generate_response(mychatagent:ChatAgent, request_content:dict) -> StreamingResponse:
    try:
        generator = agent_generate_response(mychatagent, request_content)
        # return StreamingResponse(generator, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return StreamingResponse(generator, media_type="text/event-stream")
    except RuntimeError as e:
        print(f"Caught error: {e}")
        traceback.print_exc()
        raise

#接收来自客户端的聊天请求消息
@app.post("/v1/chat/completions")
async def receive_message(request: Request):
    # 从 app.state 获取实例
    mychatagent = request.app.state.ChatAgent
    # print"Attempting to parse request body as JSON...")
    try:
        # 尝试使用 request.json() 解析请求体
        request_content = await request.json()
        logger.debug(f"===================>receive_message request_content:{request_content}\n\n\n")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        # 如果解析失败，回退到使用 request.body() 并手动解码
        try:
            body = await request.body()
            request_content = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as inner_e:
            print(f"Failed to decode body: {inner_e}")
            raise HTTPException(status_code=401, detail=f"Invalid JSON format or encoding : {inner_e}")

    if not isinstance(request_content, dict):
        print("Parsed content is not a dictionary")
        raise HTTPException(status_code=402, detail=f"Invalid JSON format: Not an object")

    try:
        response = await generate_response(mychatagent, request_content) 
    except Exception as e:
        traceback.print_exc()
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    return response

# http请求错误记录中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.warning(f"Request processing failed: {e}")
        raise



# 启动应用
if __name__ == "__main__":
    logger.info("===================start up step to main")
    #init_data()
    #register_service()
    #config = uvicorn.Config("agent_main:app", host="0.0.0.0", port=int(que_config['my_queing_port']), log_level="debug", access_log=True)
    #server = uvicorn.Server(config)
    #server.run()
    # 在主程序中启动调试器
    # threading.Thread(target=debug_threads, daemon=True, name="DebugThread").start()
    def is_port_in_use(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except OSError:
                return True

    # 启动前预检查
    if is_port_in_use("0.0.0.0", settings.agent_listen_port):
        logger.error(f"Port {settings.agent_listen_port} is already in use")
        exit(1)

    uvicorn.run("super_agent_main:app", host="0.0.0.0", port=settings.agent_listen_port, log_level="debug")
