import gradio as gr
import requests
import json
import time
import re
import uuid
import os
import sys
from typing import Generator

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

from src.settings import settings
from loguru import logger

# 服务器地址
server_port = settings.agent_listen_port
SERVER_URL = f"http://localhost:{server_port}/v1/chat/completions"

# 全局 session_id，在模块加载时生成一次，整个会话复用同一个 session_id
GLOBAL_SESSION_ID = f"session_{uuid.uuid4()}"
logger.info(f"初始化全局 session_id: {GLOBAL_SESSION_ID}")

# 处理流式响应的函数
def stream_response(message: str, history: list) -> Generator[str, None, None]:
    # 复用全局 session_id，保证同一会话的所有消息使用相同的 session_id

    # 构造请求数据
    payload = {
        "data": {
            "content": message,  # 用户输入的内容
            'metadata': {
                "session_id": GLOBAL_SESSION_ID,
                "language": "中文"
            }
        }
    }
    
    logger.info(f"===========stream_response:{message}")
    # 记录开始时间
    start_time = time.time()

    # 发送 POST 请求，启用流式响应
    try:
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )
        response.raise_for_status()

        accumulated_reasoning = ""  # 累积推理内容
        accumulated_content = ""    # 累积正式回答内容
        first_packet_time = None

        for line in response.iter_lines():
            if line:
                # 解码并移除 "data: " 前缀
                decoded_line = line.decode("utf-8").replace("data: ", "")
                #logger.info(f"=====decoded_line:{decoded_line}")
                try:
                    data = json.loads(decoded_line)

                    think_content = ""
                    content = ""

                    # 提取 reasoning_content 和 content
                    data_dict = data.get("data", {})
                    role = data_dict.get("role", "assistant") # 默认是接收来自Assistant的响应内容。
                    if role == "function":
                        tool_calls = data_dict.get("tool_calls", [])
                        tool_call_content = ""
                        for tool_call in tool_calls:
                            function = tool_call.get("function", None) 
                            function_name = function.get("name", None) # 获取工具调用的函数名
                            function_args = function.get("arguments", None) # 获取工具调用的参数
                            tool_call_text = f"\n工具调用：【{function_name}】【{function_args}】"
                            tool_call_content += tool_call_text # 获取工具调用的内容
                        # tool_call_content 合并到 think_content
                        think_content += tool_call_content + "\n"
                    else:
                        # 提取 think 和 text 内容
                        content_blocks = data_dict.get("content", [])
                        if content_blocks and len(content_blocks) > 0:
                            content_type = content_blocks[0].get("type", "text")
                            if content_type == "thinking":
                                think_content = content_blocks[0].get("thinking", "")
                            else:
                                content = content_blocks[0].get("text", "")
                    

                    # 累积内容
                    accumulated_reasoning += think_content
                    accumulated_content += content

                    # 构造显示内容：蓝色推理 + 黑色回答
                    display_parts = []

                    if accumulated_reasoning.strip():
                        # 蓝色显示推理内容
                        accumulated_reasoning = accumulated_reasoning.replace("<", "＜").replace(">", "＞") # 替换尖括号，避免HTML解析问题
                        blue_part = f"<span style='color:blue'>{accumulated_reasoning.strip().replace(chr(10), '<br>')}</span>"
                        display_parts.append(blue_part)

                    if accumulated_content.strip():
                        # 黑色显示正式回答（默认就是黑色，可不加 style）
                        black_part = accumulated_content.strip().replace(chr(10), '<br>')
                        display_parts.append(black_part)

                    # 合并显示内容
                    formatted_content = "<br><br>".join(display_parts) if display_parts else ""

                    # 首包时间 & 输出
                    if first_packet_time is None:
                        first_packet_time = time.time() - start_time

                    formatted_content = re.sub(r'</?zh\s*>', '', formatted_content, flags=re.IGNORECASE)

                    #logger.info(f"=====formatted_content:{formatted_content}")
                    # 每次打印结果，都要显示首包时间，否则后续流式输出，首包时间就看不到了
                    yield f"{formatted_content}<br><br>首条报文响应时间: {first_packet_time:.2f}秒"
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as e:
        yield f"请求失败: {str(e)}"

# Gradio 聊天界面
def create_chat_interface():
    with gr.Blocks(title="大模型响应速度测试") as demo:
        gr.Markdown("# 大模型响应速度测试工具")
        gr.Markdown("输入消息内容，测试模型的流式响应速度。蓝色文字为 `think` 内容，仅显示首条报文时间。")

        # 使用 Chatbot 组件展示对话
        chatbot = gr.Chatbot(label="对话历史", height=400)

        # 输入框和发送按钮
        msg = gr.Textbox(label="输入消息", placeholder="请输入消息内容，例如：你好")
        submit_btn = gr.Button("发送")

        # 定义流式响应的逻辑
        def respond(message, chat_history):
            # 将用户消息添加到历史
            chat_history.append([message, None])
            # 生成流式响应
            for response in stream_response(message, chat_history):
                chat_history[-1][1] = response  # 更新最后一条消息的响应
                #print(f"==========chat_history:{chat_history}")
                yield chat_history, ""  # 返回更新后的历史记录和空字符串以清空输入框

        # 点击发送按钮触发响应，并清空输入框
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],  # 输出到 chatbot 和 msg，清空输入框
        )

        # 按回车键也可以触发，并清空输入框
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],  # 输出到 chatbot 和 msg，清空输入框
        )

    return demo

# 启动 Gradio 应用，指定地址和端口
if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch(
        server_name="0.0.0.0",  # 监听地址，改为本地监听
        server_port=settings.gradio_test_port # 监听端口
    )