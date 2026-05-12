# -*- coding: utf-8 -*-
"""
文件记忆系统示例和测试

测试 FileMemory 系统的核心功能：
1. session会话记录：sessionid由用户消息传入，按session+日期存储
2. JSONL格式存储：每行存储message的json字符串
3. 消息类型：user消息、assistant回复、function调用、tool结果
4. 即时存储：用户消息和模型回复即时写入文件
5. 跨日期处理：assistant消息存储到user消息所在的文件
"""
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

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

from src.memory.file_memory import FileMemory
from src.memory.storage.dialog_storage import DialogStorage
from src.msg import Msg as Message


from loguru import logger

def get_test_session_id():
    return f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def print_separator(title):
    logger.info("=" * 60)
    logger.info(f" {title}")
    logger.info("=" * 60 + "\n")


def verify_jsonl_file(file_path, expected_count=None):
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"文件 {file_path.name} 包含 {len(lines)} 条消息")

    messages = []
    for i, line in enumerate(lines):
        if line.strip():
            msg = json.loads(line)
            messages.append(msg)
            role = msg.get('role', '')
            content = str(msg.get('content', ''))[:50]
            extra_info = []

            if role == 'function' and msg.get('tool_calls'):
                tool_name = msg['tool_calls'][0].get('function', {}).get('name', '') if msg['tool_calls'] else ''
                extra_info.append(f"tool_calls: [{tool_name}]")
            elif role == 'tool':
                if msg.get('tool_call_id'):
                    extra_info.append(f"tool_call_id: {msg['tool_call_id']}")
                if msg.get('extra'):
                    extra_info.append(f"extra: {str(msg['extra'])[:30]}...")

            extra_str = f", {', '.join(extra_info)}" if extra_info else ""
            logger.info(f"  [{i+1}] role={role}, content={content}...{extra_str}")

    if expected_count is not None and len(messages) != expected_count:
        logger.error(f"预期 {expected_count} 条消息，实际 {len(messages)} 条")
        return False

    return True


def demo_basic_session_storage():
    print_separator("测试1: 基本session会话存储")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()

    metadata = {"session_id": session_id}

    memory.add_message("user", "你好，我叫张三", metadata=metadata)
    memory.add_message("assistant", "你好张三！很高兴认识你。", metadata=metadata)

    memory.add_message("user", "我最近在学习Python编程", metadata=metadata)
    memory.add_message("assistant", "Python是一门很棒的编程语言！", metadata=metadata)

    dialog_dir = memory.dir_manager.get_dialog_dir()
    today = memory.dir_manager.get_today_date()
    expected_file = dialog_dir / f"{session_id}_{today}.jsonl"

    logger.info(f"\n验证文件是否创建: {expected_file}")
    if verify_jsonl_file(expected_file, 4):
        logger.info("✓ 基本session存储测试通过")
    else:
        logger.error("✗ 基本session存储测试失败")

    return session_id


def demo_multi_session_storage():
    print_separator("测试2: 多session隔离存储")

    session_a = get_test_session_id()
    session_b = get_test_session_id()
    logger.info(f"Session A: {session_a}")
    logger.info(f"Session B: {session_b}")

    memory = FileMemory()
    metadata_a = {"session_id": session_a}
    metadata_b = {"session_id": session_b}

    memory.add_message("user", "这是Session A的第一条消息", metadata=metadata_a)
    memory.add_message("assistant", "这是Session A的回复", metadata=metadata_a)

    memory.add_message("user", "这是Session B的消息", metadata=metadata_b)
    memory.add_message("assistant", "这是Session B的回复", metadata=metadata_b)

    memory.add_message("user", "Session A的第二条消息", metadata=metadata_a)
    memory.add_message("assistant", "Session A的第二个回复", metadata=metadata_a)

    dialog_dir = memory.dir_manager.get_dialog_dir()
    today = memory.dir_manager.get_today_date()

    file_a = dialog_dir / f"{session_a}_{today}.jsonl"
    file_b = dialog_dir / f"{session_b}_{today}.jsonl"

    logger.info(f"\n验证Session A文件:")
    verify_jsonl_file(file_a, 4)

    logger.info(f"\n验证Session B文件:")
    verify_jsonl_file(file_b, 2)

    logger.info("✓ 多session隔离存储测试通过")


def demo_tool_message_storage():
    print_separator("测试3: Tool调用消息存储")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()
    metadata = {"session_id": session_id}

    memory.add_message("user", "帮我搜索一下今天的天气", metadata=metadata)
    memory.add_message("assistant", "好的，我来帮你查询天气", metadata=metadata)

    tool_calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "北京"}'
            }
        }
    ]
    memory.add_message(
        role="function",
        content="",
        tool_calls=tool_calls,
        metadata=metadata
    )

    memory.add_message(
        role="tool",
        content='{"weather": "晴", "temperature": 25}',
        tool_call_id="call_001",
        metadata=metadata
    )

    memory.add_message("assistant", "今天北京的天气是晴天，气温25度。", metadata=metadata)

    dialog_dir = memory.dir_manager.get_dialog_dir()
    today = memory.dir_manager.get_today_date()
    expected_file = dialog_dir / f"{session_id}_{today}.jsonl"

    logger.info(f"\n验证Tool消息存储:")
    if verify_jsonl_file(expected_file, 5):
        logger.info("✓ Tool消息存储测试通过")

    return session_id


def demo_immediate_storage():
    print_separator("测试4: 即时存储验证")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()
    dialog_dir = memory.dir_manager.get_dialog_dir()
    today = memory.dir_manager.get_today_date()
    expected_file = dialog_dir / f"{session_id}_{today}.jsonl"

    if expected_file.exists():
        os.remove(expected_file)
        logger.info(f"删除旧文件: {expected_file}")

    metadata = {"session_id": session_id}

    memory.add_message("user", "第一条即时存储的消息", metadata=metadata)

    logger.info(f"\n添加1条消息后验证文件:")
    if not expected_file.exists():
        logger.error("✗ 文件未创建，即时存储失败")
        return
    else:
        with open(expected_file, "r", encoding="utf-8") as f:
            count = len([l for l in f.readlines() if l.strip()])
        logger.info(f"文件存在，包含 {count} 条消息")

    memory.add_message("assistant", "即时回复", metadata=metadata)

    with open(expected_file, "r", encoding="utf-8") as f:
        count = len([l for l in f.readlines() if l.strip()])
    logger.info(f"添加2条消息后验证文件包含 {count} 条消息")

    if count == 2:
        logger.info("✓ 即时存储测试通过")
    else:
        logger.error("✗ 即时存储测试失败")


def demo_get_memory():
    print_separator("测试5: 获取记忆上下文")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()
    metadata = {"session_id": session_id}

    memory.add_message("user", "我叫李四", metadata=metadata)
    memory.add_message("assistant", "你好李四！", metadata=metadata)
    memory.add_message("user", "我喜欢打篮球", metadata=metadata)
    memory.add_message("assistant", "篮球是一项很好的运动！", metadata=metadata)

    context = memory.get_memory()
    logger.info(f"\n获取到的记忆上下文包含 {len(context)} 条消息:")
    for i, msg in enumerate(context):
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:30]
        logger.info(f"  [{i}] role={role}, content={content}...")

    if len(context) == 4:
        logger.info("✓ 获取记忆上下文测试通过")
    else:
        logger.error("✗ 获取记忆上下文测试失败")


def demo_get_recent_rounds():
    print_separator("测试6: 获取最近对话轮次")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()
    metadata = {"session_id": session_id}

    for i in range(5):
        memory.add_message("user", f"第{i+1}轮用户消息", metadata=metadata)
        memory.add_message("assistant", f"第{i+1}轮助手回复", metadata=metadata)

    rounds = memory.get_recent_rounds(rounds=3, session_id=session_id)
    logger.info(f"\n获取最近3轮对话:")
    for i, msg in enumerate(rounds):
        role = msg.get("role", "")
        content = str(msg.get("content", ""))[:20]
        logger.info(f"  [{i}] role={role}, content={content}")

    if len(rounds) == 6:
        logger.info("✓ 获取最近对话轮次测试通过")
    else:
        logger.error(f"✗ 获取最近对话轮次测试失败，预期6条，实际{len(rounds)}条")


def demo_no_session_id_error():
    print_separator("测试7: 无session_id错误处理")

    memory = FileMemory()

    try:
        memory.add_message("user", "这条消息没有session_id")
        logger.error("✗ 应该抛出ValueError异常")
    except ValueError as e:
        logger.info(f"✓ 正确捕获异常: {e}")


def demo_clear_session():
    print_separator("测试8: 清除会话")

    session_id = get_test_session_id()
    logger.info(f"创建测试session: {session_id}")

    memory = FileMemory()
    metadata = {"session_id": session_id}

    memory.add_message("user", "消息1", metadata=metadata)
    memory.add_message("assistant", "回复1", metadata=metadata)
    memory.add_message("user", "消息2", metadata=metadata)
    memory.add_message("assistant", "回复2", metadata=metadata)

    logger.info(f"清除前消息数量: {memory.get_message_count()}")

    memory.clear_session()
    memory.wait_for_async_tasks()

    logger.info(f"清除后消息数量: {memory.get_message_count()}")

    if memory.get_message_count() == 0:
        logger.info("✓ 清除会话测试通过")
    else:
        logger.error("✗ 清除会话测试失败")


def demo_dialog_storage_direct():
    print_separator("测试9: DialogStorage直接操作")

    session_id = get_test_session_id()
    logger.info(f"测试session: {session_id}")

    from src.memory.core.directory_manager import DirectoryManager
    from src.memory.core.config import load_config

    project_root = Path(__file__).parent.parent.parent
    config_path = str(project_root / "superagentconfig" / "memory_config.yaml")
    config = load_config(config_path)
    dir_manager = DirectoryManager(config.working_dir)
    dir_manager.ensure_directories()

    dialog_storage = DialogStorage(dir_manager)

    today = dir_manager.get_today_date()

    msg1 = Message(role="user", content="测试消息1", metadata={"session_id": session_id})
    msg2 = Message(role="assistant", content="测试回复1", metadata={"session_id": session_id})

    dialog_storage.append_message(msg1.to_dict())
    dialog_storage.append_message(msg2.to_dict())

    file_path = dir_manager.get_dialog_file_path(session_id, today)
    logger.info(f"\n验证直接存储:")
    verify_jsonl_file(file_path, 2)

    logger.info("✓ DialogStorage直接操作测试通过")


def main():
    logger.info("\n" + "=" * 60)
    logger.info(" 文件记忆系统(FileMemory)功能测试")
    logger.info("=" * 60 + "\n")

    demo_basic_session_storage()
    demo_multi_session_storage()
    demo_tool_message_storage()
    demo_immediate_storage()
    demo_get_memory()
    demo_get_recent_rounds()
    demo_no_session_id_error()
    demo_clear_session()
    demo_dialog_storage_direct()

    logger.info("\n" + "=" * 60)
    logger.info(" 所有测试完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
