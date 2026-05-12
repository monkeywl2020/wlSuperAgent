# wlSuperAgent 智能助手框架

> 一个基于 LLM 的智能 Agent 框架，支持工具调用、记忆系统、沙箱隔离和多 Agent 协作。

---

## 📁 项目结构

```
wlSuperAgent/
├── SuperAgent/                    # 核心框架代码
│   ├── src/
│   │   ├── agent/                 # 🤖 Agent 核心模块
│   │   │   ├── core/
│   │   │   │   ├── base_agent.py  # Agent 基类
│   │   │   │   └── operator.py    # 操作符基类
│   │   │   └── eq_agent.py        # 任务执行 Agent
│   │   │
│   │   ├── tools/                 # 🔧 工具系统
│   │   │   ├── base.py            # 工具基类
│   │   │   └── agent_tools.py     # 核心工具实现
│   │   │
│   │   ├── llm/                   # 💬 LLM 模型客户端
│   │   │   ├── core/
│   │   │   │   └── base.py        # LLM 基类
│   │   │   ├── openai_model_client.py   # OpenAI 兼容客户端
│   │   │   └── post_model_client.py     # 后处理客户端
│   │   │
│   │   ├── memory/                # 🧠 记忆系统
│   │   │   ├── file_memory.py     # 记忆主类
│   │   │   ├── vector_store.py    # 向量存储
│   │   │   ├── embedding.py       # Embedding 接口
│   │   │   ├── llm.py             # LLM 接口
│   │   │   ├── search/            # 搜索引擎
│   │   │   │   ├── file_store.py  # 混合搜索
│   │   │   │   ├── bm25_index.py  # BM25 索引
│   │   │   │   └── qa_extractor.py # QA 提取
│   │   │   ├── compression/       # 压缩模块
│   │   │   └── storage/           # 存储模块
│   │   │
│   │   ├── sandbox/               # 🐳 沙箱环境
│   │   │   └── sandbox.py         # Docker 沙箱
│   │   │
│   │   ├── msg/                   # 📩 消息结构
│   │   └── settings.py            # ⚙️ 配置管理
│   │
│   └── example/                   # 示例代码
│       └── workspace/             # 工作目录
│
├── superagentconfig/              # 🎯 Agent 配置
│   ├── config.yaml                # Agent 主配置
│   ├── sandbox.yaml               # 沙箱配置
│   ├── scheduler_config.yaml      # 调度器配置
│   ├── memory_config.yaml         # 记忆配置
│   └── md/                        # 📄 提示词模板
│       └── cn/
│           ├── AGENTS.md          # Agent 规范
│           ├── SOUL.md            # 身份灵魂
│           ├── USER.md            # 用户信息
│           ├── IDENTITY.md        # 身份标识
│           ├── TOOLS.md           # 工具说明
│           ├── HEARTBEAT.md       # 心跳配置
│           └── BOOTSTRAP.md       # 初始化引导
│
└── memory_data/                   # 💾 记忆数据目录
    ├── dialog/                    # 会话对话记录
    ├── memory/                    # 每日记忆摘要
    ├── MEMORY.md                  # 长期记忆
    ├── memory_qa/                 # QA 对索引
    ├── memory_index/              # 向量索引
    └── bm25_index/                # BM25 索引
```

---

## 🏗️ 核心模块

### 1. Agent 系统

**基类：`BaseAgent`**

所有 Agent 的父类，继承自 `Operator`，管理工具注册、LLM 调用、Tool Call 处理。

```python
from src.agent.core.base_agent import get_chat_agent_cls

# 获取 Agent 类
AgentCls = get_chat_agent_cls('chat_agent')

# 实例化
agent = AgentCls(agent_config, sandbox)

# 异步生成响应
async for response in agent.a_generate_rsp(messages):
    print(response)
```

**核心方法：**

| 方法 | 说明 |
|------|------|
| `a_generate_rsp()` | 异步生成响应（主入口） |
| `get_llm_reply()` | 获取 LLM 回复 |
| `_handle_tool_calls()` | 处理工具调用 |
| `_a_call_tool()` | 执行工具 |
| `_detect_tool()` | 检测 tool_call |

---

### 2. Tool 系统

**基类：`BaseTool`**

使用装饰器注册工具：

```python
from src.tools.base import BaseTool, agent_register_tool

@agent_register_tool('my_tool')
class MyTool(BaseTool):
    name = 'my_tool'
    description = '我的自定义工具'
    parameters = [
        {'name': 'query', 'type': 'str', 'description': '查询内容', 'required': True},
    ]
    
    async def call(self, params, **kwargs) -> str:
        # 业务逻辑
        return result
```

**参数格式转换：**

```python
# list 格式（自动转换为 OpenAI 格式）
parameters = [
    {'name': 'query', 'type': 'str', 'description': '查询'},
]

# dict 格式（OpenAI JSON Schema）
parameters = {
    "type": "object",
    "properties": {"query": {"type": "str", "description": "查询"}},
    "required": ["query"]
}
```

**内置工具：**

| 工具名 | 功能 |
|--------|------|
| `exec` | 在沙箱中执行 shell 命令 |
| `read_file` | 读取文件内容（支持二进制） |
| `write_file` | 写入文件内容（支持二进制） |
| `web_search` | 搜索互联网 |
| `web_fetch` | 获取网页内容 |
| `ask_user` | 向用户提问（最后手段） |

---

### 3. 记忆系统

**主类：`FileMemory`**

分层记忆架构：

```
┌─────────────────────────────────────────────────────┐
│                    FileMemory                        │
├─────────────────────────────────────────────────────┤
│  内存层    │  self.messages (当前会话消息)           │
│            │  self.current_summary (摘要)            │
├─────────────────────────────────────────────────────┤
│  文件层    │  dialog/*.jsonl (对话记录)              │
│            │  memory/YYYY-MM-DD.md (每日摘要)        │
│            │  MEMORY.md (长期记忆)                   │
├─────────────────────────────────────────────────────┤
│  索引层    │  memory_index/ (向量索引 FAISS)         │
│            │  bm25_index/ (BM25 关键词索引)          │
└─────────────────────────────────────────────────────┘
```

**核心功能：**

```python
# 添加消息
memory.add_message('user', '你好', metadata={'session_id': 'xxx'})

# 搜索记忆
results = memory.memory_search('之前我让你做什么来着？')

# 获取对话上下文
context = memory.get_memory()

# 检查并压缩（token 超阈值时）
memory.check_and_compact()
```

**混合搜索：**

```python
# 向量搜索 + BM25 = 混合搜索
results = file_store.search(
    query,
    vector_weight=0.7,   # 向量权重
    bm25_weight=0.3,     # BM25 权重
    top_k=5
)
```

---

### 4. 沙箱系统

**主类：`Sandbox`**

基于 Docker 的隔离执行环境：

```python
from src.sandbox.sandbox import Sandbox

# 启动沙箱
sb = Sandbox()

# 执行命令
result = sb.exec("python --version")
# {'exit_code': 0, 'stdout': 'Python 3.12.x', 'stderr': ''}

# 读写文件
await sb.write_file("test.txt", "Hello")
content = await sb.read_file("test.txt")

# 清理
sb.cleanup()
```

**特性：**

- 自动启动/清理 Docker 容器
- workspace 目录双向共享
- 支持资源限制（内存、CPU）
- 支持 base64 编解码（二进制文件）

---

## 🔄 请求处理流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        HTTP 请求入口                              │
│                   POST /api/agent/{agent_name}/chat             │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│  1️⃣ 预处理                                                        │
│  • 解析请求体 (messages, context)                                │
│  • 加载 Agent 配置 (config.yaml)                                 │
│  • 初始化 Sandbox 沙箱                                            │
│  • 构建工具列表 (tools_list)                                     │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│  2️⃣ 上下文构建                                                    │
│  • 合并 system_prompt + md 文件模板                               │
│  • 注入动态变量:                                                  │
│    - {current_time} → 当前时间                                   │
│    - {workspace} → 工作目录                                      │
│    - {knowledge_content} → RAG 检索结果                           │
│    - {memory_text} → 记忆检索结果                                │
│  • 拼接历史 messages                                              │
└──────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│  3️⃣ LLM 调用                                                      │
│  • 调用 get_llm_reply()                                          │
│  • 构造 OpenAI 格式请求                                           │
│  • 流式返回响应                                                   │
└──────────────────────────────────────────────────────────────────┘
                                ↓
                    ┌───────────────┬───────────────┐
                    │ 普通回复       │ Tool Call     │
                    └───────┬───────┴───────┬───────┘
                            │               │
                            ↓               ↓
                        返回用户        执行工具
                                        ↓
                              ┌───────────────┐
                              │ 工具执行阶段   │
                              │ _handle_tool  │
                              │ _calls()      │
                              └───────┬───────┘
                                      │
                                      ↓
                              再次调用 LLM
                              （循环直到结束）
```

---

## ⚙️ 配置系统

**config.yaml 结构：**

```yaml
nuwaagentcfg:
  agentcfg:
    - agent_type: chat_agent          # Agent 类型
      agent_name: agent_abc           # Agent 名称
      description: 情感聊天Agent      # 描述
      
      sys_prompt_cn: |               # 中文提示词（支持模板变量）
        你是一名运行于 wlsuper_agent 内部的个人助理。
        当前时间：{current_time}
        工作目录：{workspace}
      
      tools_list:                     # 挂载的工具
        - exec
        - read_file
        - write_file
        - web_search
        - web_fetch
        
      llmcfg:                         # LLM 配置
        model_type: openai
        model: minimax
        base_url: http://xxx:60005/v1
        api_key: EMPTY
        temperature: 0.7
        top_p: 0.95

chat_template:                        # 聊天模板
  cn:
    workspace_perfix: |               # 工作目录前缀
      您的工作目录为：/workspace
    timezone_perfix: |                # 时区前缀
      当前时间：{current_time}

file_log_enable: True                 # 日志开关
agent_listen_port: 7701              # 监听端口
gradio_test_port: 7860               # 测试端口
sandbox_container_name: agent-sandbox # 容器名称
```

**模板变量：**

| 变量 | 说明 |
|------|------|
| `{current_time}` | 当前时间 (Asia/Shanghai) |
| `{workspace}` | 工作目录 (/workspace) |
| `{knowledge_content}` | 知识库 RAG 结果 |
| `{memory_text}` | 记忆检索结果 |
| `{skill_context}` | 技能上下文 |

---

## 🔧 快速开发

### 新增工具

```python
# 1. 继承 BaseTool
# 2. 用 @agent_register_tool 注册
# 3. 实现 call() 方法

from src.tools.base import BaseTool, agent_register_tool

@agent_register_tool('my_tool')
class MyTool(BaseTool):
    name = 'my_tool'
    description = '我的自定义工具'
    parameters = [
        {'name': 'query', 'type': 'str', 'description': '查询内容', 'required': True},
    ]
    
    async def call(self, params, **kwargs) -> str:
        # 解析参数
        params = self._verify_json_format_args(params)
        query = params['query']
        
        # 业务逻辑
        result = f"处理了: {query}"
        return result
```

### 新增 Agent

```python
from src.agent.core.base_agent import BaseAgent, register_agent

@register_agent('my_agent')
class MyAgent(BaseAgent):
    
    async def a_generate_rsp(self, messages, **kwargs):
        # 自定义响应生成逻辑
        pass
    
    async def get_llm_reply(self, messages, **kwargs):
        # 自定义 LLM 调用逻辑
        pass
    
    def generate_rsp(self, messages, **kwargs):
        # 同步版本
        pass
```

### 注册 Agent

在 `config.yaml` 中添加：

```yaml
nuwaagentcfg:
  agentcfg:
    - agent_type: my_agent
      agent_name: my_agent_1
      # ... 其他配置
```

---

## 🚀 启动示例

```python
import asyncio
from src.agent.core.base_agent import get_chat_agent_cls
from src.sandbox.sandbox import Sandbox
from src.settings import AgentConfig

async def main():
    # 1. 启动沙箱
    sandbox = Sandbox()
    
    # 2. 创建 Agent 配置
    config = AgentConfig(
        agent_name="agent_abc",
        sys_prompt_cn="你是我的助手",
        llmcfg={...},
        tools_list=["exec", "read_file", "write_file"]
    )
    
    # 3. 获取 Agent 类并实例化
    AgentCls = get_chat_agent_cls('chat_agent')
    agent = AgentCls(config, sandbox)
    
    # 4. 对话
    messages = [{"role": "user", "content": "你好"}]
    async for response in agent.a_generate_rsp(messages):
        print(response)

asyncio.run(main())
```

---

## 📝 许可证

MIT License