import json5
from typing import List, Dict, Optional, Any
import textwrap

from .base import agent_register_tool,BaseTool
from src.sandbox.sandbox import Sandbox
from loguru import logger

# 元工具 exec 执行系统命令
@agent_register_tool('exec')
class exec(BaseTool):
    description =  (
        "在隔离的 Python 3.12 Docker 沙箱中执行代码或 shell 命令。\n"
        "这是最强大的工具——当其他工具无法完成任务时，写代码来解决。\n\n"

        "【两种模式】\n"
        "- language='bash'：直接执行 shell 命令，适合：\n"
        "    安装依赖（pip install xxx）\n"
        "    文件操作（ls / mkdir / cp / curl 下载）\n"
        "    调用命令行工具\n"
        "- language='python'：执行多行 Python 脚本，适合：\n"
        "    数据处理、格式转换、逻辑计算\n"
        "    调用已安装的第三方库\n"
        "    生成文件（图表、JSON、CSV 等）\n\n"

        "【典型工作流】\n"
        "  Step1: exec(bash,   'pip install pandas openpyxl -q')\n"
        "  Step2: exec(python, '..处理逻辑..')\n"
        "  Step3: read_file('output/result.xlsx')\n\n"

        "【文件路径规范】\n"
        "  工作目录固定为 /workspace，子目录建议：\n"
        "    /workspace/input/   — 输入数据\n"
        "    /workspace/scripts/ — 临时脚本\n"
        "    /workspace/output/  — 最终产物\n\n"

        "【返回格式】成功时返回 stdout；失败时返回 exit_code + stderr，据此分析原因并重试。\n\n"

        "【重要】执行 Python 时，用 print() 输出结果，否则 Observation 里看不到任何内容。"
    )
    parameters = [{
        "name": "language",
        "type": "string", 
        "description": (
            "'bash'：执行 shell 命令字符串，如 'pip install rich && echo done'\n"
            "'python'：执行完整 Python 脚本，支持多行、import、文件读写等"
        ), 
        'enum':["python", "bash"],
        'required': True
    },{
        "name": "code",
        "type": "string", 
        "description": (
            "要执行的完整代码或命令。\n\n"
            "bash 示例：\n"
            "  pip install pandas matplotlib -q\n\n"
            "python 示例：\n"
            "  import pandas as pd\n"
            "  df = pd.read_csv('/workspace/input/data.csv')\n"
            "  print(df.shape)\n"
            "  df.describe().to_csv('/workspace/output/summary.csv')\n"
            "  print('done')\n\n"
            "注意：Python 脚本会被写入 /workspace/scripts/_tmp.py 再执行，"
            "路径相关操作务必用 /workspace/... 绝对路径。"
        ), 
        'required': False
    },
    {
        "name": "timeout",
        "type": "integer", 
        "description": (
            "超时秒数（默认 60，最大 300）。\n"
            "安装大型包（numpy/torch）或长时间计算时设为 300。"
        ), 
        "default": 60,
        "minimum": 1,
        "maximum": 300,
        'required': False
    }]

    # params 是llm返回的工具调用部分的参数的json字符串，需要解析成json格式的字典
    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"exec:call exec tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        # exec 执行系统命令的3个参数
        language = params_data.get('language',"bash")
        code = params_data.get('code',"") # 获取 code 参数
        timeout = params_data.get('timeout',60) # 获取 timeout 参数

        # 从 kwargs 中获取 sandbox
        sandbox: Sandbox = kwargs.get('sandbox', None)
        if not sandbox:
            return f"[ERROR] exec: The 'sandbox' parameter was not provided."

        if language == "bash":
            result = sandbox.exec(code, timeout)
            return self._format_exec_result(result)
 
        elif language == "python":
            # textwrap.dedent：去掉 LLM 生成代码可能携带的多余公共缩进(就是将行公共空格去掉，保留相对缩进关系)
            clean_code = textwrap.dedent(code)
 
            # 写入沙箱内固定临时路径，容器内路径为 /workspace/scripts/_agent_tmp.py
            tmp_rel = "scripts/_agent_tmp.py"
            try:
                await sandbox.write_file(tmp_rel, clean_code)
            except Exception as e:
                return f"[ERROR] 写入临时脚本失败: {e}"

            # exec执行命令和 python的区别就是python脚本是先写到python临时文件中去，然后执行这个文件获取结果
            result = sandbox.exec(f"python /workspace/{tmp_rel}", timeout)
            return self._format_exec_result(result)
 
        else:
            return (
                f"[ERROR] 不支持的 language 值: {language!r}，"
                "只允许 'python' 或 'bash'"
            )

    def _format_exec_result(self, result: dict) -> str:
        """
        将 sandbox.exec() 返回的 dict 格式化为 LLM 可读字符串。
    
        成功（exit_code=0）：返回 stdout，若无输出给出提示
        失败（exit_code!=0）：返回 exit_code + stdout + stderr，便于 LLM 分析原因
        """
        exit_code = result.get("exit_code", -1)
        stdout    = (result.get("stdout") or "").strip()
        stderr    = (result.get("stderr") or "").strip()
    
        if exit_code == 0:
            return stdout if stdout else "(执行成功，无 stdout 输出)"
    
        parts = [f"[EXEC FAILED] exit_code={exit_code}"]
        if stdout:
            parts.append(f"--- stdout ---\n{stdout}")
        if stderr:
            parts.append(f"--- stderr ---\n{stderr}")
        return "\n".join(parts)

# read_file 读取文件
@agent_register_tool('read_file')
class read_file(BaseTool):
    description = (
        "读取 /workspace 目录中的文件内容。\n\n"
        "【何时使用】\n"
        "- 读取上一步 exec 生成的输出文件\n"
        "- 读取保存的中间状态（JSON、文本等）\n"
        "- 验证生成文件的内容是否正确\n\n"
        "【路径规范】\n"
        "- 传入相对于 /workspace 的路径，不要包含 /workspace/ 前缀\n"
        "- 正确：'output/result.json'\n"
        "- 错误：'/workspace/output/result.json'\n\n"
        "【注意】超过 50KB 的文件建议先用 exec 做预处理（head/tail/grep）再读取"
    )
    parameters = [{
        "name": "path",
        "type": "string", 
        "description": (
            "相对于 /workspace 的文件路径（不含 /workspace/ 前缀）。\n"
            "示例：'output/report.md'、'input/data.csv'、'state.json'"
        ), 
        'required': True
    },{
        "name": "encoding",
        "type": "string", 
        "description": (
            "文件读取编码方式。\n"
            "- 'utf-8'（默认）：强制文本模式\n"
            "- 'auto'：自动检测，文本用 utf-8，二进制用 base64\n"
            "- 'base64'：强制二进制模式（图片、PDF 等）"
        ), 
        "enum": ["utf-8", "base64", "auto"],
        'required': False
    }]

    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"read_file:call read_file tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        path = params_data.get('path',"")
        encoding = params_data.get('encoding',"utf-8") # 获取 encoding 参数

        # 从 kwargs 中获取 sandbox
        sandbox: Sandbox = kwargs.get('sandbox', None)
        if not sandbox:
            return f"[ERROR] read_file: The 'sandbox' parameter was not provided."

        content = await sandbox.read_file(path, encoding)
        return content # 返回正常的读取文件结果

# read_file 读取文件
@agent_register_tool('write_file')
class write_file(BaseTool):
    description = (
        "向 /workspace 目录写入文本文件。\n\n"
        "【何时使用】\n"
        "- 保存任务规划和中间状态（防止长对话中信息丢失）\n"
        "- 将 LLM 生成的 Python 脚本写入文件，再用 exec 执行\n"
        "- 保存最终文本产物（Markdown 报告、配置文件、JSON 数据等）\n\n"
        "【路径规范】同 read_file，传相对路径，不含 /workspace/ 前缀\n\n"
        "【最佳实践】\n"
        "- 任务开始时将规划写入 state.json\n"
        "- 生成二进制文件（图表/PDF/Excel）请用 exec 在容器内完成，不要用此工具"
    )
    parameters = [{
        "name": "path",
        "type": "string", 
        "description": (
            "相对于 /workspace 的目标路径（不含 /workspace/ 前缀）。\n"
            "目录不存在时会自动创建。\n"
            "示例：'scripts/process.py'、'output/summary.md'、'state.json'"
        ), 
        'required': True
    },{
        "name": "content",
        "type": "string", 
        "description": "要写入的文本内容。", 
        'required': True
    },{
        "name": "mode",
        "type": "string", 
        "description":  (
            "写入模式：\n"
            "- 'overwrite'（默认）：覆盖写入，文件不存在则创建\n"
            "- 'append'：追加到文件末尾，适合写日志"
        ), 
        "default": "overwrite",
        "enum": ["overwrite", "append"],
        'required': False
    }]

    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"write_file:call write_file tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        path = params_data.get('path',"")
        content = params_data.get('content',"")
        mode = params_data.get('mode',"overwrite")

        # 从 kwargs 中获取 sandbox
        sandbox: Sandbox = kwargs.get('sandbox', None)
        if not sandbox:
            return f"[ERROR] write_file: The 'sandbox' parameter was not provided." 

        if mode == "append":
            # sandbox.write_file 本身是覆盖写，手动实现 append
            try:
                existing = await sandbox.read_file(path)
                content  = existing + content
            except FileNotFoundError:
                pass  # 文件不存在，等同于新建

        await sandbox.write_file(path, content)
        return f"[OK] 已写入 /workspace/{path}（{len(content)} 字符）"

# ------------------------------------------------------------------
# 4. web_search
#    通过 exec 在容器内运行 duckduckgo-search（免费无 Key）
#    第一次调用会自动 pip install，后续命中缓存极快
# ------------------------------------------------------------------
@agent_register_tool('web_search')
class web_search(BaseTool):
    description = (
        "搜索互联网获取实时信息。底层在沙箱容器内执行，结果结构化返回。\n\n"
        "【何时使用】\n"
        "- 查找训练截止后的最新内容（新版本、时事、价格等）\n"
        "- 查找库的安装方法、API 文档链接\n"
        "- 验证事实，避免幻觉\n\n"
        "【不需要使用的情况】\n"
        "- 数学计算、代码编写（用 exec）\n"
        "- 已知的基础知识\n\n"
        "【搜索技巧】查询词保持简洁（3~8词），英文搜索技术问题效果更好"
    )
    parameters = [{
        "name": "query",
        "type": "string", 
        "description": (
            "搜索关键词，简洁精准。\n"
            "好：'pandas read parquet python 2024'\n"
            "差：'我想知道如何用最新版本的 pandas 库来读取 parquet 格式的文件'"
        ),
        'required': True
    },{
        "name": "num_results",
        "type": "integer", 
        "description": "返回结果数量，默认 5，最多 10。", 
        "default": 5,
        "minimum": 1,
        "maximum": 10,
        'required': False
    }
    ]

    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"web_search:call web_search tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        query = params_data.get('query',"") # 查询输入的内容
        num_results = params_data.get('num_results',5) # 返回的结果数量
        
        # 从 kwargs 中获取 sandbox
        sandbox: Sandbox = kwargs.get('sandbox', None)
        if not sandbox:
            return f"[ERROR] web_search: The 'sandbox' parameter was not provided." 
            
        # Step 1: 幂等安装（已安装则立即退出） 如果安装了则退出
        install = sandbox.exec(
            "pip show ddgs > /dev/null 2>&1 || "
            "pip install ddgs -q -i https://pypi.tuna.tsinghua.edu.cn/simple/"
        )
        if install["exit_code"] != 0:
            return f"[ERROR] 安装 ddgs 失败:\n{install['stderr']}"

        # Step 2: 执行搜索，将结果序列化为 JSON 输出
        search_script = textwrap.dedent(f"""\
            import json, sys
            from ddgs import DDGS
 
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text({json5.dumps(query)}, max_results={num_results}))
                results = [
                    {{"title": r.get("title",""), "url": r.get("href",""), "body": r.get("body","")}}
                    for r in raw
                ]
                print(json.dumps(results, ensure_ascii=False))
            except Exception as e:
                print(json.dumps({{"error": str(e)}}), file=sys.stderr)
                sys.exit(1)
        """)
        await sandbox.write_file("scripts/_web_search_tmp.py", search_script)
        result = sandbox.exec("python /workspace/scripts/_web_search_tmp.py")

        if result["exit_code"] != 0:
            return f"[ERROR] 搜索执行失败:\n{result['stderr']}"
 
        try:
            data = json5.loads(result["stdout"].strip())
        except (ValueError, TypeError) as e:
            return f"[ERROR] 搜索结果解析失败，原始输出:\n{result['stdout']}"
 
        if isinstance(data, dict) and "error" in data:
            return f"[ERROR] 搜索返回错误: {data['error']}"
 
        # 格式化为结构化文本返回给 LLM
        lines = [f"搜索 [{query}] 共 {len(data)} 条结果：\n"]
        for i, item in enumerate(data, 1):
            lines.append(f"[{i}] {item['title']}")
            lines.append(f"    URL : {item['url']}")
            lines.append(f"    摘要: {item['body'][:500]}")
            lines.append("")
        return "\n".join(lines)

@agent_register_tool('web_fetch')
class web_fetch(BaseTool):
    description = (
        "获取指定 URL 页面的完整文本内容。通常配合 web_search 使用：\n"
        "先搜索拿到 URL，再 fetch 读取完整文档。\n\n"
        "【何时使用】\n"
        "- web_search 摘要不够，需要完整文档内容\n"
        "- 读取官方文档、README、技术博客\n\n"
        "【注意】\n"
        "- 只能访问公开页面，不支持登录墙\n"
        "- 返回提取后的纯文本\n"
        "- 不要用于下载二进制文件，请在 exec 中用 curl/wget"
    )
    parameters = [{
        "name": "url",
        "type": "string", 
        "description": (
            "完整 URL，必须包含协议头。\n"
            "示例：'https://docs.python.org/3/library/pathlib.html'"
        ),
        'required': True
    },{
        "name": "max_chars",
        "type": "integer", 
        "description": "返回最大字符数，默认 8000，避免超出上下文窗口。",
        "default": 8000,
        "minimum": 500,
        "maximum": 30000,
        'required': False
    }
    ]

    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"web_fetch:call web_fetch tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        url = params_data.get('url','')
        max_chars = params_data.get('max_chars',8000) # 获取 max_chars 参数

        # 从 kwargs 中获取 sandbox
        sandbox: Sandbox = kwargs.get('sandbox', None)
        if not sandbox:
            return f"[ERROR] web_fetch: The 'sandbox' parameter was not provided." 

        fetch_script = textwrap.dedent(f"""\
            import urllib.request, html, re, sys
 
            url       = {json5.dumps(url)}
            max_chars = {max_chars}
 
            req = urllib.request.Request(
                url,
                headers={{"User-Agent": "Mozilla/5.0 (AgentBot/1.0)"}},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    charset = "utf-8"
                    ct = resp.headers.get_content_charset()
                    if ct:
                        charset = ct
                    raw = resp.read().decode(charset, errors="replace")
            except Exception as e:
                print(f"[FETCH ERROR] {{e}}", file=sys.stderr)
                sys.exit(1)
 
            # 去除 <style> <script> 后提取纯文本
            text = re.sub(r"<style[^>]*>.*?</style>",  " ", raw,  flags=re.S|re.I)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S|re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(text)
            text = re.sub(r"\\s{{3,}}", "\\n\\n", text).strip()
 
            out = text[:max_chars]
            print(out)
            if len(text) > max_chars:
                print(f"\\n...[已截断，页面共 {{len(text)}} 字符]")
        """)
 
        await sandbox.write_file("scripts/_web_fetch_tmp.py", fetch_script)
        result = sandbox.exec("python /workspace/scripts/_web_fetch_tmp.py")
 
        if result["exit_code"] != 0:
            return f"[ERROR] 获取页面失败 ({url}):\n{result['stderr']}"
 
        return result["stdout"] or "(页面内容为空)"

@agent_register_tool('ask_user')
class ask_user(BaseTool):
    description = (
        "向用户提问以获取关键信息。\n\n"
        "【严格限制，仅在以下情况使用】\n"
        "- 即将执行不可逆操作（覆盖数据、发送邮件、扣款）且存在歧义\n"
        "- 需要用户提供的凭证（API Key、密码）\n"
        "- 任务目标存在根本性歧义，两种理解会产生完全不同的结果\n\n"
        "【以下情况不应使用，应自主判断】\n"
        "- 技术方案选择 → 自己选最合理的\n"
        "- 输出格式偏好 → 给出合理默认值\n"
        "- 可以通过搜索解决的信息缺失\n"
        "- 对进度的确认 → 继续执行即可\n\n"
        "【规范】一次最多问 2 个问题，提供选项比开放问题更易回答"
    )
    parameters = [{
        "name": "question",
        "type": "string", 
        "description": (
            "向用户提出的问题，简洁明确。\n"
            "好：'output/ 目录已有 234 个文件，继续将全部覆盖，是否继续？(yes/no)'\n"
            "差：'您希望我如何处理已存在的文件呢？'"
        ),
        'required': True
    },{
        "name": "options",
        "type": "array", 
        "items": {"type": "string"},
        "description":  (
            "可选：提供选项列表。\n"
            "示例：['A. 覆盖所有文件', 'B. 跳过已存在的文件', 'C. 取消']"
        ), 
        'required': False
    }
    ]

    async def call(self, params: str, **kwargs) -> str:
        logger.info(f"ask_user:call ask_user tool!")
        # 只解析一次 JSON（提高性能，避免重复解析）
        try:
            params_data = json5.loads(params)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid JSON in params: {params}, error: {e}")
            return ''

        question = params_data.get('question','')
        options = params_data.get('options',[]) # 获取 extract_mode 参数
 
        # 默认实现：命令行 input()（开发调试用）
        print("\n" + "="*60)
        print("[Agent 需要您的输入]")
        print(f"\n{question}")
        if options:
            for opt in options:
                print(f"  {opt}")
        answer = input("\n您的回答：").strip()
        print("="*60 + "\n")
        return answer
