
import subprocess
import os
import json
import time
from pathlib import Path
import shutil
import base64
from typing import Union, Optional, Dict, Any, List
from loguru import logger

from src.settings import sandbox_settings

CONTAINER_NAME = sandbox_settings.container_name if sandbox_settings else "agent-sandbox"
IMAGE_NAME = sandbox_settings.image_name if sandbox_settings else "python:3.12-slim"
CONTAINER_WORKSPACE = sandbox_settings.container_workspace if sandbox_settings else "/workspace"
MEMORY_LIMIT = sandbox_settings.memory_limit if sandbox_settings else "1024m"
CPUS = sandbox_settings.cpus if sandbox_settings else 1
NETWORK = sandbox_settings.network if sandbox_settings else "bridge"
WORKSPACE_SUBDIRS = sandbox_settings.workspace_subdirs if sandbox_settings else ["input", "output", "scripts"]

HOST_WORKSPACE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../example/workspace")
)


class Sandbox:
    def __init__(self):
        self._start()

    def _start(self):
        """启动 sandbox 容器（如果已存在就先清理）"""
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True
        )

        logger.info(f"🚀 Start the sandbox container:{CONTAINER_NAME}")
        logger.info(f"   Image: {IMAGE_NAME}")
        logger.info(f"   Mount: {HOST_WORKSPACE}  →  {CONTAINER_WORKSPACE}")

        result = subprocess.run([
            "docker", "run",
            "--name",    CONTAINER_NAME,
            "--detach",
            "--tty",
            "--volume",  f"{HOST_WORKSPACE}:{CONTAINER_WORKSPACE}",
            "--memory",  MEMORY_LIMIT,
            "--cpus",    str(CPUS),
            "--network", NETWORK,
            IMAGE_NAME,
            "sleep", "infinity"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"容器启动失败：{result.stderr}")

        logger.info(f"✅ 容器已启动\n")

    def exec(self, command: str, timeout: int = 60, workdir: str = CONTAINER_WORKSPACE) -> dict:
        """
        在容器内执行 shell 命令。
        这就是 agent 的 exec tool 的底层实现。
        """
        result = subprocess.run([
            "docker", "exec",
            "--workdir", workdir,
            CONTAINER_NAME,
            "bash", "-c", command
        ], capture_output=True, text=True, timeout=timeout)

        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout.strip(),
            "stderr":    result.stderr.strip(),
        }

    async def write_file(self, relative_path: str, content: str, encoding: str = "utf-8"):
        """
        从宿主机写文件到 workspace。
        因为 workspace 是挂载目录，容器内立刻可见。

        relative_path: 相对于 workspace 根目录的文件路径
        content: 要写入的内容（字符串）
        encoding: 编码方式，可选 'utf-8'、'base64'、'auto'
            'utf-8' : 文本模式写入（默认）
            'base64': 将 content 视为 base64 编码字符串，解码后以二进制写入
            'auto'  : 目前等同于 'utf-8'
        """
        full_path = os.path.join(HOST_WORKSPACE, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        if encoding == "base64":
            binary_data = base64.b64decode(content)
            with open(full_path, "wb") as f:
                f.write(binary_data)
            logger.info(f"==> 📝 Writing base64 decoded binary to file: {relative_path}")
        else:  # "utf-8" 或 "auto"
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"==> 📝 Writing text file (utf-8): {relative_path}")

    async def read_file(self, relative_path: str, encoding: str = "utf-8") -> str:
        """
        从 workspace 读取文件（宿主机直接读）。

        relative_path: 相对于 workspace 根目录的文件路径
        encoding: 编码方式，可选 'utf-8'、'base64'、'auto'
                        'utf-8' : 以 UTF‑8 文本读取，返回字符串（默认）
                        'base64': 以二进制读取，返回 base64 编码字符串
                        'auto'  : 自动检测，若 UTF‑8 解码成功返回文本，否则返回 base64 字符串
        return: 文件内容字符串（文本或 base64）
        """
        full_path = os.path.join(HOST_WORKSPACE, relative_path)

        if encoding == "base64":
            with open(full_path, "rb") as f:
                binary_data = f.read()
            return base64.b64encode(binary_data).decode("ascii")

        elif encoding == "auto":
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(full_path, "rb") as f:
                    binary_data = f.read()
                return base64.b64encode(binary_data).decode("ascii")

        else:  # "utf-8"
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()

    def cleanup(self):
        """销毁容器"""
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        logger.info("🗑  Container destroyed.")


def show_result(label: str, result: dict):
    status = "✅" if result["exit_code"] == 0 else "❌"
    logger.info(f"{status} {label}")
    if result["stdout"]:
        for line in result["stdout"].split("\n"):
            logger.info(f"   {line}")
    if result["stderr"]:
        logger.info(f"   [stderr] {result['stderr']}")


def show_env_info(sb: Sandbox):
    logger.info("="*50)
    logger.info("📋 演示 1：sandbox 内部环境")
    logger.info("="*50)

    show_result("Python 版本",    sb.exec("python --version"))
    show_result("操作系统",       sb.exec("cat /etc/os-release | head -3"))
    show_result("当前用户",       sb.exec("whoami"))
    show_result("工作目录内容",   sb.exec("ls -la /workspace"))
    show_result("可用磁盘空间",   sb.exec("df -h /workspace"))
    show_result("内存限制",       sb.exec("cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '（cgroup v1）'"))
    show_result("网络连通性",     sb.exec("python -c \"import urllib.request,json; d=json.load(urllib.request.urlopen('https://pypi.org/pypi/requests/json', timeout=5)); print('pypi 可访问，requests 最新版：', d['info']['version'])\""))


def init_workspace_and_sandbox():
    sb = Sandbox()
    show_env_info(sb)
    return sb


def copy_folder_advanced(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    move: bool = False,
    recursive: bool = False,
    overwrite: bool = False,
    extensions: Optional[List[str]] = None
) -> int:
    """
    高级文件夹拷贝/移动函数

    功能:
        - 支持拷贝或移动文件
        - 支持递归搜索子目录
        - 支持按文件后缀过滤（如 ['.md', '.txt']）
        - 支持覆盖控制
        - 返回成功处理的文件数量

    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        move: 是否移动源文件（默认拷贝，移动会删除源文件）
        recursive: 是否递归遍历子目录
        overwrite: 目标文件存在时是否覆盖
        extensions: 文件后缀过滤列表，例如 ['.md', '.py']，None表示不过滤

    返回:
        int: 成功处理的文件数量
    """

    source = Path(source_dir)
    target = Path(target_dir)
    logger.debug(f"copy_folder_advanced ---- source:{source}")
    logger.debug(f"copy_folder_advanced ---- target:{target}")

    if not source.exists():
        logger.error(f"Error: Source directory does not exist:{source}")
        return 0

    if not source.is_dir():
        logger.error(f"Error: Source path is not a directory:{source}")
        return 0

    target.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0
    skipped = 0

    search_pattern = "**/*" if recursive else "*"

    for file_path in source.glob(search_pattern):

        if not file_path.is_file():
            continue

        if extensions is not None:
            if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                continue

        # 保持子目录结构：获取文件相对于源目录的路径
        relative_path = file_path.relative_to(source)
        dest_path = target / relative_path

        # 确保目标文件的父目录存在
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and not overwrite:
            logger.warning(f"Skip (Already exists): {relative_path}")
            skipped += 1
            continue

        try:
            if move:
                shutil.move(str(file_path), str(dest_path))
                action = "move"
            else:
                shutil.copy2(str(file_path), str(dest_path))
                action = "copy"

            count += 1
            logger.info(f"{action} success: {relative_path}")

        except PermissionError:
            errors += 1
            logger.error(f"Permission error: {file_path.name}")

        except Exception as e:
            errors += 1
            logger.error(f"Error: {file_path.name}: {type(e).__name__} - {e}")

    logger.info(f"\nSuccess! Total: {count} | Skipped: {skipped} | Failed: {errors}")

    return count


def demo_file_rw(sb: Sandbox):
    logger.info("\n" + "="*50)
    logger.info("📁 演示 2：文件读写（workspace 双向共享）")
    logger.info("="*50)

    sb.write_file("input/data.json", json.dumps({
        "task":    "计算斐波那契数列前 10 项",
        "version": "1.0"
    }, ensure_ascii=False, indent=2))

    show_result("容器内读取宿主机写入的文件",
                sb.exec("cat /workspace/input/data.json"))

    sb.exec("echo '容器写入的内容' > /workspace/output/from_container.txt")
    content = sb.read_file("output/from_container.txt")
    logger.info(f"\n✅ 宿主机读取容器写入的文件：\n   {content.strip()}")


def demo_pip_and_run(sb: Sandbox):
    logger.info("\n" + "="*50)
    logger.info("📦 演示 3：pip 安装第三方包并运行")
    logger.info("="*50)

    logger.info("\n   安装 rich 库中...")
    result = sb.exec("pip install rich -i https://pypi.tuna.tsinghua.edu.cn/simple/ --quiet")
    show_result("pip install rich", result)

    sb.write_file("scripts/fibonacci.py", '''
from rich.table import Table
from rich.console import Console

def fibonacci(n):
    a, b = 0, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq

console = Console()
table = Table(title="斐波那契数列")
table.add_column("索引", style="cyan")
table.add_column("值",   style="magenta")

for i, v in enumerate(fibonacci(10)):
    table.add_row(str(i), str(v))

console.print(table)
import json, pathlib
out = {"fibonacci": fibonacci(10)}
pathlib.Path("/workspace/output/fibonacci.json").write_text(
    json.dumps(out, indent=2)
)
print("\\n✅ 结果已写入 /workspace/output/fibonacci.json")
''')

    show_result("运行 fibonacci.py", sb.exec("python /workspace/scripts/fibonacci.py"))

    result_json = sb.read_file("output/fibonacci.json")
    logger.info(f"\n✅ 宿主机读取输出文件：\n   {result_json.strip()}")


def demo_retry(sb: Sandbox):
    logger.info("\n" + "="*50)
    logger.info("🔄 演示 4：反复试错（沙盒内改 bug）")
    logger.info("="*50)

    buggy_code = '''
# bug1: 变量名拼错
nubmers = [3, 1, 4, 1, 5, 9, 2, 6]

# bug2: 函数名写错
def srot(arr):
    return sorted(arr)

# bug3: 调用了拼错的变量
result = srot(numbers)
print("排序结果:", result)
'''
    sb.write_file("scripts/buggy.py", buggy_code)

    logger.info("\n   第 1 次运行（有 bug）：")
    r = sb.exec("python /workspace/scripts/buggy.py")
    show_result("运行结果", r)

    fixed_code = '''
# 修复：变量名统一
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

def sort_list(arr):
    return sorted(arr)

result = sort_list(numbers)
print("排序结果:", result)
'''
    sb.write_file("scripts/buggy.py", fixed_code)

    logger.info("\n   第 2 次运行（已修复）：")
    r = sb.exec("python /workspace/scripts/buggy.py")
    show_result("运行结果", r)


def demo_boundary(sb: Sandbox):
    logger.info("\n" + "="*50)
    logger.info("🔒 演示 5：沙盒权限边界")
    logger.info("="*50)

    show_result("workspace 内写文件（允许）",
                sb.exec("echo 'test' > /workspace/output/test.txt && echo 'OK'"))

    show_result("写入 /tmp（允许）",
                sb.exec("echo 'tmp ok' > /tmp/test.txt && echo 'OK'"))

    show_result("列出容器根目录（可见但业务层应限制路径）",
                sb.exec("ls /"))

    show_result("容器内进程列表（与宿主机隔离）",
                sb.exec("ps aux"))


if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   Docker Sandbox + Workspace 演示    ║")
    logger.info("╚══════════════════════════════════════╝")

    sb = Sandbox()

    try:
        show_env_info(sb)
        demo_file_rw(sb)
        demo_pip_and_run(sb)
        demo_retry(sb)
        demo_boundary(sb)

        logger.info("\n" + "="*50)
        logger.info("🎉 所有演示完成！")
        logger.info(f"   workspace 目录：{HOST_WORKSPACE}")
        logger.info("   可以查看里面的 output/ 目录看生成的文件")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"\n❌ 出错：{e}")
    finally:
        sb.cleanup()
