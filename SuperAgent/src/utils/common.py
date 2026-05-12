# -*- coding: utf-8 -*-
""" Tools for agentscope """
import base64
import datetime
import json
import os.path
import secrets
import string
import socket
import hashlib
import random
import re
import time
import sys
from typing import Any, Literal, List, Optional, Tuple
import shutil
import traceback
import json5

import urllib.parse
import psutil
import requests
import asyncio
from collections.abc import Coroutine

from src.msg import Msg as Message

from loguru import logger


def get_basename_from_url(path_or_url: str) -> str:
    if re.match(r'^[A-Za-z]:\\', path_or_url):
        # "C:\\a\\b\\c" -> "C:/a/b/c"
        path_or_url = path_or_url.replace('\\', '/')

    # "/mnt/a/b/c" -> "c"
    # "https://github.com/here?k=v" -> "here"
    # "https://github.com/" -> ""
    basename = urllib.parse.urlparse(path_or_url).path
    basename = os.path.basename(basename)
    basename = urllib.parse.unquote(basename)
    basename = basename.strip()

    # "https://github.com/" -> "" -> "github.com"
    if not basename:
        basename = [x.strip() for x in path_or_url.split('/') if x.strip()][-1]

    return basename

def sanitize_windows_file_path(file_path: str) -> str:
    # For Linux and macOS.
    if os.path.exists(file_path):
        return file_path

    # For native Windows, drop the leading '/' in '/C:/'
    win_path = file_path
    if win_path.startswith('/'):
        win_path = win_path[1:]
    if os.path.exists(win_path):
        return win_path

    # For Windows + WSL.
    if re.match(r'^[A-Za-z]:/', win_path):
        wsl_path = f'/mnt/{win_path[0].lower()}/{win_path[3:]}'
        if os.path.exists(wsl_path):
            return wsl_path

    # For native Windows, replace / with \.
    win_path = win_path.replace('/', '\\')
    if os.path.exists(win_path):
        return win_path

    return file_path

def sanitize_chrome_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path

    # Dealing with "file:///...":
    new_path = urllib.parse.urlparse(file_path)
    new_path = urllib.parse.unquote(new_path.path)
    new_path = sanitize_windows_file_path(new_path)
    if os.path.exists(new_path):
        return new_path

    return sanitize_windows_file_path(file_path)

def is_http_url(path_or_url: str) -> bool:
    if path_or_url.startswith('https://') or path_or_url.startswith('http://'):
        return True
    return False

# 从url下载 文件到本地
def save_url_to_local_work_dir(url: str, save_dir: str, save_filename: str = '') -> str:
    if not save_filename:
        save_filename = get_basename_from_url(url)
    new_path = os.path.join(save_dir, save_filename)
    if os.path.exists(new_path):
        os.remove(new_path)
    logger.info(f'Downloading {url} to {new_path}...')
    start_time = time.time()
    if not is_http_url(url):
        url = sanitize_chrome_file_path(url)
        shutil.copy(url, new_path)
    else:
        headers = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(new_path, 'wb') as file:
                file.write(response.content)
        else:
            raise ValueError('Can not download this file. Please check your network or the file link.')
    end_time = time.time()
    logger.info(f'Finished downloading {url} to {new_path}. Time spent: {end_time - start_time} seconds.')
    return new_path

# 打印3层错误堆栈，并且将堆栈记录 到日志中去打出来
def print_traceback(is_error: bool = True):
    tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
    if is_error:
        logger.error(tb)
    else:
        logger.warning(tb)

def is_image(path_or_url: str) -> bool:
    filename = get_basename_from_url(path_or_url).lower()
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        if filename.endswith(ext):
            return True
    return False

CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')

def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))

def json_loads(text: str) -> dict:
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json5.loads(text)
        except ValueError:
            raise json_err

def print_traceback(is_error: bool = True):
    tb = ''.join(traceback.format_exception(*sys.exc_info(), limit=3))
    if is_error:
        logger.error(tb)
    else:
        logger.warning(tb)

########################################################################################################################

def _get_timestamp(
    format_: str = "%Y-%m-%d %H:%M:%S",
    time: datetime.datetime = None,
) -> str:
    """Get current timestamp."""
    if time is None:
        return datetime.datetime.now().strftime(format_)
    else:
        return time.strftime(format_)


def to_openai_dict(item: dict) -> dict:
    """Convert `Msg` to `dict` for OpenAI API."""
    clean_dict = {}

    if "name" in item:
        clean_dict["name"] = item["name"]

    if "role" in item:
        clean_dict["role"] = item["role"]
    else:
        clean_dict["role"] = "assistant"

    if "content" in item:
        clean_dict["content"] = _convert_to_str(item["content"])
    else:
        raise ValueError("The content of the message is missing.")

    return clean_dict


def to_dialog_str(item: dict) -> str:
    """Convert a dict into string prompt style."""
    speaker = item.get("name", None) or item.get("role", None)
    content = item.get("content", None)

    if content is None:
        return str(item)

    if speaker is None:
        return content
    else:
        return f"{speaker}: {content}"


def find_available_port() -> int:
    """Get an unoccupied socket port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def check_port(port: Optional[int] = None) -> int:
    """Check if the port is available.

    Args:
        port (`int`):
            the port number being checked.

    Returns:
        `int`: the port number that passed the check. If the port is found
        to be occupied, an available port number will be automatically
        returned.
    """
    if port is None:
        new_port = find_available_port()
        return new_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", port)) == 0:
            new_port = find_available_port()
            return new_port
    return port


def _guess_type_by_extension(
    url: str,
) -> Literal["image", "audio", "video", "file"]:
    """Guess the type of the file by its extension."""
    extension = url.split(".")[-1].lower()

    if extension in [
        "bmp",
        "dib",
        "icns",
        "ico",
        "jfif",
        "jpe",
        "jpeg",
        "jpg",
        "j2c",
        "j2k",
        "jp2",
        "jpc",
        "jpf",
        "jpx",
        "apng",
        "png",
        "bw",
        "rgb",
        "rgba",
        "sgi",
        "tif",
        "tiff",
        "webp",
    ]:
        return "image"
    elif extension in [
        "amr",
        "wav",
        "3gp",
        "3gpp",
        "aac",
        "mp3",
        "flac",
        "ogg",
    ]:
        return "audio"
    elif extension in [
        "mp4",
        "webm",
        "mkv",
        "flv",
        "avi",
        "mov",
        "wmv",
        "rmvb",
    ]:
        return "video"
    else:
        return "file"


def _to_openai_image_url(url: str) -> str:
    """Convert an image url to openai format. If the given url is a local
    file, it will be converted to base64 format. Otherwise, it will be
    returned directly.

    Args:
        url (`str`):
            The local or public url of the image.
    """
    # See https://platform.openai.com/docs/guides/vision for details of
    # support image extensions.
    support_image_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    )

    parsed_url = urllib.parse(url)

    lower_url = url.lower()

    # Web url
    if parsed_url.scheme != "":
        if any(lower_url.endswith(_) for _ in support_image_extensions):
            return url

    # Check if it is a local file
    elif os.path.exists(url) and os.path.isfile(url):
        if any(lower_url.endswith(_) for _ in support_image_extensions):
            with open(url, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode(
                    "utf-8",
                )
            extension = parsed_url.path.lower().split(".")[-1]
            mime_type = f"image/{extension}"
            return f"data:{mime_type};base64,{base64_image}"

    raise TypeError(f"{url} should be end with {support_image_extensions}.")


def _download_file(url: str, path_file: str, max_retries: int = 3) -> bool:
    """Download file from the given url and save it to the given path.

    Args:
        url (`str`):
            The url of the file.
        path_file (`str`):
            The path to save the file.
        max_retries (`int`, defaults to `3`)
            The maximum number of retries when fail to download the file.
    """
    for n_retry in range(1, max_retries + 1):
        response = requests.get(url, stream=True)
        if response.status_code == requests.codes.ok:
            with open(path_file, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return True
        else:
            raise RuntimeError(
                f"Failed to download file from {url} (status code: "
                f"{response.status_code}). Retry {n_retry}/{max_retries}.",
            )
    return False


def _generate_random_code(
    length: int = 6,
    uppercase: bool = True,
    lowercase: bool = True,
    digits: bool = True,
) -> str:
    """Get random code."""
    characters = ""
    if uppercase:
        characters += string.ascii_uppercase
    if lowercase:
        characters += string.ascii_lowercase
    if digits:
        characters += string.digits
    return "".join(secrets.choice(characters) for i in range(length))


def generate_id_from_seed(seed: str, length: int = 8) -> str:
    """Generate random id from seed str.

    Args:
        seed (`str`): seed string.
        length (`int`): generated id length.
    """
    hasher = hashlib.sha256()
    hasher.update(seed.encode("utf-8"))
    hash_digest = hasher.hexdigest()

    random.seed(hash_digest)
    id_chars = [
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length)
    ]
    return "".join(id_chars)


def is_web_accessible(url: str) -> bool:
    """Whether the url is accessible from the Web.

    Args:
        url (`str`):
            The url to check.

    Note:
        This function is not perfect, it only checks if the URL starts with
        common web protocols, e.g., http, https, ftp, oss.
    """
    parsed_url = urllib.parse(url)
    return parsed_url.scheme in ["http", "https", "ftp", "oss"]


def _is_json_serializable(obj: Any) -> bool:
    """Check if the given object is json serializable."""
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def _convert_to_str(content: Any) -> str:
    """Convert the content to string.

    Note:
        For prompt engineering, simply calling `str(content)` or
        `json.dumps(content)` is not enough.

        - For `str(content)`, if `content` is a dictionary, it will turn double
        quotes to single quotes. When this string is fed into prompt, the LLMs
        may learn to use single quotes instead of double quotes (which
        cannot be loaded by `json.loads` API).

        - For `json.dumps(content)`, if `content` is a string, it will add
        double quotes to the string. LLMs may learn to use double quotes to
        wrap strings, which leads to the same issue as `str(content)`.

        To avoid these issues, we use this function to safely convert the
        content to a string used in prompt.

    Args:
        content (`Any`):
            The content to be converted.

    Returns:
        `str`: The converted string.
    """

    if isinstance(content, str):
        return content
    elif isinstance(content, (dict, list, int, float, bool, tuple)):
        return json.dumps(content, ensure_ascii=False)
    else:
        return str(content)


def reform_dialogue(input_msgs: list[dict]) -> list[dict]:
    """record dialog history as a list of strings"""
    messages = []

    dialogue = []
    for i, unit in enumerate(input_msgs):
        if i == 0 and unit["role"] == "system":
            # system prompt
            messages.append(
                {
                    "role": unit["role"],
                    "content": _convert_to_str(unit["content"]),
                },
            )
        else:
            # Merge all messages into a dialogue history prompt
            dialogue.append(
                f"{unit['name']}: {_convert_to_str(unit['content'])}",
            )

    dialogue_history = "\n".join(dialogue)

    user_content_template = "## Dialogue History\n{dialogue_history}"

    messages.append(
        {
            "role": "user",
            "content": user_content_template.format(
                dialogue_history=dialogue_history,
            ),
        },
    )

    return messages


def _join_str_with_comma_and(elements: List[str]) -> str:
    """Return the JSON string with comma, and use " and " between the last two
    elements."""

    if len(elements) == 0:
        return ""
    elif len(elements) == 1:
        return elements[0]
    elif len(elements) == 2:
        return " and ".join(elements)
    else:
        return ", ".join(elements[:-1]) + f", and {elements[-1]}"


class ImportErrorReporter:
    """Used as a placeholder for missing packages.
    When called, an ImportError will be raised, prompting the user to install
    the specified extras requirement.
    """

    def __init__(self, error: ImportError, extras_require: str = None) -> None:
        """Init the ImportErrorReporter.

        Args:
            error (`ImportError`): the original ImportError.
            extras_require (`str`): the extras requirement.
        """
        self.error = error
        self.extras_require = extras_require

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._raise_import_error()

    def __getattr__(self, name: str) -> Any:
        return self._raise_import_error()

    def __getitem__(self, __key: Any) -> Any:
        return self._raise_import_error()

    def _raise_import_error(self) -> Any:
        """Raise the ImportError"""
        err_msg = f"ImportError occorred: [{self.error.msg}]."
        if self.extras_require is not None:
            err_msg += (
                f" Please install [{self.extras_require}] version"
                " of agentscope."
            )
        raise ImportError(err_msg)


def _hash_string(
    data: str,
    hash_method: Literal["sha256", "md5", "sha1"],
) -> str:
    """Hash the string data."""
    hash_func = getattr(hashlib, hash_method)()
    hash_func.update(data.encode())
    return hash_func.hexdigest()


def _get_process_creation_time() -> datetime.datetime:
    """Get the creation time of the process."""
    pid = os.getpid()
    # Find the process by pid
    current_process = psutil.Process(pid)
    # Obtain the process creation time
    create_time = current_process.create_time()
    # Change the timestamp to a readable format
    return datetime.datetime.fromtimestamp(create_time)


def _is_process_alive(
    pid: int,
    create_time_str: str,
    create_time_format: str = "%Y-%m-%d %H:%M:%S",
    tolerance_seconds: int = 10,
) -> bool:
    """Check if the process is alive by comparing the actual creation time of
    the process with the given creation time.

    Args:
        pid (`int`):
            The process id.
        create_time_str (`str`):
            The given creation time string.
        create_time_format (`str`, defaults to `"%Y-%m-%d %H:%M:%S"`):
            The format of the given creation time string.
        tolerance_seconds (`int`, defaults to `10`):
            The tolerance seconds for comparing the actual creation time with
            the given creation time.

    Returns:
        `bool`: True if the process is alive, False otherwise.
    """
    try:
        # Try to create a process object by pid
        proc = psutil.Process(pid)
        # Obtain the actual creation time of the process
        actual_create_time_timestamp = proc.create_time()

        # Convert the given creation time string to a datetime object
        given_create_time_datetime = datetime.datetime.strptime(
            create_time_str,
            create_time_format,
        )

        # Calculate the time difference between the actual creation time and
        time_difference = abs(
            actual_create_time_timestamp
            - given_create_time_datetime.timestamp(),
        )

        # Compare the actual creation time with the given creation time
        if time_difference <= tolerance_seconds:
            return True

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # If the process is not found, access is denied, or the process is a
        # zombie process, return False
        return False

    return False


# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(
    *coroutines: Coroutine,
    max_coroutines: int | None = None,
) -> list[Any]:
    default_max_coroutines = 5 # 默认不超过5个并发
    limit = max_coroutines if max_coroutines is not None else default_max_coroutines
    semaphore = asyncio.Semaphore(limit)

    async def _wrap_coroutine(coroutine):
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*(_wrap_coroutine(coroutine) for coroutine in coroutines))