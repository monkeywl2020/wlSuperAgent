# -*- coding: utf-8 -*-
# pylint: disable=R0901
"""The content blocks of messages"""

from typing import Literal, List
from typing_extensions import TypedDict, Required

# TypedDict 是什么？
# TypedDict 是 Python 提供的一种类型注解方式，用于声明一个字典（dict）的结构和类型约束。
# 传统方式：无法约束字典结构 user = {"name": "Alice", "age": 30}  # 随便什么结构都行,不能指定key对应的value类型
# TypedDict 可以约束字典的结构和类型，例如：
# class User(TypedDict):
#    name: str
#    age: int
# user: User = {"name": "Alice", "age": 30}  # ✓ 类型检查通过
# user: User = {"name": "Alice", "age": "30"}  # ✗ 类型错误
class TextBlock(TypedDict, total=False):#  total=True 表示字典所有字段都是必填的 ,False 表示字段是可选的
    """The text block."""
    # required 表示该字段是必填的,这个是配合 total=False 使用的
    type: Required[Literal["text"]]#  Literal["text"] 相当于 enum，但用于类型检查，只允许是 "text"
    """The type of the block"""
    text: str
    """The text content"""

# Literal 用于限制变量的值必须是指定的值之一。 Literal["thinking"]  表示这个字段的值只能是 thinking
class ThinkingBlock(TypedDict, total=False):
    """The thinking block."""

    type: Required[Literal["thinking"]]
    """The type of the block"""
    thinking: str


class Base64Source(TypedDict, total=False):
    """The base64 source"""

    type: Required[Literal["base64"]]
    """The type of the src, must be `base64`"""

    media_type: Required[str]
    """The media type of the data, e.g. `image/jpeg` or `audio/mpeg`"""

    data: Required[str]
    """The base64 data, in format of RFC 2397"""


class URLSource(TypedDict, total=False):
    """The URL source"""

    type: Required[Literal["url"]]
    """The type of the src, must be `url`"""

    url: Required[str]
    """The URL of the image or audio"""


class ImageBlock(TypedDict, total=False):
    """The image block"""

    type: Required[Literal["image"]]
    """The type of the block"""

    source: Required[Base64Source | URLSource]
    """The src of the image"""


class AudioBlock(TypedDict, total=False):
    """The audio block"""

    type: Required[Literal["audio"]]
    """The type of the block"""

    source: Required[Base64Source | URLSource]
    """The src of the audio"""


class VideoBlock(TypedDict, total=False):
    """The video block"""

    type: Required[Literal["video"]]
    """The type of the block"""

    source: Required[Base64Source | URLSource]
    """The src of the audio"""

class ToolUseBlock(TypedDict, total=False):
    """The tool use block."""

    type: Required[Literal["tool_use"]]
    """The type of the block, must be `tool_use`"""
    id: Required[str]
    """The identity of the tool call"""
    name: Required[str]
    """The name of the tool"""
    input: Required[dict[str, object]]
    """The input of the tool"""
    raw_input: str
    """The raw string input of the tool from the model API"""


class ToolResultBlock(TypedDict, total=False):
    """The tool result block."""

    type: Required[Literal["tool_result"]]
    """The type of the block"""
    id: Required[str]
    """The identity of the tool call result"""
    output: Required[
        str | List[TextBlock | ImageBlock | AudioBlock | VideoBlock]
    ]
    """The output of the tool function"""
    name: Required[str]
    """The name of the tool function"""

# 用户自定义块
class CustomBlock(TypedDict, total=False):
    """The custom block"""

    type: Required[Literal["custom"]]
    """The type of the block"""

    id: Required[str]
    """The identity of the tool call result"""

    output: Required[dict[str, object] | str]
    """The output of the tool function"""
    
    name: Required[str]
    """The name of the tool function"""

# The content block
ContentBlock = (
    ToolUseBlock
    | ToolResultBlock
    | TextBlock
    | ThinkingBlock
    | ImageBlock
    | AudioBlock
    | VideoBlock
    | CustomBlock
)

ContentBlockTypes = Literal[
    "text",
    "thinking",
    "tool_use",
    "tool_result",
    "image",
    "audio",
    "video",
    "custom",
]
