# -*- coding: utf-8 -*-
"""The message module in agentscope."""

from ._message_block import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    CustomBlock,
    Base64Source,
    URLSource,
    ContentBlockTypes,
)
from ._message_base import Msg


__all__ = [
    "TextBlock",
    "ThinkingBlock",
    "Base64Source",
    "URLSource",
    "ImageBlock",
    "AudioBlock",
    "VideoBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    "CustomBlock",
    "ContentBlockTypes",
    "Msg",
]
