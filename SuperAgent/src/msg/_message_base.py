# -*- coding: utf-8 -*-
"""The message class in agentscope."""
from datetime import datetime
from typing import Literal, List, overload, Sequence, Optional

import shortuuid

from ._message_block import (
    TextBlock,
    ToolUseBlock,
    ImageBlock,
    AudioBlock,
    ContentBlock,
    VideoBlock,
    ToolResultBlock,
    ContentBlockTypes,
)
from ._json import JSONSerializableObject


class Msg:
    """The message class in agentscope."""

    # Sequence 是 typing 模块中的一个抽象基类，表示有序的序列类型。
    # 例如：list, tuple, str 等，也就是  ContentBlock 可以 List[ContentBlock] 也可以是 tuple[ContentBlock]

    def __init__(
        self,
        content: str | Sequence[ContentBlock],
        role: Literal["user", "assistant", "system", "tool","function"],
        name: str | None = None, # 消息发送者名称, 改成可选参数
        metadata: dict[str, JSONSerializableObject] | None = None,
        timestamp: str | None = None,
        invocation_id: str | None = None,
        tool_calls: List[dict] | None = None, # 新增：工具调用信息，这个信息得用于后面处理工具调用的结果。这个内容模型返回dict 不需要解析
        extra: dict | None = None, # 新增：额外信息，例如：函数调用的结果。
        compressed: bool = False, # 新增：是否已压缩标记（用于记忆系统）
        summary: str | None = None, # 新增：压缩后的摘要（用于记忆系统）
        file_path: str | None = None, # 新增：文件引用路径（压缩后）
        tool_call_id: str | None = None, # 新增：工具调用ID（用于tool角色）
    ) -> None:
        """Initialize the Msg object.

        Args:
            name: str | None = None, # 消息发送者名称, 改成可选参数
                The name of the message sender.
            content (`str | list[ContentBlock]`):
                The content of the message.
            role (`Literal["user", "assistant", "system"]`):
                The role of the message sender.
            metadata (`dict[str, JSONSerializableObject] | None`, optional):
                The metadata of the message, e.g. structured output.
            timestamp (`str | None`, optional):
                The created timestamp of the message. If not given, the
                timestamp will be set automatically.
            invocation_id (`str | None`, optional):
                The related API invocation id, if any. This is useful for
                tracking the message in the context of an API call.
            compressed: 是否已压缩
            summary: 压缩后的摘要
            file_path: 文件引用路径（压缩后）
            tool_call_id: 工具调用ID（用于tool角色）
        """

        self.name = name

        assert isinstance(
            content,
            (list, str),
        ), "The content must be a string or a list of content blocks."

        self.content = content

        # 新增 tool 和 function 角色, 这个是为了方便处理工具调用和函数调用的消息。function是模型的工具调用。 tool是用户调用的工具的响应消息。
        assert role in ["user", "assistant", "system", "tool","function"]
        self.role = role

        self.metadata = metadata or {}

        self.id = shortuuid.uuid() # 生成一个唯一的ID，这个是message id，根据这个标识可以判断是否是同一个消息
        self.timestamp = (
            timestamp
            or datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f",
            )[:-3]
        )
        self.invocation_id = invocation_id

        # 新增：工具调用 和 extra 信息
        self.tool_calls = tool_calls or None
        self.extra = extra or None
        
        # 新增：压缩相关属性（用于记忆系统）
        self.compressed = compressed
        self.summary = summary
        self.file_path = file_path
        
        # 工具调用ID（用于tool角色）
        self.tool_call_id = tool_call_id


    def to_dict(self) -> dict: # 转换为字典格式
        """Convert the message into JSON dict data."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls,
            "extra": self.extra,
            "compressed": self.compressed,
            "summary": self.summary,
            "file_path": self.file_path,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, json_data: dict) -> "Msg": # 从字典格式加载消息对象
        """Load a message object from the given JSON data."""
        new_obj = cls(
            content=json_data["content"],
            role=json_data["role"],
            name=json_data.get("name", None), # 从字典中获取name, 如果没有, 则设为None
            metadata=json_data.get("metadata", None),
            timestamp=json_data.get("timestamp", None),
            invocation_id=json_data.get("invocation_id", None),
            tool_calls=json_data.get("tool_calls", None),
            extra=json_data.get("extra", None),
            compressed=json_data.get("compressed", False),
            summary=json_data.get("summary", None),
            file_path=json_data.get("file_path", None),
            tool_call_id=json_data.get("tool_call_id", None),
        )

        new_obj.id = json_data.get("id", new_obj.id)# 如果有id，就使用这个id，如果没有，就生成一个唯一的id
        return new_obj

    def has_content_blocks(
        self,
        block_type: Literal[
            "text",
            "tool_use",
            "tool_result",
            "image",
            "audio",
            "video",
        ]
        | None = None,
    ) -> bool:
        """Check if the message has content blocks of the given type.

        Args:
            block_type (Literal["text", "tool_use", "tool_result", "image", \
            "audio", "video"] | None, defaults to None):
                The type of the block to be checked. If `None`, it will
                check if there are any content blocks.
        """
        return len(self.get_content_blocks(block_type)) > 0

    def get_text_content(self, separator: str = "\n") -> str | None:
        """Get the pure text blocks from the message content.

        Args:
            separator (`str`, defaults to `\n`):
                The separator to use when concatenating multiple text blocks.
                Defaults to newline character.

        Returns:
            `str | None`:
                The concatenated text content, or `None` if there is no text
                content.
        """
        if isinstance(self.content, str):
            return self.content

        gathered_text = []
        for block in self.content:
            if block.get("type") == "text":
                gathered_text.append(block["text"])

        if gathered_text:
            return separator.join(gathered_text)

        return None

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["text"],
    ) -> Sequence[TextBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_use"],
    ) -> Sequence[ToolUseBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_result"],
    ) -> Sequence[ToolResultBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["image"],
    ) -> Sequence[ImageBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["audio"],
    ) -> Sequence[AudioBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["video"],
    ) -> Sequence[VideoBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: None = None,
    ) -> Sequence[ContentBlock]:
        ...

    def get_content_blocks(
        self,
        block_type: ContentBlockTypes | List[ContentBlockTypes] | None = None,
    ) -> Sequence[ContentBlock]:
        """Get the content in block format. If the content is a string,
        it will be converted to a text block.

        Args:
            block_type (`ContentBlockTypes | List[ContentBlockTypes] | None`, \
            optional):
                The type of the block to be extracted. If `None`, all blocks
                will be returned.

        Returns:
            `List[ContentBlock]`:
                The content blocks.
        """
        blocks = []
        if isinstance(self.content, str):
            blocks.append(
                TextBlock(type="text", text=self.content),
            )
        else:
            # blocks 是content这个list ，两种类型，str和Sequence[ContentBlock]，这里是后者
            blocks = self.content or []

        # 从 content 这个list中 将 type属性 为 block_type 的block 提取出来
        if isinstance(block_type, str):
            blocks = [_ for _ in blocks if _["type"] == block_type]

        elif isinstance(block_type, list):# 可以过滤多种，例如["text", "tool_use"]
            blocks = [_ for _ in blocks if _["type"] in block_type]

        return blocks

    def __repr__(self) -> str:
        """Get the string representation of the message."""
        return (
            f"Msg(id='{self.id}', "
            f"name='{self.name}', "
            f"content={repr(self.content)}, "
            f"role='{self.role}', "
            f"metadata={repr(self.metadata)}, "
            f"timestamp='{self.timestamp}', "
            f"invocation_id='{self.invocation_id}', "
            f"tool_calls={repr(self.tool_calls)}, "
            f"extra={repr(self.extra)}, "
            f"compressed={self.compressed}, "
            f"summary={repr(self.summary)}, "
            f"file_path={repr(self.file_path)}, "
            f"tool_call_id={repr(self.tool_call_id)}"
        )
