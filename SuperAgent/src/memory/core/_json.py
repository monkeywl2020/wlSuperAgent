# -*- coding: utf-8 -*-
"""
JSON 序列化模块

定义可 JSON 序列化的对象类型
"""

# 导入标准库
from typing import Any  # 类型注解
import json  # JSON序列化


class JSONSerializableObject:
    """可 JSON 序列化的对象基类"""

    def to_json(self) -> Any:
        """
        转换为 JSON 兼容的对象

        Returns:
            JSON 兼容的对象（dict, list, str, int, float, bool, None）
        """
        # 默认实现：转换为字典
        return self.__dict__

    def to_json_string(self) -> str:
        """
        转换为 JSON 字符串

        Returns:
            JSON 格式的字符串
        """
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_data: dict) -> "JSONSerializableObject":
        """
        从 JSON 数据创建对象

        Args:
            json_data: JSON 字典数据

        Returns:
            对象实例
        """
        obj = cls()
        for key, value in json_data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj
