# -*- coding: utf-8 -*-
"""The JSON related types"""

from typing import Union

JSONPrimitive = Union[
    str,
    int,
    float,
    bool,
    None,
]# 原始类型（字符串、数字、布尔、空） 这个union结构，可以这些类型

JSONSerializableObject = Union[
    JSONPrimitive, # 原始类型（字符串、数字、布尔、空）
    list["JSONSerializableObject"], #列表类型，列表中的元素必须是 JSONSerializableObject 类型（嵌套）
    dict[
        str,
        "JSONSerializableObject",
    ], #  这表示一个字典  Key（键）：必须是 str 类型 , Value（值）：必须是 JSONSerializableObject 类型（嵌套）
]
