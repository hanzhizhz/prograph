"""
实体数据类
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """实体数据类"""
    text: str
    type: str  # PERSON, LOCATION, ORGANIZATION, DATE, NUMBER, CUSTOM
    custom_type: Optional[str] = None  # 仅对 CUSTOM 类型：自定义类型名称
    reason: Optional[str] = None  # 仅对 CUSTOM 类型：价值说明
    prop_idx: int = 0  # 来源命题索引
    doc_id: str = ""  # 来源文档 ID

    def __repr__(self):
        if self.type == "CUSTOM":
            return f"Entity({self.text}, type=CUSTOM, custom_type={self.custom_type})"
        return f"Entity({self.text}, type={self.type})"
