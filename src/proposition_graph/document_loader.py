"""
文档加载器
支持 HotpotQA, 2WikiMultihopQA, MuSiQue 数据集
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import json


@dataclass
class Document:
    """文档数据类"""
    doc_id: str
    title: str
    content: List[str]  # 句子列表

    def __repr__(self):
        return f"Document(doc_id={self.doc_id}, title={self.title[:30]}..., sentences={len(self.content)})"


class DocumentLoader:
    """
    文档加载器

    支持三种数据集格式的自动检测和加载
    """

    def load(self, file_path: str) -> List[Document]:
        """
        加载文档文件

        Args:
            file_path: 文档文件路径

        Returns:
            文档列表
        """
        path = Path(file_path)

        if not path.exists():
            # 尝试从 raw_dataset 目录加载
            if "dataset/" in file_path:
                raw_path = file_path.replace("dataset/", "raw_dataset/")
                path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(f"文档文件不存在: {file_path}")

        # 检测数据集类型并加载
        if "HotpotQA" in str(path) or "hotpotqa" in str(path).lower():
            return self._load_hotpotqa(path)
        elif "2WikiMultihopQA" in str(path) or "2wiki" in str(path).lower():
            return self._load_2wiki(path)
        elif "MuSiQue" in str(path) or "musique" in str(path).lower():
            return self._load_musique(path)
        else:
            # 默认使用 HotpotQA 格式
            return self._load_hotpotqa(path)

    def _load_hotpotqa(self, path: Path, prefix: str = "hotpot") -> List[Document]:
        """
        加载 HotpotQA 格式文档

        格式: [{"title": "...", "content": ["sentence1", "sentence2", ...]}, ...]
        
        Args:
            path: 文档文件路径
            prefix: 文档ID前缀，默认为 "hotpot"
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for idx, item in enumerate(data):
            doc = Document(
                doc_id=f"{prefix}_{idx}",
                title=item.get("title", ""),
                content=item.get("content", [])
            )
            documents.append(doc)

        return documents

    def _load_2wiki(self, path: Path) -> List[Document]:
        """
        加载 2WikiMultihopQA 格式文档

        格式与 HotpotQA 类似，使用 "2wiki" 作为文档ID前缀
        """
        return self._load_hotpotqa(path, prefix="2wiki")

    def _load_musique(self, path: Path) -> List[Document]:
        """
        加载 MuSiQue 格式文档

        格式与 HotpotQA 类似，使用 "musique" 作为文档ID前缀
        """
        return self._load_hotpotqa(path, prefix="musique")

    def load_from_jsonl(self, file_path: str) -> List[Document]:
        """
        从 JSONL 文件加载文档

        Args:
            file_path: JSONL 文件路径

        Returns:
            文档列表
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文档文件不存在: {file_path}")

        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    doc = Document(
                        doc_id=data.get("doc_id", f"doc_{idx}"),
                        title=data.get("title", ""),
                        content=data.get("content", [])
                    )
                    documents.append(doc)

        return documents
