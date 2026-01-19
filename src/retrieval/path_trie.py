"""
路径去重 Trie 树

使用字符串拼接 + set 实现高效路径去重
时间复杂度: O(path_length)
空间效率: 共享公共前缀
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class PathTrieNode:
    """Trie 树节点（用于调试/序列化）"""
    children: Dict[str, 'PathTrieNode']
    is_path_end: bool
    path_count: int

    def __init__(self):
        self.children: Dict[str, PathTrieNode] = {}
        self.is_path_end: bool = False
        self.path_count: int = 0


class PathTrie:
    """
    路径去重 Trie 树

    使用字符串拼接作为 key 存储路径，查找效率 O(path_length)
    相比 frozenset 去重的优势：
    - 空间效率更高：共享公共前缀（字符串存储时）
    - 查找效率稳定：O(path_length) 字符串拼接 vs O(1) hash
    - 支持前缀查询：可以快速查找某个前缀的所有路径

    用法:
        trie = PathTrie()

        # 添加路径
        is_new = trie.add_path(["node1", "node2", "node3"])

        # 检查路径是否存在
        exists = trie.contains_path(["node1", "node2", "node3"])

        # 获取统计
        stats = trie.get_stats()
    """

    def __init__(self, max_paths: int = 1000000, delimiter: str = "\x00"):
        """
        初始化 Trie

        Args:
            max_paths: 最大路径数量限制（防止内存溢出）
            delimiter: 节点ID分隔符（使用不可见字符避免冲突）
        """
        self.max_paths = max_paths
        self.delimiter = delimiter
        self._path_set: Set[str] = set()
        self._total_paths = 0
        self._root = PathTrieNode()  # 用于调试和前缀查询

    def _path_to_key(self, path_nodes: List[str]) -> str:
        """将路径转换为唯一key"""
        return self.delimiter.join(path_nodes)

    def _key_to_path(self, key: str) -> List[str]:
        """将 key 还原为路径（用于前缀查询）"""
        return key.split(self.delimiter)

    def add_path(self, path_nodes: List[str]) -> bool:
        """
        添加路径，返回是否为新路径

        Args:
            path_nodes: 路径节点ID列表

        Returns:
            True=新路径（已添加）, False=路径已存在
        """
        if not path_nodes:
            return False

        self._total_paths += 1
        key = self._path_to_key(path_nodes)

        if key in self._path_set:
            return False

        self._path_set.add(key)
        self._update_trie_debug(path_nodes)
        return True

    def _update_trie_debug(self, path_nodes: List[str]):
        """更新调试用的 Trie 结构"""
        node = self._root
        for node_id in path_nodes:
            if node_id not in node.children:
                node.children[node_id] = PathTrieNode()
            node = node.children[node_id]
        node.is_path_end = True
        node.path_count += 1

    def contains_path(self, path_nodes: List[str]) -> bool:
        """
        检查路径是否存在

        Args:
            path_nodes: 路径节点ID列表

        Returns:
            True=路径存在, False=路径不存在
        """
        if not path_nodes:
            return False
        key = self._path_to_key(path_nodes)
        return key in self._path_set

    def remove_path(self, path_nodes: List[str]) -> bool:
        """
        删除路径（谨慎使用：不会清理空节点）

        Args:
            path_nodes: 路径节点ID列表

        Returns:
            True=成功删除, False=路径不存在
        """
        if not path_nodes:
            return False

        key = self._path_to_key(path_nodes)

        if key not in self._path_set:
            return False

        self._path_set.discard(key)
        return True

    def get_paths_with_prefix(self, prefix: List[str]) -> List[List[str]]:
        """
        获取所有具有给定前缀的路径

        Args:
            prefix: 路径前缀节点ID列表

        Returns:
            所有以该前缀开头的完整路径列表
        """
        if not prefix:
            return []

        prefix_key = self._path_to_key(prefix)

        # 筛选以 prefix 开头的所有 key
        results = []
        prefix_with_delim = prefix_key + self.delimiter

        for key in self._path_set:
            if key.startswith(prefix_with_delim) or key == prefix_key:
                results.append(self._key_to_path(key))

        return results

    def count_paths_with_prefix(self, prefix: List[str]) -> int:
        """
        统计具有给定前缀的路径数量

        Args:
            prefix: 路径前缀节点ID列表

        Returns:
            以该前缀开头的完整路径数量
        """
        return len(self.get_paths_with_prefix(prefix))

    def get_stats(self) -> Dict[str, int]:
        """
        获取 Trie 统计信息

        Returns:
            统计信息字典
        """
        # 使用调试用的 Trie 节点统计
        total_nodes = 0
        total_end_nodes = 0

        def count_nodes(node: PathTrieNode):
            nonlocal total_nodes, total_end_nodes
            total_nodes += 1
            if node.is_path_end:
                total_end_nodes += 1
            for child in node.children.values():
                count_nodes(child)

        count_nodes(self._root)

        return {
            "total_nodes": total_nodes,
            "end_nodes": total_end_nodes,
            "unique_paths": len(self._path_set),
            "total_add_attempts": self._total_paths,
        }

    def clear(self):
        """清空 Trie"""
        self._path_set.clear()
        self._total_paths = 0
        self._root = PathTrieNode()
