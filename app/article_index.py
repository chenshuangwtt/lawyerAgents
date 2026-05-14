"""
法条条号内存索引模块。

构建 law_name → article_num → [chunks] 的嵌套字典，
实现 O(1) 前后条查找，避免 ChromaDB 子串查询限制。
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document


def build_article_index(chunks: List[Document]) -> Dict[str, Dict[int, List[Document]]]:
    """
    从 chunk 列表构建条号索引。

    结构：{ law_name: { article_num: [chunk1, chunk2, ...], ... }, ... }

    Args:
        chunks: 已包含 article_numbers_int 元数据的 chunk 列表。

    Returns:
        条号索引字典。
    """
    index: Dict[str, Dict[int, List[Document]]] = {}

    for chunk in chunks:
        law_name = chunk.metadata.get("source", "")
        int_str = chunk.metadata.get("article_numbers_int", "")
        if not law_name or not int_str:
            continue

        for num_str in int_str.split(","):
            try:
                num = int(num_str)
            except ValueError:
                continue

            if law_name not in index:
                index[law_name] = {}
            if num not in index[law_name]:
                index[law_name][num] = []
            index[law_name][num].append(chunk)

    total_articles = sum(len(v) for v in index.values())
    print(f"条号索引构建完成：{len(index)} 部法律，{total_articles} 个条号")
    return index


def get_adjacent_articles(
    article_index: Dict[str, Dict[int, List[Document]]],
    law_name: str,
    article_nums: List[int],
    n: int = 1,
    exclude_contents: Optional[set] = None,
) -> List[Document]:
    """
    获取指定条号的前后 n 条相关 chunk。

    Args:
        article_index: 条号索引。
        law_name: 法律名称。
        article_nums: 当前已有的条号列表。
        n: 前后扩展范围。
        exclude_contents: 需要排除的 chunk 内容（去重用）。

    Returns:
        前后条的 chunk 列表（不含已有 chunk）。
    """
    if not article_nums or law_name not in article_index:
        return []

    law_index = article_index[law_name]
    target_nums = set()
    for num in article_nums:
        for offset in range(1, n + 1):
            target_nums.add(num - offset)
            target_nums.add(num + offset)
    # 排除已有条号
    target_nums -= set(article_nums)

    if exclude_contents is None:
        exclude_contents = set()

    result = []
    seen_contents = set()
    for num in sorted(target_nums):
        for chunk in law_index.get(num, []):
            content_key = chunk.page_content[:200]
            if content_key not in exclude_contents and content_key not in seen_contents:
                seen_contents.add(content_key)
                result.append(chunk)

    return result
