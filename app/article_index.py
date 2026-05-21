"""
法条条号内存索引模块。

构建 law_name → article_num → [chunks] 的嵌套字典，
实现 O(1) 前后条查找，避免 ChromaDB 子串查询限制。
"""

import logging
from typing import List, Dict, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _parse_article_numbers_str(text: str) -> List[int]:
    """从"第X条、第Y条"格式的字符串中提取整数条号列表。"""
    from app.loader import ARTICLE_PATTERN, _chinese_num_to_int
    nums = []
    for m in ARTICLE_PATTERN.finditer(text):
        base = _chinese_num_to_int(m.group(1))
        if base <= 0:
            continue
        sub = m.group(2)
        if sub:
            nums.append(base * 10 + _chinese_num_to_int(sub))
        else:
            nums.append(base)
    return nums


def build_article_index(chunks: List[Document]) -> Dict[str, Dict[int, List[Document]]]:
    """
    从 chunk 列表构建条号索引。

    结构：{ law_name: { article_num: [chunk1, chunk2, ...], ... }, ... }

    条号来源优先级：
      1. article 元数据（主条号，最可靠）
      2. article_numbers 元数据（完整条号字符串，含合并后的结果）
      3. article_numbers_int 元数据（从 chunk 正文提取，可能不全）

    Args:
        chunks: chunk 列表。

    Returns:
        条号索引字典。
    """
    index: Dict[str, Dict[int, List[Document]]] = {}

    for chunk in chunks:
        law_name = chunk.metadata.get("source", "")
        if not law_name:
            continue

        nums = set()

        # 1. 从 article 元数据取主条号（最可靠，每个 chunk 都有）
        article_meta = chunk.metadata.get("article", "")
        if article_meta:
            nums.update(_parse_article_numbers_str(article_meta))

        # 2. 从 article_numbers 元数据取完整条号（合并后仍有效）
        article_numbers_str = chunk.metadata.get("article_numbers", "")
        if article_numbers_str:
            nums.update(_parse_article_numbers_str(article_numbers_str))

        # 3. 从 article_numbers_int 元数据补充（旧字段，兜底）
        int_str = chunk.metadata.get("article_numbers_int", "")
        if int_str:
            for num_str in int_str.split(","):
                try:
                    nums.add(int(num_str))
                except ValueError:
                    continue

        if not nums:
            continue

        # 展开条号范围：如 [271, 275] → [271, 272, 273, 274, 275]
        expanded = _expand_article_range(sorted(nums))

        if law_name not in index:
            index[law_name] = {}
        for num in expanded:
            if num not in index[law_name]:
                index[law_name][num] = []
            index[law_name][num].append(chunk)

    total_articles = sum(len(v) for v in index.values())
    logger.info("条号索引构建完成：%d 部法律，%d 个条号", len(index), total_articles)
    return index


def _expand_article_range(nums: List[int]) -> List[int]:
    """展开条号范围列表。如 [271, 275] → [271, 272, 273, 274, 275]。"""
    if len(nums) < 2:
        return nums

    result = set()
    sorted_nums = sorted(nums)

    for i in range(len(sorted_nums)):
        result.add(sorted_nums[i])
        # 与下一个数的差在 1~10 之间时，认为是连续范围需要展开
        if i + 1 < len(sorted_nums):
            gap = sorted_nums[i + 1] - sorted_nums[i]
            if 1 < gap <= 10:
                for j in range(sorted_nums[i] + 1, sorted_nums[i + 1]):
                    result.add(j)

    return sorted(result)


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
