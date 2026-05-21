"""
引用语义溯源模块：向量库验证引用准确性 + 检测遗漏引用。

与原有 _verify_citations（仅检查条号存在性）的区别：
- 向量库查询引用法条的原文，判断与 answer 描述的语义相关性
- 给每条引用标记 confidence（high/medium/low）
- 检测检索结果中高相关但 answer 未引用的法条（suggested）
"""

import logging
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
import jieba
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from langchain_core.documents import Document

from app.rag_chain import ARTICLE_PATTERN, _chinese_num_to_int


# 置信度阈值（基于 jieba 分词后的关键词重叠率）
HIGH_THRESHOLD = 0.3
MEDIUM_THRESHOLD = 0.1


def _compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    文本相似度（基于 jieba 分词的关键词重叠率）。
    不依赖向量库，用于快速判断引用相关性。
    """
    if not text_a or not text_b:
        return 0.0
    # jieba 分词，取长度 >= 2 的中文词
    words_a = {w for w in jieba.cut(text_a) if len(w) >= 2 and re.match(r'^[一-鿿]+$', w)}
    words_b = {w for w in jieba.cut(text_b) if len(w) >= 2 and re.match(r'^[一-鿿]+$', w)}
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


class CitationVerifier:
    """引用语义验证器。"""

    def __init__(self, article_index: Dict):
        self.article_index = article_index

    def verify_citations(
        self,
        sources: List[Dict[str, str]],
        answer: str,
    ) -> List[Dict[str, str]]:
        """
        验证每条引用的语义相关性。

        对每个引用：
        1. 从 article_index 查找法条原文
        2. 用关键词重叠率判断 answer 中对该条的描述是否准确
        3. 标记 confidence: high / medium / low
        """
        if not self.article_index or not sources:
            return sources

        verified = []
        for src in sources:
            label = src["source"]
            parts = label.split(" ", 1)

            if len(parts) < 2:
                verified.append({**src, "confidence": ""})
                continue

            law_name = parts[0]
            articles_str = parts[1]

            if law_name not in self.article_index:
                verified.append({**src, "confidence": "medium"})
                continue

            # 提取引用的所有条号
            article_list = re.split(r"[、,]", articles_str)
            article_list = [a.strip() for a in article_list if a.strip()]

            max_confidence = "low"
            for art in article_list:
                clean_art = re.sub(r"\s*等\d+条$", "", art)
                art_match = ARTICLE_PATTERN.search(clean_art)
                if not art_match:
                    continue

                art_num = _chinese_num_to_int(art_match.group(1))
                if art_num <= 0:
                    continue

                # 查找该条原文
                chunks = self.article_index.get(law_name, {}).get(art_num, [])
                if not chunks:
                    continue

                # 用原文与 answer 做相似度比较
                law_text = chunks[0].page_content[:500]
                similarity = _compute_text_similarity(law_text, answer)

                if similarity >= HIGH_THRESHOLD:
                    confidence = "high"
                elif similarity >= MEDIUM_THRESHOLD:
                    confidence = "medium"
                else:
                    confidence = "low"

                # 取最高置信度
                priority = {"high": 3, "medium": 2, "low": 1}
                if priority.get(confidence, 0) > priority.get(max_confidence, 0):
                    max_confidence = confidence

            verified.append({**src, "confidence": max_confidence})

        return verified

    def detect_missing_citations(
        self,
        answer: str,
        reranked_docs: List[Document],
    ) -> List[Dict[str, str]]:
        """
        检测 answer 未引用但检索中高相关的法条。

        从 reranked_docs 中找出 answer 未提到的法条，
        判断其与 answer 的语义相关性，相关性高的标记为 suggested。
        """
        if not answer or not reranked_docs:
            return []

        # 从 answer 中提取已引用的法律名 + 条号
        cited_articles = set()
        for match in ARTICLE_PATTERN.finditer(answer):
            cited_articles.add(match.group(0))

        # 提取已引用的法律名
        cited_laws = set()
        for src_text in re.findall(r'《([^》]+)》', answer):
            cited_laws.add(src_text)

        missing = []
        seen_sources = set()

        for doc in reranked_docs:
            source = doc.metadata.get("source", "")
            if not source or source in seen_sources:
                continue

            # 检查这个法律是否已被引用
            if source in cited_laws:
                continue

            # 检查该 doc 的条号是否在 answer 中出现
            int_str = doc.metadata.get("article_numbers_int", "")
            article_nums = []
            if int_str:
                try:
                    article_nums = [int(x) for x in int_str.split(",") if x.strip()]
                except ValueError:
                    pass

            # 如果该 doc 的所有条号都未被引用
            doc_cited = False
            for num in article_nums:
                # 将数字转为中文条号格式检查
                for cited in cited_articles:
                    if str(num) in cited or _num_to_chinese(num) in cited:
                        doc_cited = True
                        break
                if doc_cited:
                    break

            if doc_cited:
                continue

            # 用关键词重叠率判断相关性
            similarity = _compute_text_similarity(doc.page_content[:300], answer)
            if similarity >= MEDIUM_THRESHOLD:
                seen_sources.add(source)
                # 提取条号信息
                art_str = doc.metadata.get("article_numbers", "")
                if art_str:
                    source_label = f"{source} {art_str}"
                else:
                    source_label = source

                missing.append({
                    "source": source_label,
                    "content": doc.page_content[:200],
                    "full_content": doc.page_content,
                })

        if missing:
            logger.info("[引用追踪] 发现 %d 个可能遗漏的关键引用", len(missing))

        return missing


def _num_to_chinese(num: int) -> str:
    """数字转中文（用于匹配条号格式）。"""
    chinese_digits = "零一二三四五六七八九"
    if num < 10:
        return chinese_digits[num]
    if num < 20:
        return "十" + (chinese_digits[num - 10] if num > 10 else "")
    if num < 100:
        tens = num // 10
        ones = num % 10
        result = ("一二三四五六七八九")[tens - 1] + "十"
        if ones:
            result += chinese_digits[ones]
        return result
    return str(num)
