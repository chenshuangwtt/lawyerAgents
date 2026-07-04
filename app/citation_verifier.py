"""
引用验证模块：向量语义相似度 + LLM 准确性验证 + 遗漏检测。

三层验证：
  1. 条号存在性（article_index 查找）
  2. 语义相关性（向量余弦相似度）
  3. 描述准确性（LLM 逐条验证，可选）
"""

import logging
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

from langchain_core.documents import Document

from app.core import PARA_PATTERN, invoke_with_timeout
from app.article_utils import ARTICLE_PATTERN, chinese_num_to_int
from app.source_item import SourceItem, normalize_sources


# 置信度阈值（向量余弦相似度）
HIGH_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.30

# 遗漏检测阈值
MISSING_THRESHOLD = 0.30

_CITATION_ARTICLE_PATTERN = r"第([零一二三四五六七八九十百千万0-9]+)条"
_BRACKET_CITATION_RE = re.compile(
    rf"《([^》]+)》\s*({_CITATION_ARTICLE_PATTERN})"
)
_GENERIC_LAW_CITATION_RE = re.compile(
    rf"(?:根据|依据|依照|按照|适用)?\s*"
    rf"([\u4e00-\u9fff]{{2,40}}?(?:法|典|条例|规定|解释))\s*"
    rf"({_CITATION_ARTICLE_PATTERN})"
)


def _cosine_similarity(vec_a, vec_b) -> float:
    """计算两个向量的余弦相似度。"""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def extract_answer_citations(answer: str, sources: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Extract explicit law-name + article-number citations from answer text."""
    text = answer or ""
    known_laws = _known_law_names(sources or [])
    seen: set[tuple[str, int]] = set()
    citations: List[Dict[str, Any]] = []

    def add(law_name: str, article_no: str) -> None:
        law_name = _clean_law_name(law_name)
        article_no = _normalize_article_label(article_no)
        article_num = _article_number(article_no)
        if not law_name or article_num <= 0:
            return
        key = (_law_key(law_name), article_num)
        if key in seen:
            return
        seen.add(key)
        citations.append({
            "citation": f"{law_name}{article_no}",
            "law_name": law_name,
            "article_no": article_no,
            "article_num": article_num,
        })

    for law_name, article_no, _num in _BRACKET_CITATION_RE.findall(text):
        add(law_name, article_no)

    for law_name, article_no, _num in _GENERIC_LAW_CITATION_RE.findall(text):
        clean_law = _clean_law_name(law_name)
        if _looks_like_known_law(clean_law, known_laws) or _looks_like_legal_source(clean_law):
            add(clean_law, article_no)

    return citations


def _known_law_names(sources: List[Dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for source in sources:
        for value in (source.get("law_name"), source.get("source"), source.get("title")):
            if not value:
                continue
            law_name = _clean_law_name(str(value).split()[0])
            if law_name:
                names.add(_law_key(law_name))
    return names


def _clean_law_name(law_name: str) -> str:
    law_name = re.sub(r"\s+", "", str(law_name or ""))
    law_name = law_name.strip("《》：:，,。；;（）()")
    law_name = re.sub(r"^(根据|依据|依照|按照|适用)", "", law_name)
    return law_name


def _law_key(law_name: str) -> str:
    return _clean_law_name(law_name).replace("中华人民共和国", "")


def _normalize_article_label(article_no: str) -> str:
    article_no = re.sub(r"\s+", "", str(article_no or ""))
    match = ARTICLE_PATTERN.search(article_no)
    return match.group(0) if match else article_no


def _article_number(article_no: str) -> int:
    match = ARTICLE_PATTERN.search(str(article_no or ""))
    if not match:
        return 0
    return chinese_num_to_int(match.group(1))


def _looks_like_known_law(law_name: str, known_laws: set[str]) -> bool:
    key = _law_key(law_name)
    return bool(key and (key in known_laws or f"中华人民共和国{key}" in known_laws))


def _looks_like_legal_source(law_name: str) -> bool:
    if len(law_name) < 2:
        return False
    return law_name.endswith(("法", "典", "条例", "规定", "解释"))


class CitationVerifier:
    """引用验证器：向量语义 + LLM 准确性。"""

    def __init__(self, article_index: Optional[Dict] = None, embeddings=None, llm=None):
        """
        Args:
            article_index: 条号索引 {law_name: {article_num: [chunks]}}。
            embeddings: LangChain Embeddings 实例（用于向量相似度）。
            llm: LLM 实例（用于描述准确性验证）。
        """
        self.article_index = article_index or {}
        self.embeddings = embeddings
        self.llm = llm
        self._embed_cache: Dict[str, list] = {}

    def verify(self, answer: str, sources: List[SourceItem | Dict[str, Any] | str]) -> Dict[str, Any]:
        """Verify answer citations against returned SourceItem entries.

        This lightweight checker only verifies whether an explicit law-name +
        article-number citation in the answer is backed by the returned sources.
        It is intentionally regex-based so the result is explainable and fast.
        """
        source_items = normalize_sources(sources)
        answer_citations = extract_answer_citations(answer, source_items)
        source_keys = {
            (_law_key(item.get("law_name", "")), _article_number(item.get("article_no", "")))
            for item in source_items
            if item.get("law_name") and item.get("article_no")
        }
        matched_citations: List[Dict[str, Any]] = []
        unsupported_citations: List[str] = []

        for citation in answer_citations:
            key = (_law_key(citation["law_name"]), citation["article_num"])
            matched_source = None
            if key[0] and key[1] > 0 and key in source_keys:
                matched_source = next(
                    (
                        item
                        for item in source_items
                        if (_law_key(item.get("law_name", "")), _article_number(item.get("article_no", ""))) == key
                    ),
                    None,
                )
            if matched_source:
                matched_citations.append({
                    "citation": citation["citation"],
                    "law_name": citation["law_name"],
                    "article_no": citation["article_no"],
                    "source_id": matched_source.get("source_id", ""),
                    "source_type": matched_source.get("source_type", "unknown"),
                    "title": matched_source.get("title", ""),
                })
            else:
                unsupported_citations.append(citation["citation"])

        warnings_list: List[str] = []
        if answer_citations and not source_items:
            warnings_list.append("回答包含引用，但 sources 为空，无法核验。")
        if unsupported_citations:
            warnings_list.append("部分引用未能在检索来源中核验。")

        return {
            "verified": not unsupported_citations,
            "unsupported_citations": unsupported_citations,
            "matched_citations": matched_citations,
            "warnings": warnings_list,
        }

    def _get_embedding(self, text: str) -> Optional[list]:
        """获取文本向量（带缓存）。"""
        if not self.embeddings:
            return None
        cache_key = text[:200]
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]
        try:
            vec = self.embeddings.embed_query(text)
            self._embed_cache[cache_key] = vec
            return vec
        except Exception as e:
            logger.debug("[引用验证] 向量化失败: %s", e)
            return None

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """向量余弦相似度，embedding 不可用时回退到关键词重叠。"""
        vec_a = self._get_embedding(text_a)
        vec_b = self._get_embedding(text_b)
        if vec_a is not None and vec_b is not None:
            return _cosine_similarity(vec_a, vec_b)
        # 回退：简单关键词重叠
        if not text_a or not text_b:
            return 0.0
        words_a = set(re.findall(r'[一-鿿]{2,}', text_a))
        words_b = set(re.findall(r'[一-鿿]{2,}', text_b))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / min(len(words_a), len(words_b))

    def _verify_with_llm(self, answer: str, law_name: str, article_text: str) -> Optional[str]:
        """
        用 LLM 验证 answer 对该法条的描述是否准确。
        返回 'high' / 'medium' / 'low' 或 None（验证失败时）。
        """
        if not self.llm:
            return None
        prompt = (
            f"你是法律引用验证专家。判断以下回答中对法条的引用是否准确。\n\n"
            f"回答内容：\n{answer[:1500]}\n\n"
            f"引用的法条原文（{law_name}）：\n{article_text[:500]}\n\n"
            f"请判断回答对这条法律的引用和描述是否准确：\n"
            f"- 如果描述完全准确，回答 'high'\n"
            f"- 如果大意正确但有细节偏差，回答 'medium'\n"
            f"- 如果描述错误或张冠李戴，回答 'low'\n"
            f"只回答 high/medium/low，不要解释。"
        )
        try:
            from langchain_core.messages import HumanMessage
            response = invoke_with_timeout(self.llm, [HumanMessage(content=prompt)], timeout=10)
            result = response.content.strip().lower()
            if result in ("high", "medium", "low"):
                return result
        except Exception as e:
            logger.debug("[引用验证] LLM 验证跳过: %s", e)
        return None

    def verify_citations(
        self,
        sources: List[Dict[str, str]],
        answer: str,
    ) -> List[Dict[str, str]]:
        """
        验证每条引用的准确性（三层）：
        1. 条号存在性
        2. 向量语义相似度
        3. LLM 描述准确性（可选）
        """
        if not self.article_index or not sources:
            return sources

        verified = []
        for src in sources:
            label = src["source"]

            # 尝试匹配已知法律名（从 article_index 中找最长前缀匹配）
            law_name = None
            articles_str = None
            for known_name in self.article_index:
                if label.startswith(known_name):
                    rest = label[len(known_name):].strip()
                    if rest:
                        law_name = known_name
                        articles_str = rest
                        break
            if not law_name:
                # 回退：按空格拆分
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
                # 尝试款级匹配
                para_match = PARA_PATTERN.search(clean_art)
                if para_match:
                    art_num = chinese_num_to_int(para_match.group(1))
                    para_num_str = para_match.group(3)
                else:
                    art_match = ARTICLE_PATTERN.search(clean_art)
                    if not art_match:
                        continue
                    art_num = chinese_num_to_int(art_match.group(1))
                    para_num_str = None

                if art_num <= 0:
                    continue

                # 查找该条原文
                chunks = self.article_index.get(law_name, {}).get(art_num, [])
                if not chunks:
                    continue

                # 有款号时，优先找匹配款的 chunk
                law_text = ""
                if para_num_str:
                    para_int = chinese_num_to_int(para_num_str)
                    if para_int > 0:
                        chinese_digits = "零一二三四五六七八九"
                        if para_int < 10:
                            para_label = f"（{chinese_digits[para_int]}）"
                        else:
                            para_label = f"（{para_int}）"
                        for chunk in chunks:
                            subpara = chunk.metadata.get("subpara", "")
                            if subpara and para_label in subpara:
                                law_text = chunk.page_content[:500]
                                break
                if not law_text:
                    law_text = chunks[0].page_content[:500]

                # 第 2 层：向量语义相似度
                similarity = self._compute_similarity(law_text, answer)
                if similarity >= HIGH_THRESHOLD:
                    confidence = "high"
                elif similarity >= MEDIUM_THRESHOLD:
                    confidence = "medium"
                else:
                    confidence = "low"

                # 第 3 层：LLM 验证（仅对 low confidence 的做二次确认）
                if confidence == "low" and self.llm:
                    llm_result = self._verify_with_llm(answer, law_name, law_text)
                    if llm_result:
                        confidence = llm_result

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
        使用向量相似度 + 提高的阈值减少噪音。
        """
        if not answer or not reranked_docs:
            return []

        # 从 answer 中提取已引用的法律名 + 条号
        cited_articles = set()
        for match in ARTICLE_PATTERN.finditer(answer):
            cited_articles.add(match.group(0))

        cited_laws = set()
        for src_text in re.findall(r'《([^》]+)》', answer):
            cited_laws.add(src_text)

        missing = []
        seen_sources = set()

        for doc in reranked_docs:
            source = doc.metadata.get("source", "")
            if not source or source in seen_sources:
                continue

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

            doc_cited = False
            for num in article_nums:
                for cited in cited_articles:
                    if str(num) in cited or _num_to_chinese(num) in cited:
                        doc_cited = True
                        break
                if doc_cited:
                    break

            if doc_cited:
                continue

            # 向量相似度判断（阈值提高到 MISSING_THRESHOLD）
            similarity = self._compute_similarity(doc.page_content[:300], answer)
            if similarity >= MISSING_THRESHOLD:
                seen_sources.add(source)
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
