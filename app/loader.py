"""
法律文档加载与文本分割模块（全面优化版）

功能概述：
  1. 加载 data 目录下所有 .docx 文件，支持并行读取。
  2. 自动提取法律文本的层级结构（编、章、节、条）。
  3. 按"第X条"优先分割，保持每条语义完整。
  4. 合并过小的相邻 chunk，避免碎片化。
  5. 对超长条内部进行递归分割（段落 → 句子 → 固定长度），保留条号前缀。
  6. 为每个 chunk 生成层级摘要（如"刑法 > 第一章 > 第二十三条"）。
  7. 记录前后条号索引，便于上下文检索扩展。
  8. 支持两种分割策略：'article'（按条，推荐）和 'fixed'（固定长度）。

依赖库：
  - langchain_core
"""

import re
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from app.docx_reader import read_docx_text
from app.lightweight_text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ================== 常量与辅助函数 ==================

# 中文数字映射（用于将"二十三"转为整数）
_CN_DIGITS = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000, '万': 10000,
}

# 匹配"第X条"及变体（如"第X条之一"、"第X条之二"）
ARTICLE_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千万0-9]+)条(?:之([一二三四五六七八九十]+))?'
)

# 匹配编、章、节等标题（可根据实际文书调整）
HIERARCHY_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百千万]+编|第[一二三四五六七八九十百千万]+章|第[一二三四五六七八九十百千万]+节)',
    re.MULTILINE
)

# 匹配子段落编号标记（款级别）：（一）（二）（三）等
SUBPARA_PATTERN = re.compile(
    r'(?=[\(（][一二三四五六七八九十百千]+[）\)])'
)

# 匹配处罚类型：判处/处以/处/判 + [可选：三年以下等修饰语] + 刑罚种类
PENALTY_PATTERN = re.compile(
    r'(?:判处|处以|处|判)'
    r'(?:[十百千万零一二三四五六七八九\d]+年[以之]?(?:上|下|内)?|)?'
    r'(无期徒刑|死刑|有期徒刑|拘役|管制|罚金|剥夺政治权利|没收财产)'
)

# 匹配定义条款：本法所称XX，是指 / 前款所称XX，是指
DEFINITION_PATTERN = re.compile(
    r'本[法条例]所称([一-鿿]{1,20})[，,]?\s*是指|'
    r'前款所称([一-鿿]{1,20})[，,]?\s*是指'
)

# 匹配罪名引用：犯XX罪 / 构成XX罪
CRIME_PATTERN = re.compile(
    r'犯([一-鿿]{2,20}罪)|构成([一-鿿]{2,20}罪)'
)


def _chinese_num_to_int(cn: str) -> int:
    """
    将中文数字字符串转换为整数。
    支持：'二十三' -> 23, '十二万' -> 120000, '二十一' -> 21
    """
    if not cn:
        return 0
    if cn.isdigit():
        return int(cn)

    digit_map = {**{str(i): i for i in range(10)},
                 **{c: i for c, i in _CN_DIGITS.items() if i < 10}}
    unit_map = {c: i for c, i in _CN_DIGITS.items() if i >= 10}

    result = 0
    cur_num = 0
    for ch in cn:
        if ch in digit_map:
            cur_num = digit_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if cur_num == 0:
                cur_num = 1
            result += cur_num * unit
            cur_num = 0
        # 忽略非法字符（如空格）
    result += cur_num
    return result


def _extract_article_numbers(text: str) -> Tuple[List[str], List[int]]:
    """
    从文本中提取所有"第X条"及变体的条号。
    返回：(原文列表, 整数列表)
    例如: (['第二十三条', '第二十三条之一'], [23, 231])
    """
    matches = ARTICLE_PATTERN.findall(text)
    seen = set()
    str_list, int_list = [], []
    for num_part, sub_part in matches:
        if not num_part:
            continue
        base_num = _chinese_num_to_int(num_part)
        if sub_part:
            sub_val = _chinese_num_to_int(sub_part) if sub_part else 0
            full_int = base_num * 10 + sub_val  # 如 23 -> 231
            full_str = f"第{num_part}条之{sub_part}"
        else:
            full_int = base_num
            full_str = f"第{num_part}条"
        key = full_str
        if key not in seen:
            seen.add(key)
            str_list.append(full_str)
            int_list.append(full_int)
    return str_list, int_list


def _extract_hierarchy(text: str) -> List[Dict]:
    """
    提取文本中的编、章、节标题及其位置。
    返回: [{'level': '编', 'title': '第一编', 'position': 10}, ...]
    """
    hierarchy = []
    for match in HIERARCHY_PATTERN.finditer(text):
        title = match.group(0)
        level = '编' if '编' in title else ('章' if '章' in title else '节')
        hierarchy.append({
            'level': level,
            'title': title,
            'position': match.start()
        })
    return hierarchy


def _extract_entities(text: str, law_name: str = "") -> dict:
    """
    从 chunk 文本中提取法律实体（处罚类型、罪名、定义标记）。
    返回 dict，可能包含：penalties, crimes, is_definition, defined_term
    """
    entities = {}

    # 1. 处罚类型
    penalties = list(set(PENALTY_PATTERN.findall(text)))
    if penalties:
        entities["penalties"] = penalties

    # 2. 罪名（仅刑法相关文档）
    if "刑法" in law_name:
        crimes = set()
        for m in CRIME_PATTERN.finditer(text):
            crime = m.group(1) or m.group(2)
            if crime:
                crimes.add(crime)
        if crimes:
            entities["crimes"] = sorted(crimes)

    # 3. 定义标记
    m = DEFINITION_PATTERN.search(text)
    if m:
        term = m.group(1) or m.group(2)
        if term:
            entities["is_definition"] = True
            entities["defined_term"] = term

    return entities


def _get_hierarchy_path(text: str, position: int, hierarchy_list: List[Dict]) -> List[str]:
    """
    根据当前 chunk 在原文中的起始位置，获取完整的 编 > 章 > 节 路径。
    用状态变量维护当前层级：遇到 编 清空章/节，遇到 章 清空节，遇到 节 覆盖。
    """
    if not hierarchy_list:
        return []
    volume = chapter = section = None
    for h in hierarchy_list:
        if h['position'] >= position:
            break
        if h['level'] == '编':
            volume, chapter, section = h['title'], None, None
        elif h['level'] == '章':
            chapter, section = h['title'], None
        elif h['level'] == '节':
            section = h['title']
    path = []
    if volume:
        path.append(volume)
    if chapter:
        path.append(chapter)
    if section:
        path.append(section)
    return path


def _split_by_subparagraphs(
    article_num: str,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    按子段落编号（如（一）（二））分割条文正文，尊重款级别的编号结构。
    若无编号或单款仍超长，退化为 RecursiveCharacterTextSplitter。
    """
    parts = SUBPARA_PATTERN.split(body)
    # parts[0] 是第一个编号前的引言部分（可能为空字符串）
    # parts[1:] 每个以（一）等编号开头

    prefix = f"{article_num}"
    max_body = chunk_size - len(prefix) - 2

    preamble = parts[0].strip() if parts[0].strip() else ""
    subparas = []
    for i, part in enumerate(parts[1:], 1):
        part = part.strip()
        if not part:
            continue
        # 提取编号标记
        m = re.match(r'^[（(][一二三四五六七八九十百千]+[）)]', part)
        label = m.group(0) if m else ""
        # 引言附到第一个子段落前面
        if i == 1 and preamble:
            part = preamble + "\n" + part
        subparas.append((label, part))

    if not subparas:
        # 无有效子段落，退化为递归分割
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            chunk_size=max_body,
            chunk_overlap=chunk_overlap,
        )
        docs = []
        for idx, chunk_body in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=f"{prefix}{chunk_body}",
                metadata={"article": article_num, "sub_part": idx}
            ))
        return docs

    docs = []
    for idx, (label, text) in enumerate(subparas):
        if len(text) <= max_body:
            docs.append(Document(
                page_content=f"{prefix}{text}",
                metadata={"article": article_num, "subpara": label}
            ))
        else:
            # 单个款仍超长，递归分割
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
                chunk_size=max_body,
                chunk_overlap=chunk_overlap,
            )
            sub_chunks = splitter.split_text(text)
            for si, sc in enumerate(sub_chunks):
                docs.append(Document(
                    page_content=f"{prefix}{sc}",
                    metadata={
                        "article": article_num,
                        "subpara": label,
                        "sub_part": si,
                        "total_sub_parts": len(sub_chunks),
                    }
                ))
    return docs


# ================== 核心分割函数 ==================

def _split_by_articles(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    hierarchy: Optional[List[Dict]] = None
) -> List[Document]:
    """
    按"第X条"分割法律文本，保证每条独立。
    超长条内部会进一步按段落→句子→固定长度分割，并保留条号前缀。
    """
    heading_pattern = re.compile(
        r'第[一二三四五六七八九十百千万0-9]+条(?:之[一二三四五六七八九十]+)?(?!的)'
    )
    matches = [
        match for match in heading_pattern.finditer(text)
        if _looks_like_article_heading(text, match.start(), match.end())
    ]

    if not matches:
        # 无任何条号，退化为普通分块
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.create_documents([text])

    article_docs = []
    for idx, match in enumerate(matches):
        article_num = match.group(0).strip()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        article_body = text[match.end():next_start].strip()
        full_article = f"{article_num}{article_body}"
        # 获取该条在原文中的起始位置（用于层级匹配）
        start_pos = match.start()
        # 获取层级路径（若有）
        hierarchy_path = _get_hierarchy_path(text, start_pos, hierarchy) if hierarchy else []
        hierarchy_str = " > ".join(hierarchy_path) if hierarchy_path else ""

        if len(full_article) <= chunk_size:
            doc = Document(
                page_content=full_article,
                metadata={
                    "article": article_num,
                    "hierarchy_path": hierarchy_str,
                    "start_pos": start_pos,
                }
            )
            article_docs.append(doc)
        else:
            # 超长条：先按子段落编号（款）分割，保留条号前缀
            subpara_docs = _split_by_subparagraphs(
                article_num, article_body, chunk_size, chunk_overlap
            )
            for d in subpara_docs:
                d.metadata["hierarchy_path"] = hierarchy_str
                d.metadata["start_pos"] = start_pos
            article_docs.extend(subpara_docs)
    return article_docs


def _looks_like_article_heading(text: str, start: int, end: int) -> bool:
    """Return True when a 第X条 occurrence is likely an article heading.

    Legal texts also contain cross references such as "依照本法第二百六十四条的规定".
    Those must not become article chunk boundaries.
    """
    prev = text[start - 1] if start > 0 else ""
    if prev and prev not in "\n\r 。；;：:！？!?）)":
        return False

    following = text[end:end + 3]
    if following.startswith(("的", "规定", "款", "项")):
        return False
    return True


def _merge_small_chunks(
    chunks: List[Document],
    min_size: int = 200,
    max_size: int = 1200
) -> List[Document]:
    """
    合并长度小于 min_size 的相邻 chunk，避免碎片化。
    合并后总长度不超过 max_size。
    """
    if not chunks:
        return []
    merged = []
    buffer = chunks[0]
    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_size and \
           len(buffer.page_content) + len(chunk.page_content) <= max_size:
            # 合并内容
            buffer.page_content += "\n\n" + chunk.page_content
            # 合并元数据中的条号
            art_old = buffer.metadata.get("article_numbers", "")
            art_new = chunk.metadata.get("article_numbers", "")
            if art_new:
                buffer.metadata["article_numbers"] = art_old + "," + art_new if art_old else art_new
            # 合并层级路径（取较详细的）
            path_old = buffer.metadata.get("hierarchy_path", "")
            path_new = chunk.metadata.get("hierarchy_path", "")
            if path_new and (not path_old or len(path_new) > len(path_old)):
                buffer.metadata["hierarchy_path"] = path_new
            # 其他字段可酌情合并
        else:
            merged.append(buffer)
            buffer = chunk
    merged.append(buffer)
    return merged


def _add_article_index(chunks: List[Document]) -> List[Document]:
    """
    为每个 chunk 添加前后条号索引，用于上下文扩展。
    需要先提取所有 chunk 中出现过的整数条号，排序后建立映射。
    """
    # 收集所有不同的整数条号
    all_articles = set()
    for ch in chunks:
        nums_str = ch.metadata.get("article_numbers_int", "")
        for n in nums_str.split(","):
            if n and n.isdigit():
                all_articles.add(int(n))
        # 如果 metadata 中没有 article_numbers_int，则从 article 字段提取
        if 'article' in ch.metadata:
            article_str = ch.metadata['article']
            # 尝试提取整数
            match = ARTICLE_PATTERN.search(article_str)
            if match:
                num_part = match.group(1)
                base_int = _chinese_num_to_int(num_part)
                sub_part = match.group(2)
                if sub_part:
                    sub_int = _chinese_num_to_int(sub_part)
                    full_int = base_int * 10 + sub_int
                else:
                    full_int = base_int
                all_articles.add(full_int)

    sorted_articles = sorted(all_articles)

    for ch in chunks:
        # 获取该 chunk 的主条号（优先用 article_numbers_int 的第一个）
        main_int = None
        nums_str = ch.metadata.get("article_numbers_int", "")
        if nums_str:
            parts = nums_str.split(",")
            if parts and parts[0].isdigit():
                main_int = int(parts[0])
        if main_int is None and 'article' in ch.metadata:
            article_str = ch.metadata['article']
            match = ARTICLE_PATTERN.search(article_str)
            if match:
                num_part = match.group(1)
                base_int = _chinese_num_to_int(num_part)
                sub_part = match.group(2)
                if sub_part:
                    sub_int = _chinese_num_to_int(sub_part)
                    main_int = base_int * 10 + sub_int
                else:
                    main_int = base_int
        if main_int is not None and main_int in sorted_articles:
            idx = sorted_articles.index(main_int)
            ch.metadata["article_index"] = idx
            if idx > 0:
                ch.metadata["prev_article"] = sorted_articles[idx - 1]
            if idx + 1 < len(sorted_articles):
                ch.metadata["next_article"] = sorted_articles[idx + 1]
    return chunks


def _generate_summary(chunk: Document) -> str:
    """
    生成该 chunk 的层级摘要，格式如："刑法 > 第一章 > 第二十三条 - （一） - Part 1/2"
    """
    hierarchy = chunk.metadata.get("hierarchy_path", "")
    article = chunk.metadata.get("article", "")
    subpara = chunk.metadata.get("subpara", "")
    sub = chunk.metadata.get("sub_part")
    total = chunk.metadata.get("total_sub_parts")

    parts = []
    if hierarchy:
        parts.append(hierarchy)
    if article:
        parts.append(article)
    base = " > ".join(parts) if parts else "未命名"
    if subpara:
        base += f" - {subpara}"
    if sub is not None and total and total > 1:
        return f"{base} - Part {sub+1}/{total}"
    return base


# ================== 对外公开接口 ==================

def _normalize_exclude_dirs(exclude_dirs: Optional[List[str] | str]) -> set[str]:
    """标准化需要排除的目录名列表。"""
    if not exclude_dirs:
        return set()
    if isinstance(exclude_dirs, str):
        items = exclude_dirs.split(",")
    else:
        items = exclude_dirs
    return {str(item).strip().strip("/\\") for item in items if str(item).strip()}


def _is_excluded_path(path: Path, root: Path, exclude_dirs: set[str]) -> bool:
    """判断 path 是否位于排除目录下。"""
    if not exclude_dirs:
        return False
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    return any(part in exclude_dirs for part in parts)


def _discover_docx_files(data_dir: str, exclude_dirs: Optional[List[str] | str] = None) -> List[Path]:
    """发现参与主法律文书索引的 docx 文件，不读取正文。"""
    data_path = Path(data_dir)
    excluded = _normalize_exclude_dirs(exclude_dirs)
    return [
        f for f in sorted(data_path.rglob("*.docx"))
        if not _is_excluded_path(f, data_path, excluded)
    ]


def load_documents(
    data_dir: str,
    parallel: bool = True,
    exclude_dirs: Optional[List[str] | str] = None,
) -> List[Document]:
    """
    加载 data 目录下参与主法条库的 .docx 文件为 Document 列表。
    Args:
        data_dir: 存放 .docx 文件的目录
        parallel: 是否并行加载（默认 True）
        exclude_dirs: 需要跳过的 data 子目录名或逗号分隔字符串
    Returns:
        List[Document]: 每个 Document 的 page_content 为原始文档全文，
                        metadata 中包含 source（法律名称）和 file_path。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    docx_files = _discover_docx_files(data_dir, exclude_dirs)
    if not docx_files:
        raise FileNotFoundError(f"未找到 .docx 文件: {data_dir}")

    def load_one(filepath: Path):
        raw_name = filepath.stem
        # 去除文件名末尾的日期（如 _20221228）
        law_name = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        text = read_docx_text(filepath)
        return [
            Document(
                page_content=text,
                metadata={"source": law_name, "file_path": str(filepath)},
            )
        ]

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_one, docx_files)
        all_docs = [doc for sublist in results for doc in sublist]
    else:
        all_docs = []
        for fp in docx_files:
            all_docs.extend(load_one(fp))

    excluded = _normalize_exclude_dirs(exclude_dirs)
    if excluded:
        logger.info("已排除 data 子目录：%s", sorted(excluded))
    logger.info("加载完成：%d 个文件，共 %d 个原始文档块", len(docx_files), len(all_docs))
    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 200,
    split_by: Literal["article", "fixed"] = "article",
) -> List[Document]:
    """
    对文档进行分割。
    Args:
        docs: Document 列表（通常由 load_documents 返回）
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 之间的重叠字符数
        min_chunk_size: 最小 chunk 长度，小于此值的会尝试与下一个合并（仅 split_by='article' 时生效）
        split_by: 'article' 按法律条文分割（推荐），'fixed' 按固定长度分割
    Returns:
        分割后的 Document 列表，每个 Document 的 metadata 包含：
            - source, file_path (来自原始文档)
            - article (当前条号)
            - article_numbers, article_numbers_int (所有出现的条号)
            - hierarchy_path (编/章/节路径)
            - sub_part, total_sub_parts (若条被拆分)
            - article_index, prev_article, next_article (用于上下文扩展)
            - summary (生成的层级摘要)
    """
    final_chunks = []

    if split_by == "article":
        # 首先，对每个原始文档分别提取层级结构（编、章、节）
        for doc in docs:
            full_text = doc.page_content
            hierarchy = _extract_hierarchy(full_text)  # 提取该文档内的所有标题
            # 按条分割
            article_chunks = _split_by_articles(full_text, chunk_size, chunk_overlap, hierarchy)
            # 继承原始 metadata，并添加条号等元数据
            for ac in article_chunks:
                ac.metadata.update(doc.metadata)
                # 提取当前 chunk 中的条号信息
                str_arts, int_arts = _extract_article_numbers(ac.page_content)
                ac.metadata["article_numbers"] = ",".join(str_arts)
                ac.metadata["article_numbers_int"] = ",".join(str(i) for i in int_arts)
                # 拆分层级路径为独立字段（用于 metadata filter）
                hp = ac.metadata.get("hierarchy_path", "")
                parts = hp.split(" > ") if hp else []
                # 按顺序：编 > 章 > 节
                level_keys = {"编": "volume", "章": "chapter", "节": "section"}
                for p in parts:
                    for suffix, key in level_keys.items():
                        if suffix in p:
                            ac.metadata[key] = p
                # 生成摘要
                ac.metadata["summary"] = _generate_summary(ac)
                # 提取法律实体（处罚类型、罪名、定义标记）
                law_name = ac.metadata.get("source", "")
                entities = _extract_entities(ac.page_content, law_name)
                ac.metadata["entities"] = json.dumps(entities, ensure_ascii=False) if entities else ""
            final_chunks.extend(article_chunks)

        # 合并过小的相邻 chunk
        final_chunks = _merge_small_chunks(final_chunks, min_size=min_chunk_size, max_size=chunk_size + chunk_overlap)
        # 添加前后条索引
        final_chunks = _add_article_index(final_chunks)
        # 提取跨条引用（在合并之后，基于最终 chunk 内容）
        for ch in final_chunks:
            own_article = ch.metadata.get("article", "")
            if not own_article:
                continue
            text_refs = set()
            for m in ARTICLE_PATTERN.finditer(ch.page_content):
                ref_full = m.group(0)
                if ref_full != own_article:
                    text_refs.add(ref_full)
            ch.metadata["referenced_articles"] = ",".join(sorted(text_refs)) if text_refs else ""
        # 更新摘要（合并后摘要可能改变，可重新生成，此处简化，保留原第一个 chunk 的摘要）
        for ch in final_chunks:
            if "summary" not in ch.metadata:
                ch.metadata["summary"] = _generate_summary(ch)
    else:
        # 固定长度分割
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_chunks = splitter.split_documents(docs)
        # 为每个 chunk 添加条号信息
        for chunk in final_chunks:
            str_arts, int_arts = _extract_article_numbers(chunk.page_content)
            chunk.metadata["article_numbers"] = ",".join(str_arts)
            chunk.metadata["article_numbers_int"] = ",".join(str(i) for i in int_arts)
            chunk.metadata["summary"] = chunk.metadata.get("article", "未命名段落")
            law_name = chunk.metadata.get("source", "")
            entities = _extract_entities(chunk.page_content, law_name)
            chunk.metadata["entities"] = json.dumps(entities, ensure_ascii=False) if entities else ""

    logger.info("分割完成：%d 个原始文档块 → %d 个 chunk", len(docs), len(final_chunks))
    return final_chunks


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 假设 data 目录下有 .docx 法律文书
    # docs = load_documents("data/")
    # chunks = split_documents(docs, chunk_size=1200, split_by="article")
    # for chunk in chunks[:3]:
    #     print(f"摘要: {chunk.metadata['summary']}")
    #     print(f"内容预览: {chunk.page_content[:100]}...")
    #     print(f"前后条: {chunk.metadata.get('prev_article')} -> {chunk.metadata.get('article')} -> {chunk.metadata.get('next_article')}")
    #     print("-" * 50)
    pass
