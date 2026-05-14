"""
法律文档加载与文本分割模块（全面优化版）

功能概述：
  1. 加载 data 目录下所有 .docx 文件，支持并行读取。
  2. 自动提取法律文本的层级结构（编、章、节、条）。
  3. 按“第X条”优先分割，保持每条语义完整。
  4. 合并过小的相邻 chunk，避免碎片化。
  5. 对超长条内部进行递归分割（段落 → 句子 → 固定长度），保留条号前缀。
  6. 为每个 chunk 生成层级摘要（如“刑法 > 第一章 > 第二十三条”）。
  7. 记录前后条号索引，便于上下文检索扩展。
  8. 支持两种分割策略：'article'（按条，推荐）和 'fixed'（固定长度）。

依赖库：
  - langchain_core, langchain_community, langchain_text_splitters
  - python-docx (Docx2txtLoader 依赖)
"""

import re
import os
from pathlib import Path
from typing import List, Tuple, Dict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ================== 常量与辅助函数 ==================

# 中文数字映射（用于将“二十三”转为整数）
_CN_DIGITS = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000, '万': 10000,
}

# 匹配“第X条”及变体（如“第X条之一”、“第X条之二”）
ARTICLE_PATTERN = re.compile(
    r'第([一二三四五六七八九十百千万0-9]+)条(?:之([一二三四五六七八九十]+))?'
)

# 匹配编、章、节等标题（可根据实际文书调整）
HIERARCHY_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百千万]+编|第[一二三四五六七八九十百千万]+章|第[一二三四五六七八九十百千万]+节)',
    re.MULTILINE
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
    从文本中提取所有“第X条”及变体的条号。
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


def _get_hierarchy_path(text: str, position: int, hierarchy_list: List[Dict]) -> List[str]:
    """
    根据当前 chunk 在原文中的起始位置（近似），获取它所属的编/章/节路径。
    实际实现中，由于按条分割时无法精确获取位置，可采用以下简化：
      先对整个文档提取所有标题，然后在按条分割时记录每条所在的最近上级标题。
    为简化示例，这里假设在 split_by_articles 中我们已经传入了包含层级信息的上下文。
    下面给出一个占位实现：若 hierarchy_list 非空，则返回最后一个匹配的标题。
    """
    # 实际生产环境中，应在分割时动态记录。此处返回简单占位。
    if not hierarchy_list:
        return []
    # 找到位置小于给定 position 的最后一个标题
    last_title = None
    for h in reversed(hierarchy_list):
        if h['position'] < position:
            last_title = h['title']
            break
    return [last_title] if last_title else []


# ================== 核心分割函数 ==================

def _split_by_articles(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    hierarchy: Optional[List[Dict]] = None
) -> List[Document]:
    """
    按“第X条”分割法律文本，保证每条独立。
    超长条内部会进一步按段落→句子→固定长度分割，并保留条号前缀。
    """
    # 先用正则切割出每条及其内容
    # 匹配从“第X条”到下一个“第X条”或文末
    pattern = re.compile(
        r'(第[一二三四五六七八九十百千万0-9]+条(?:之[一二三四五六七八九十]+)?)(.*?)(?=第[一二三四五六七八九十百千万0-9]+条(?:之[一二三四五六七八九十]+)?|$)',
        re.DOTALL
    )
    matches = list(pattern.finditer(text))

    if not matches:
        # 无任何条号，退化为普通分块
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.create_documents([text])

    article_docs = []
    for match in matches:
        article_num = match.group(1).strip()
        article_body = match.group(2).strip()
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
            # 超长条：对正文部分进行递归分割，保留条号前缀
            inner_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
                chunk_size=chunk_size - len(article_num) - 2,
                chunk_overlap=chunk_overlap,
            )
            body_chunks = inner_splitter.split_text(article_body)
            total = len(body_chunks)
            for idx, chunk_body in enumerate(body_chunks):
                doc = Document(
                    page_content=f"{article_num}{chunk_body}",
                    metadata={
                        "article": article_num,
                        "sub_part": idx,
                        "total_sub_parts": total,
                        "hierarchy_path": hierarchy_str,
                        "start_pos": start_pos,
                    }
                )
                article_docs.append(doc)
    return article_docs


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
    生成该 chunk 的层级摘要，格式如：“刑法 > 第一章 > 第二十三条 - Part 1/2”
    """
    hierarchy = chunk.metadata.get("hierarchy_path", "")
    article = chunk.metadata.get("article", "")
    sub = chunk.metadata.get("sub_part")
    total = chunk.metadata.get("total_sub_parts")

    parts = []
    if hierarchy:
        parts.append(hierarchy)
    if article:
        parts.append(article)
    base = " > ".join(parts) if parts else "未命名"
    if sub is not None and total and total > 1:
        return f"{base} - Part {sub+1}/{total}"
    return base


# ================== 对外公开接口 ==================

def load_documents(data_dir: str, parallel: bool = True) -> List[Document]:
    """
    加载 data 目录下所有 .docx 文件为 Document 列表。
    Args:
        data_dir: 存放 .docx 文件的目录
        parallel: 是否并行加载（默认 True）
    Returns:
        List[Document]: 每个 Document 的 page_content 为原始文档全文，
                        metadata 中包含 source（法律名称）和 file_path。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    docx_files = sorted(data_path.glob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"未找到 .docx 文件: {data_dir}")

    def load_one(filepath: Path):
        raw_name = filepath.stem
        # 去除文件名末尾的日期（如 _20221228）
        law_name = raw_name.rsplit("_", 1)[0] if "_" in raw_name else raw_name
        loader = Docx2txtLoader(str(filepath))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = law_name
            doc.metadata["file_path"] = str(filepath)
        return docs

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_one, docx_files)
        all_docs = [doc for sublist in results for doc in sublist]
    else:
        all_docs = []
        for fp in docx_files:
            all_docs.extend(load_one(fp))

    print(f"加载完成：{len(docx_files)} 个文件，共 {len(all_docs)} 个原始文档块")
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
                # 生成摘要
                ac.metadata["summary"] = _generate_summary(ac)
            final_chunks.extend(article_chunks)

        # 合并过小的相邻 chunk
        final_chunks = _merge_small_chunks(final_chunks, min_size=min_chunk_size, max_size=chunk_size + chunk_overlap)
        # 添加前后条索引
        final_chunks = _add_article_index(final_chunks)
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

    print(f"分割完成：{len(docs)} 个原始文档块 → {len(final_chunks)} 个 chunk")
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