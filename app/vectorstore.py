"""
ChromaDB 向量存储管理模块。

功能：
  - 构建向量库：将文档 embedding 后存入 ChromaDB 并持久化
  - 加载已有向量库
  - 智能判断：存在则加载，不存在则自动构建
  - 文档变更感知：data/ 目录文件变化时自动重建索引
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


def _compute_data_hash(data_dir: str) -> str:
    """计算 data 目录下所有 .docx 文件的指纹（文件名+大小+修改时间）。"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return ""
    entries = []
    for f in sorted(data_path.glob("*.docx")):
        stat = f.stat()
        entries.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def _read_stored_hash(persist_dir: str) -> str:
    """读取上次构建时保存的数据指纹。"""
    hash_file = os.path.join(persist_dir, ".data_hash")
    if os.path.exists(hash_file):
        return open(hash_file).read().strip()
    return ""


def _save_data_hash(persist_dir: str, data_hash: str):
    """保存当前数据指纹到向量库目录。"""
    hash_file = os.path.join(persist_dir, ".data_hash")
    with open(hash_file, "w") as f:
        f.write(data_hash)


def build_vectorstore(
    docs: List[Document],
    embeddings: Embeddings,
    persist_dir: str,
    data_dir: str = "",
) -> Chroma:
    """
    从文档列表构建 ChromaDB 向量库并持久化到磁盘。

    若目标目录已有旧数据，会先清空再重建，确保数据与当前文档一致。

    Args:
        docs: 分割后的 Document 列表。
        embeddings: Embedding 模型实例。
        persist_dir: 持久化目录路径。
        data_dir: 数据目录路径，用于保存文件指纹。

    Returns:
        构建完成的 Chroma 向量存储实例。
    """
    # 清空旧数据，确保每次 build 都是全新状态
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"向量库构建完成，{len(docs)} 个 chunk 已存入 {persist_dir}")

    # 保存数据指纹，用于后续判断是否需要重建
    if data_dir:
        data_hash = _compute_data_hash(data_dir)
        _save_data_hash(persist_dir, data_hash)

    return vectorstore


def load_vectorstore(
    embeddings: Embeddings,
    persist_dir: str,
) -> Chroma:
    """
    从磁盘加载已有的 ChromaDB 向量库。

    Args:
        embeddings: 与构建时一致的 Embedding 模型实例。
        persist_dir: 向量库持久化目录路径。

    Returns:
        加载的 Chroma 向量存储实例。
    """
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    print(f"向量库已从 {persist_dir} 加载")
    return vectorstore


def get_or_create_vectorstore(
    docs: List[Document],
    embeddings: Embeddings,
    persist_dir: str,
    data_dir: str = "",
) -> Chroma:
    """
    自动判断加载或构建向量库。

    逻辑：
      - 若 persist_dir 存在且数据指纹一致 → 直接加载
      - 否则 → 从 docs 构建新库

    Args:
        docs: 分割后的 Document 列表（仅在构建时使用）。
        embeddings: Embedding 模型实例。
        persist_dir: 向量库持久化目录路径。
        data_dir: 数据目录路径，用于文件变更检测。

    Returns:
        Chroma 向量存储实例。
    """
    chroma_sqlite = os.path.join(persist_dir, "chroma.sqlite3")

    if os.path.exists(chroma_sqlite) and os.path.getsize(chroma_sqlite) > 0:
        # 检查数据文件是否发生变化
        if data_dir:
            current_hash = _compute_data_hash(data_dir)
            stored_hash = _read_stored_hash(persist_dir)
            if current_hash and stored_hash and current_hash != stored_hash:
                print("检测到 data/ 目录文件变化，重新构建向量库...")
                return build_vectorstore(docs, embeddings, persist_dir, data_dir)

        print("检测到已有向量库，直接加载...")
        return load_vectorstore(embeddings, persist_dir)
    else:
        print("向量库不存在，开始构建...")
        return build_vectorstore(docs, embeddings, persist_dir, data_dir)
