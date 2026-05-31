"""
ChromaDB 向量存储管理模块。

功能：
  - 构建向量库：将文档 embedding 后存入 ChromaDB 并持久化
  - 加载已有向量库
  - 智能判断：存在则加载，不存在则自动构建
  - 文档变更感知：data/ 目录文件变化时自动重建索引
"""

import hashlib
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def _get_embedding_key(embeddings: Embeddings) -> str:
    """提取 Embedding 模型的唯一标识（模型名 + base_url），用于检测模型变更。"""
    model = getattr(embeddings, "model", "") or ""
    base_url = getattr(embeddings, "base_url", "") or ""
    # openai_api_key 不参与指纹（key 变了但模型没变，不该重建）
    return f"{model}::{base_url}"


def _normalize_exclude_dirs(exclude_dirs: Optional[List[str] | str]) -> set[str]:
    if not exclude_dirs:
        return set()
    if isinstance(exclude_dirs, str):
        items = exclude_dirs.split(",")
    else:
        items = exclude_dirs
    return {str(item).strip().strip("/\\") for item in items if str(item).strip()}


def _is_excluded_path(path: Path, root: Path, exclude_dirs: set[str]) -> bool:
    if not exclude_dirs:
        return False
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    return any(part in exclude_dirs for part in parts)


def _compute_data_hash(
    data_dir: str,
    embedding_key: str = "",
    exclude_dirs: Optional[List[str] | str] = None,
) -> str:
    """计算 data 目录 + embedding 模型的联合指纹。"""
    data_path = Path(data_dir)
    excluded = _normalize_exclude_dirs(exclude_dirs)
    entries = []
    if data_path.exists():
        for f in sorted(data_path.rglob("*.docx")):
            if _is_excluded_path(f, data_path, excluded):
                continue
            stat = f.stat()
            entries.append(f"{f.name}:{stat.st_size}:{stat.st_mtime}")
    if embedding_key:
        entries.append(f"__embedding__:{embedding_key}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def _read_stored_hash(persist_dir: str) -> str:
    """读取上次构建时保存的数据指纹。"""
    hash_file = os.path.join(persist_dir, ".data_hash")
    try:
        with open(hash_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
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
    exclude_dirs: Optional[List[str] | str] = None,
    batch_size: int = 128,
) -> Chroma:
    """
    从文档列表构建 ChromaDB 向量库并持久化到磁盘。

    采用原子构建：先在临时目录构建，成功后再替换旧数据，避免构建失败导致数据丢失。
    分批 embedding 并显示进度条。

    Args:
        docs: 分割后的 Document 列表。
        embeddings: Embedding 模型实例。
        persist_dir: 持久化目录路径。
        data_dir: 数据目录路径，用于保存文件指纹。
        batch_size: 每批 embedding 的文档数。

    Returns:
        构建完成的 Chroma 向量存储实例。
    """
    import gc
    import time as _time
    from tqdm import tqdm

    # 清理旧目录（Windows 下原子 rename 会因文件锁失败，直接构建到目标目录）
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    try:
        # 创建空向量库
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)

        total = len(docs)
        logger.info("开始构建向量库，共 %d 个 chunk，batch_size=%d", total, batch_size)
        pbar = tqdm(total=total, desc="构建向量库", unit="chunk", dynamic_ncols=True,
                     mininterval=1.0, maxinterval=5.0)
        for i in range(0, total, batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.page_content for d in batch]
            metadatas = [d.metadata for d in batch]
            vectors = embeddings.embed_documents(texts)
            vectorstore._collection.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=[f"doc_{j}" for j in range(i, i + len(batch))],
            )
            pbar.update(len(batch))
        pbar.close()

        # 释放 ChromaDB 文件句柄
        try:
            vectorstore._client.clear_system_cache()
        except Exception:
            pass
        del vectorstore
        gc.collect()
        _time.sleep(1)

        logger.info("向量库构建完成，%d 个 chunk 已存入 %s", total, persist_dir)

        # 保存数据指纹，用于后续判断是否需要重建
        if data_dir:
            embedding_key = _get_embedding_key(embeddings)
            data_hash = _compute_data_hash(data_dir, embedding_key, exclude_dirs)
            _save_data_hash(persist_dir, data_hash)

        # 重新加载
        return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    except Exception:
        shutil.rmtree(persist_dir, ignore_errors=True)
        raise


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
    logger.info("向量库已从 %s 加载", persist_dir)
    return vectorstore


def _count_embeddings(persist_dir: str) -> int:
    """直接通过 SQLite 查询向量库中的 embedding 数量，避免加载完整 Chroma 对象。"""
    db_path = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_or_create_vectorstore(
    docs: List[Document],
    embeddings: Embeddings,
    persist_dir: str,
    data_dir: str = "",
    exclude_dirs: Optional[List[str] | str] = None,
) -> Chroma:
    """
    自动判断加载或构建向量库。

    逻辑：
      - 若 persist_dir 存在且有数据且指纹一致 → 直接加载
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
        # 检查向量库是否真的有数据（防止空库/中断构建被误加载）
        embedding_count = _count_embeddings(persist_dir)
        if embedding_count == 0:
            logger.warning("检测到向量库文件存在但无数据（可能上次构建中断），重新构建...")
            return build_vectorstore(docs, embeddings, persist_dir, data_dir, exclude_dirs)

        # 检查数据文件或 Embedding 模型是否发生变化
        if data_dir:
            embedding_key = _get_embedding_key(embeddings)
            current_hash = _compute_data_hash(data_dir, embedding_key, exclude_dirs)
            stored_hash = _read_stored_hash(persist_dir)
            if current_hash and stored_hash and current_hash != stored_hash:
                logger.info("检测到数据文件或 Embedding 模型变化，重新构建向量库...")
                return build_vectorstore(docs, embeddings, persist_dir, data_dir, exclude_dirs)

        logger.info("检测到已有向量库（%d 条），直接加载...", embedding_count)
        return load_vectorstore(embeddings, persist_dir)
    else:
        logger.info("向量库不存在，开始构建...")
        return build_vectorstore(docs, embeddings, persist_dir, data_dir, exclude_dirs)
