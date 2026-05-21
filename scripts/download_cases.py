"""
下载 CaseMatch 案例库（cases.sqlite3）。

使用方式：
  # 直接运行（默认使用 hf-mirror 镜像）
  python scripts/download_cases.py

  # 指定官方源
  python scripts/download_cases.py --mirror ""

  # 指定目标目录
  python scripts/download_cases.py --local-dir ./data/CaseMatch

  # 用 huggingface-cli（不通过此脚本）
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    --repo-type dataset Yuel-P/CaseMatch-Agent-data \
    --local-dir data/CaseMatch
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Yuel-P/CaseMatch-Agent-data"
DB_FILENAME = "cases.sqlite3"
MIRROR_URL = "https://hf-mirror.com"


def download_with_hf_hub(local_dir: str, mirror: str):
    """通过 huggingface_hub 下载 cases.sqlite3。"""
    try:
        from huggingface_hub import hf_hub_download, HfFolder
    except ImportError:
        print("需要安装 huggingface_hub：")
        print("  pip install huggingface_hub")
        sys.exit(1)

    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"[下载] 使用镜像源: {mirror}")
    else:
        print("[下载] 使用 HuggingFace 官方源")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    target = local_dir / DB_FILENAME
    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"[跳过] 已存在: {target} ({size_mb:.1f} MB)")
        return str(target)

    print(f"[下载] 从 {REPO_ID} 获取 {DB_FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=DB_FILENAME,
        local_dir=str(local_dir),
    )

    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"[完成] {path} ({size_mb:.1f} MB)")
    return path


def main():
    parser = argparse.ArgumentParser(description="下载 CaseMatch 案例库")
    parser.add_argument(
        "--local-dir",
        default="./data/CaseMatch",
        help="下载目标目录 (默认: ./data/CaseMatch)",
    )
    parser.add_argument(
        "--mirror",
        default=MIRROR_URL,
        help=f"HF 镜像地址 (默认: {MIRROR_URL}，留空使用官方源)",
    )
    args = parser.parse_args()

    download_with_hf_hub(args.local_dir, args.mirror)


if __name__ == "__main__":
    main()
