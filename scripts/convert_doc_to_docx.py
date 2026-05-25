"""
将 data/司法解释/ 下的 .doc 文件转换为 .docx。

使用 WPS COM 接口进行转换（需要安装 WPS Office 和 pywin32）。
跳过已有同名 .docx 的文件。

用法：
    python scripts/convert_doc_to_docx.py [--delete-doc]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "司法解释"

# WPS 保存为 .docx 的格式常量
WPS_FORMAT_DOCX = 12  # wdFormatXMLDocument


def get_wps_app():
    """获取 WPS Writer COM 对象。"""
    import win32com.client

    # 尝试常见的 WPS COM 名称
    for name in ["kwps.Application", "WPS.Application", "Word.Application"]:
        try:
            app = win32com.client.Dispatch(name)
            app.Visible = False
            return app, name
        except Exception:
            continue
    raise RuntimeError(
        "无法启动 WPS/Word COM 接口。请确认 WPS Office 已安装。"
    )


def convert_doc_to_docx(wps_app, doc_path: Path) -> bool:
    """用 WPS 打开 .doc 并另存为 .docx。"""
    try:
        doc = wps_app.Documents.Open(str(doc_path))
        docx_path = str(doc_path) + "x"  # .doc -> .docx
        doc.SaveAs2(docx_path, FileFormat=WPS_FORMAT_DOCX)
        doc.Close()
        return True
    except Exception as e:
        print(f"  转换失败: {e}")
        try:
            doc.Close()
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="将 .doc 转换为 .docx")
    parser.add_argument(
        "--delete-doc",
        action="store_true",
        help="转换成功后删除原 .doc 文件",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"目录不存在: {DATA_DIR}")
        sys.exit(1)

    # 找出需要转换的 .doc 文件（没有对应 .docx 的）
    doc_files = sorted(DATA_DIR.glob("*.doc"))
    docx_stems = {f.stem for f in DATA_DIR.glob("*.docx")}
    to_convert = [f for f in doc_files if f.stem not in docx_stems]

    print(f".doc 总数: {len(doc_files)}")
    print(f"已有 .docx 对应（跳过）: {len(doc_files) - len(to_convert)}")
    print(f"需要转换: {len(to_convert)}")

    if not to_convert:
        print("没有需要转换的文件。")
        return

    # 启动 WPS
    print("\n启动 WPS...")
    wps_app, com_name = get_wps_app()
    print(f"已启动: {com_name}")

    success = 0
    failed = 0

    for i, doc_path in enumerate(to_convert, 1):
        print(f"[{i}/{len(to_convert)}] {doc_path.name}...", end="", flush=True)
        ok = convert_doc_to_docx(wps_app, doc_path)
        if ok:
            success += 1
            print(" OK")
            if args.delete_doc:
                doc_path.unlink()
        else:
            failed += 1
            print(" FAIL")

    # 关闭 WPS
    try:
        wps_app.Quit()
    except Exception:
        pass

    print(f"\n完成: 成功 {success}, 失败 {failed}")
    if args.delete_doc:
        print("原 .doc 文件已删除。")


if __name__ == "__main__":
    main()
