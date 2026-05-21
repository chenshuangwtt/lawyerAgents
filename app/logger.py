"""
日志配置模块：统一管理项目日志格式和级别。

用法：
    # run.py 启动时调用一次
    from app.logger import setup_logging
    setup_logging()

    # 各模块中使用
    import logging
    logger = logging.getLogger(__name__)
    logger.info("服务启动")
    logger.warning("配置缺失: %s", key)
    logger.error("连接失败", exc_info=True)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    """初始化全局日志配置，级别从 .env 的 LOG_LEVEL 读取。"""
    root = logging.getLogger()
    if root.handlers:
        return  # 避免重复初始化

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # stdout 输出
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件输出（自动轮转，最大 10MB，保留 3 个备份）
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
