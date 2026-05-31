"""
日志配置模块：统一管理项目日志格式和级别。

支持两种格式：
  - text：人类可读的控制台格式（默认）
  - json：结构化 JSON 格式（生产环境推荐，便于 ELK/Loki 采集）

环境变量：
  LOG_LEVEL  - 日志级别（DEBUG/INFO/WARNING/ERROR），默认 INFO
  LOG_FORMAT - 格式类型（text/json），默认 text

用法：
    from app.logger import setup_logging
    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info("服务启动")
"""

import json as _json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式，便于 ELK/Loki 等日志系统采集。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        return _json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """初始化全局日志配置。"""
    root = logging.getLogger()
    if root.handlers:
        return  # 避免重复初始化

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    log_format = os.getenv("LOG_FORMAT", "text").lower()

    if log_format == "json":
        formatter = JSONFormatter()
    else:
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
    # 文件始终用 JSON 格式（便于后续解析）
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)
