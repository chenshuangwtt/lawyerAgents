"""
法律领域注册表：从 law_registry.yaml 加载领域配置，为 classifier 和前端提供统一数据源。

新增法律只需编辑 law_registry.yaml，无需修改其他文件。
"""

from pathlib import Path
from typing import Dict, List

import yaml


_REGISTRY_PATH = Path(__file__).parent / "law_registry.yaml"


def _load_yaml() -> dict:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_domain_law_map() -> Dict[str, List[str]]:
    """领域 → 法律名称映射，供 classifier 使用。"""
    data = _load_yaml()
    return {d["name"]: d.get("laws", []) for d in data["domains"]}


def load_domain_keywords() -> Dict[str, List[str]]:
    """领域 → 关键词列表，供 classifier fallback 使用。"""
    data = _load_yaml()
    return {d["name"]: d.get("keywords", []) for d in data["domains"]}


def load_classify_prompt_text() -> str:
    """生成分类器 system prompt 文本。"""
    data = _load_yaml()
    domains = data["domains"]
    names = "、".join(d["name"] for d in domains)
    rules = "\n".join(f"- {d['rule']}" for d in domains if d.get("rule"))
    return (
        f"你是一个法律问题分类器。根据用户问题，判断属于哪个法律领域。\n\n"
        f"可选领域：{names}\n\n"
        f"规则：\n"
        f"- 只输出领域名称，不要加任何解释\n"
        f"- 如果涉及多个领域或无法确定，输出\"综合\"\n"
        f"{rules}"
    )


def load_domain_colors() -> Dict[str, str]:
    """领域 → CSS 类名映射，供前端使用。"""
    data = _load_yaml()
    return {d["name"]: d.get("color", "bg-gray-50 text-gray-500 ring-gray-200") for d in data["domains"]}
