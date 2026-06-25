"""Lightweight article-number helpers shared by citation code.

Keep this module free of document-loader or ML dependencies so citation tests
and API formatting do not import sentence-transformers/torch.
"""

from __future__ import annotations

import re

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}

ARTICLE_PATTERN = re.compile(
    r"第([零一二三四五六七八九十百千万0-9]+)条(?:之([零一二三四五六七八九十]+))?"
)


def chinese_num_to_int(cn: str) -> int:
    """Convert Chinese article numerals such as '二百六十四' to int."""
    if not cn:
        return 0
    if cn.isdigit():
        return int(cn)

    digit_map = {**{str(i): i for i in range(10)}, **{c: i for c, i in _CN_DIGITS.items() if i < 10}}
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
    result += cur_num
    return result
