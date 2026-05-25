"""
检索质量评估脚本：对分类模块进行批量测试，输出结构化报告。

用法:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --verbose
    python scripts/eval_retrieval.py --output report.txt
    python scripts/eval_retrieval.py --test-cases data/eval/test_cases.json --verbose
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.classifier import classify_question


def create_llm(settings: Settings):
    """根据配置创建 LLM 实例。"""
    if settings.llm_provider == "qwen":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.qwen_chat_model,
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    elif settings.llm_provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.deepseek_chat_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif settings.llm_provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    else:
        raise ValueError(f"不支持的 LLM 提供商: {settings.llm_provider}")


def load_test_cases(path: str) -> list:
    """加载测试用例 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_cases: list, llm, verbose: bool = False) -> dict:
    """
    对每个测试用例执行分类评估。

    Returns:
        {
            "results": [...],
            "total": int,
            "correct": int,
            "total_time_ms": float,
            "method_counts": {"keyword": n, "llm": n, ...},
            "domain_stats": {domain: {"correct": n, "total": n}},
            "tag_stats": {tag: {"correct": n, "total": n}},
            "failures": [...],
        }
    """
    results = []
    correct = 0
    total_time = 0.0
    method_counts = defaultdict(int)
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    tag_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    failures = []

    for case in test_cases:
        case_id = case["id"]
        question = case["question"]
        expected_domain = case["expected_domain"]
        tags = case.get("tags", [])

        start = time.perf_counter()
        try:
            prediction = classify_question(llm, question)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result = {
                "id": case_id,
                "question": question,
                "expected_domain": expected_domain,
                "predicted_domain": "ERROR",
                "method": "error",
                "confidence": 0.0,
                "is_correct": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "error": str(e),
            }
            results.append(result)
            failures.append(result)
            total_time += elapsed_ms
            domain_stats[expected_domain]["total"] += 1
            for tag in tags:
                tag_stats[tag]["total"] += 1
            if verbose:
                print(f"  [ERR] #{case_id} 预期:{expected_domain} 错误: {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted_domain = prediction["domain"]
        method = prediction["method"]
        is_correct = predicted_domain == expected_domain

        if is_correct:
            correct += 1

        total_time += elapsed_ms
        method_counts[method] += 1

        # 按领域统计
        domain_stats[expected_domain]["total"] += 1
        if is_correct:
            domain_stats[expected_domain]["correct"] += 1

        # 按标签统计
        for tag in tags:
            tag_stats[tag]["total"] += 1
            if is_correct:
                tag_stats[tag]["correct"] += 1

        result = {
            "id": case_id,
            "question": question,
            "expected_domain": expected_domain,
            "predicted_domain": predicted_domain,
            "method": method,
            "confidence": prediction.get("confidence", 0.0),
            "is_correct": is_correct,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        results.append(result)

        if not is_correct:
            failures.append(result)

        if verbose:
            status = "OK" if is_correct else "FAIL"
            print(
                f"  [{status}] #{case_id} "
                f"预期:{expected_domain} 实际:{predicted_domain} "
                f"方法:{method} 耗时:{elapsed_ms:.0f}ms"
            )

    return {
        "results": results,
        "total": len(test_cases),
        "correct": correct,
        "total_time_ms": total_time,
        "method_counts": dict(method_counts),
        "domain_stats": dict(domain_stats),
        "tag_stats": dict(tag_stats),
        "failures": failures,
    }


def format_report(eval_result: dict) -> str:
    """生成可读的评估报告文本。"""
    total = eval_result["total"]
    correct = eval_result["correct"]
    accuracy = correct / total * 100 if total else 0
    avg_time = eval_result["total_time_ms"] / total if total else 0
    method_counts = eval_result["method_counts"]

    lines = []
    lines.append("=" * 50)
    lines.append("检索质量评估报告")
    lines.append("=" * 50)
    lines.append(f"总用例数: {total}")
    lines.append(f"分类准确率: {correct}/{total} ({accuracy:.1f}%)")
    lines.append(f"平均分类耗时: {avg_time:.0f}ms")
    lines.append(
        "方法分布: "
        + ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items()))
    )
    lines.append("")

    # 按领域
    lines.append("按领域:")
    domain_stats = eval_result["domain_stats"]
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        d_total = stats["total"]
        d_correct = stats["correct"]
        d_acc = d_correct / d_total * 100 if d_total else 0
        lines.append(f"  {domain}: {d_correct}/{d_total} ({d_acc:.1f}%)")
    lines.append("")

    # 按用例类型
    lines.append("按用例类型:")
    tag_stats = eval_result["tag_stats"]
    tag_order = ["single-domain", "dual-domain", "three-domain", "four-domain", "case-retrieval"]
    for tag in tag_order:
        if tag not in tag_stats:
            continue
        stats = tag_stats[tag]
        t_total = stats["total"]
        t_correct = stats["correct"]
        t_acc = t_correct / t_total * 100 if t_total else 0
        lines.append(f"  {tag}: {t_correct}/{t_total} ({t_acc:.1f}%)")
    lines.append("")

    # 失败用例
    failures = eval_result["failures"]
    if failures:
        lines.append("分类失败用例:")
        for f in failures:
            q_short = f["question"][:30] + "..." if len(f["question"]) > 30 else f["question"]
            lines.append(
                f'  #{f["id"]} "{q_short}" '
                f"-> 预期:{f['expected_domain']}, "
                f"实际:{f['predicted_domain']}, "
                f"方法:{f['method']}"
            )
    else:
        lines.append("分类失败用例: 无")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="检索质量评估脚本")
    parser.add_argument(
        "--test-cases",
        default=str(ROOT / "data" / "eval" / "test_cases.json"),
        help="测试用例 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出文件路径（可选）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="逐条输出每个用例的分类结果",
    )
    args = parser.parse_args()

    # 加载测试用例
    test_cases = load_test_cases(args.test_cases)
    print(f"已加载 {len(test_cases)} 个测试用例")

    # 初始化 LLM
    settings = Settings()
    print(f"LLM 提供商: {settings.llm_provider}")
    llm = create_llm(settings)

    # 执行评估
    print("开始评估...\n")
    eval_result = run_evaluation(test_cases, llm, verbose=args.verbose)

    # 生成报告
    report = format_report(eval_result)
    print("\n" + report)

    # 保存报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
