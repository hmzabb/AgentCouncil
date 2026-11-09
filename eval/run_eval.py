"""真实评测驱动器：同 ticker 重复运行 N 次，统计决策一致性 + 延迟分布。

设计（所有数字来自真实运行，禁止编造）：
- 固定 MOCK_MODE=false：完整真实链路（AkShare 行情/财务/新闻 + Milvus
  RAG + DeepSeek LLM），N 次串行运行（串行保证延迟样本互不干扰）；
- 每次运行产物完整存档到 eval/runs/run_NNN.json（三分析师报告 / 逐轮
  辩论 / 风控 / 最终决策 / 分超步计时），供 judge.py 评审与
  render_report.py 渲染，评测可复现、可复核；
- 计时粒度：整次 wall clock + 分超步（superstep）墙钟；
  分析师阶段取三分支真实并行墙钟（首分支开始 → 末分支结束）；
- 指标：评级分布与一致率、置信度均值/标准差、端到端 P95/均值/最大、
  分阶段均值、引用来源比例（确定性幻觉率下界）等。

用法（项目根目录）：
    python -m eval.run_eval              # 默认 5 次，query=600519.SH
    python -m eval.run_eval --runs 3 --query 000001.SZ
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 固定真实模式：须在导入 config 前设置
os.environ["MOCK_MODE"] = "false"

import config as config_module  # noqa: E402
from agents.llm import enable_token_tracking as _enable_llm_tracking  # noqa: E402
from agents.llm import get_and_reset_token_records as _get_llm_token_records  # noqa: E402
from graph.builder import build_council_graph  # noqa: E402

RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"

# 五档评级（决策一致性的统计空间）
_RATINGS: tuple[str, ...] = ("买入", "增持", "中性", "减持", "卖出")

# DeepSeek 定价（deepseek-chat，2026 Q3）
# 参考：https://platform.deepseek.com/api-docs/pricing
_DEEPSEEK_INPUT_PRICE_PER_1M: float = 0.27  # USD / 1M input tokens（缓存命中 0.07）
_DEEPSEEK_OUTPUT_PRICE_PER_1M: float = 1.10  # USD / 1M output tokens


def _dump(obj: Any) -> Any:
    """Pydantic 模型 → JSON 安全 dict。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def run_once(query: str, run_id: int) -> dict[str, Any]:
    """运行一次完整链路并记录分超步计时，返回运行产物 dict。"""
    graph = build_council_graph()
    thread_id: str = f"eval-{run_id:03d}-{uuid.uuid4().hex[:6]}"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # 开启 LLM token 追踪（本运行内所有 structured_call 累积记录）
    _enable_llm_tracking()

    # 记录每个超步的 [开始, 结束] 墙钟（相对运行起点）
    chunks: list[tuple[str, float, float]] = []
    t0: float = time.perf_counter()
    last: float = t0
    for chunk in graph.stream({"query": query}, config=config, stream_mode="updates"):
        now: float = time.perf_counter()
        chunks.append(("+".join(sorted(chunk.keys())), last, now))
        last = now
    wall: float = time.perf_counter() - t0
    final_state: dict[str, Any] = graph.get_state(config).values

    # 收集本运行全部 LLM 调用的 token 用量
    token_records: list[tuple[str, dict[str, int]]] = _get_llm_token_records()

    stages: list[dict[str, Any]] = [
        {"node": name, "seconds": round(end - start, 3)} for name, start, end in chunks
    ]
    # 分析师并行阶段墙钟：首分支开始 → 末分支结束（三分支真实并发窗口）
    analyst_span = [c for c in chunks if c[0] == "analyst"]
    analyst_stage = (
        round(max(c[2] for c in analyst_span) - min(c[1] for c in analyst_span), 3)
        if analyst_span else 0.0
    )
    debate_stage = round(sum(c[2] - c[1] for c in chunks if c[0] == "researcher_debate"), 3)
    risk_stage = round(sum(c[2] - c[1] for c in chunks if c[0] == "risk_review"), 3)
    manager_stage = round(sum(c[2] - c[1] for c in chunks if c[0] == "manager_summarize"), 3)

    decision = final_state.get("final_decision")
    return {
        "run_id": run_id,
        "query": query,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 3),
        "stages": stages,
        "stage_seconds": {
            "analyst_parallel": analyst_stage,
            "debate": debate_stage,
            "risk_review": risk_stage,
            "manager_summarize": manager_stage,
            "overhead": round(wall - analyst_stage - debate_stage - risk_stage - manager_stage, 3),
        },
        "analyst_reports": [_dump(r) for r in final_state.get("analyst_reports", [])],
        "debate_history": [_dump(h) for h in final_state.get("debate_history", [])],
        "debate_report": _dump(final_state.get("debate_report")),
        "risk_report": _dump(final_state.get("risk_report")),
        "final_decision": _dump(decision),
        "pipeline_errors": list(final_state.get("errors", [])),
        "token_usage": {
            "call_count": len(token_records),
            "total_prompt_tokens": sum(u["prompt_tokens"] for _, u in token_records),
            "total_completion_tokens": sum(u["completion_tokens"] for _, u in token_records),
            "total_tokens": sum(u["total_tokens"] for _, u in token_records),
            "by_schema": [
                {"schema": s, **u}
                for s, u in token_records
            ],
        },
    }


def compute_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """由全部运行产物计算一致性 / 延迟 / 确定性质量指标（真实统计）。"""
    ratings: list[str] = [r["final_decision"]["rating"] for r in runs]
    rating_counts: dict[str, int] = {k: ratings.count(k) for k in _RATINGS if ratings.count(k)}
    mode_rating: str = max(rating_counts, key=lambda k: rating_counts[k])
    confidences: list[float] = [float(r["final_decision"]["confidence"]) for r in runs]

    totals: list[float] = [float(r["wall_seconds"]) for r in runs]
    totals_sorted: list[float] = sorted(totals)
    n: int = len(totals)
    p95: float = totals_sorted[min(n - 1, max(0, round(0.95 * n) - 1))]  # n 小时取上界样本

    def stage_stats(key: str) -> dict[str, Any]:
        values = [float(r["stage_seconds"][key]) for r in runs]
        return {"mean": round(statistics.mean(values), 3), "max": round(max(values), 3)}

    reports = [rep for r in runs for rep in r["analyst_reports"]]
    cited = [rep for rep in reports if rep["sources"]]
    grounded_rounds = [
        h for r in runs for h in r["debate_history"]
        if h["bull_argument"] and h["bear_argument"] and h["bull_argument"] != h["bear_argument"]
    ]
    total_rounds = [h for r in runs for h in r["debate_history"]]

    # 置信度变化分析：检测 LLM 输出是否过于确定（temperature 过低或输入同质）
    conf_stdev: float = round(statistics.stdev(confidences), 3) if n > 1 else 0.0
    conf_range: float = round(max(confidences) - min(confidences), 3)
    confidence_frozen: bool = conf_stdev == 0.0 and n > 1
    # 各角色分析师置信度变化（跨运行）：separate 各角色统计
    analyst_roles: dict[str, list[float]] = {}
    for r in runs:
        for rep in r["analyst_reports"]:
            analyst_roles.setdefault(rep["role"], []).append(float(rep["confidence"]))
    analyst_conf_stats: dict[str, dict[str, float]] = {}
    for role, vals in analyst_roles.items():
        if len(vals) > 1:
            analyst_conf_stats[role] = {
                "mean": round(statistics.mean(vals), 3),
                "stdev": round(statistics.stdev(vals), 3),
                "range": round(max(vals) - min(vals), 3),
            }
        else:
            analyst_conf_stats[role] = {
                "mean": round(vals[0], 3), "stdev": 0.0, "range": 0.0,
            }

    return {
        "runs": n,
        "query": runs[0]["query"],
        "consistency": {
            "rating_counts": rating_counts,
            "mode_rating": mode_rating,
            "consistency_ratio": round(rating_counts[mode_rating] / n, 3),
            "confidence_mean": round(statistics.mean(confidences), 3),
            "confidence_stdev": conf_stdev,
            "confidence_range": conf_range,
            "confidence_frozen": confidence_frozen,
            "confidence_frozen_warning": (
                "WARNING: 所有运行置信度完全相同。可能原因：1) LLM temperature 过低（当前 0.2）"
                "；2) 同标的短时间内行情/财务数据无变化导致 LLM 输出确定性过高；"
                "3) 经理综合置信度的变化被截断（如 round 到 0.01 精度不足）。"
                "建议：提高 temperature 至 0.4~0.6 以引入合理变化，或扩大评测时间窗口。"
            ) if confidence_frozen else None,
            "analyst_confidence_by_role": analyst_conf_stats,
        },
        "latency_seconds": {
            "mean": round(statistics.mean(totals), 3),
            "p95": round(p95, 3),
            "min": round(min(totals), 3),
            "max": round(max(totals), 3),
            "samples": totals,
            "by_stage": {key: stage_stats(key) for key in ("analyst_parallel", "debate", "risk_review", "manager_summarize")},
        },
        "deterministic_quality": {
            "citation_ratio": round(len(cited) / len(reports), 3) if reports else 0.0,
            "debate_both_sides_ratio": round(len(grounded_rounds) / len(total_rounds), 3) if total_rounds else 0.0,
            "pipeline_error_count": sum(len(r["pipeline_errors"]) for r in runs),
        },
        "token_cost": _compute_token_metrics(runs),
    }


def _compute_token_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """由全部运行的 token_usage 计算用量与成本。"""
    all_prompt: list[int] = [r["token_usage"]["total_prompt_tokens"] for r in runs]
    all_completion: list[int] = [r["token_usage"]["total_completion_tokens"] for r in runs]
    all_total: list[int] = [r["token_usage"]["total_tokens"] for r in runs]
    call_counts: list[int] = [r["token_usage"]["call_count"] for r in runs]

    def _cost(prompt: int, completion: int) -> float:
        return round(
            prompt / 1_000_000 * _DEEPSEEK_INPUT_PRICE_PER_1M
            + completion / 1_000_000 * _DEEPSEEK_OUTPUT_PRICE_PER_1M,
            4,
        )

    per_run_cost: list[float] = [_cost(p, c) for p, c in zip(all_prompt, all_completion, strict=False)]
    return {
        "per_run": {
            "prompt_tokens": {
                "mean": round(statistics.mean(all_prompt), 1),
                "stdev": round(statistics.stdev(all_prompt), 1) if len(all_prompt) > 1 else 0.0,
                "min": min(all_prompt), "max": max(all_prompt),
            },
            "completion_tokens": {
                "mean": round(statistics.mean(all_completion), 1),
                "stdev": round(statistics.stdev(all_completion), 1) if len(all_completion) > 1 else 0.0,
                "min": min(all_completion), "max": max(all_completion),
            },
            "total_tokens": {
                "mean": round(statistics.mean(all_total), 1),
                "stdev": round(statistics.stdev(all_total), 1) if len(all_total) > 1 else 0.0,
            },
            "call_count": {
                "mean": round(statistics.mean(call_counts), 1),
                "range": f"{min(call_counts)}-{max(call_counts)}",
            },
            "cost_usd": {
                "mean": round(statistics.mean(per_run_cost), 4),
                "total": round(sum(per_run_cost), 4),
                "samples": per_run_cost,
            },
        },
        "pricing_note": (
            "DeepSeek deepseek-chat (2026 Q3): $0.27/1M input, $1.10/1M output "
            "(缓存命中 $0.07/1M input)。"
            "来源: https://platform.deepseek.com/api-docs/pricing"
        ),
    }


def _cross_ticker_summary(all_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """跨标的交叉汇总：对比不同标的一致率、置信度、延迟模式。"""
    ticker_meta: list[dict[str, Any]] = []
    all_conf_stdevs: list[float] = []
    all_conf_means: list[float] = []
    all_latency_means: list[float] = []

    for m in all_metrics:
        c = m["consistency"]
        lat = m["latency_seconds"]
        ticker_meta.append({
            "query": m["query"],
            "runs": m["runs"],
            "mode_rating": c["mode_rating"],
            "consistency_ratio": c["consistency_ratio"],
            "confidence_mean": c["confidence_mean"],
            "confidence_stdev": c["confidence_stdev"],
            "confidence_frozen": c.get("confidence_frozen", False),
            "latency_mean": lat["mean"],
            "pipeline_error_count": m["deterministic_quality"]["pipeline_error_count"],
        })
        all_conf_stdevs.append(c["confidence_stdev"])
        all_conf_means.append(c["confidence_mean"])
        all_latency_means.append(lat["mean"])

    return {
        "cross_ticker_summary": {
            "n_tickers": len(ticker_meta),
            "tickers": ticker_meta,
            "overall_consistency": {
                "mean_consistency_ratio": round(
                    sum(t["consistency_ratio"] for t in ticker_meta) / len(ticker_meta), 3
                ) if ticker_meta else 0,
                "frozen_confidence_count": sum(
                    1 for t in ticker_meta if t["confidence_frozen"]
                ),
            },
            "confidence_variation": {
                "mean_of_means": round(statistics.mean(all_conf_means), 3) if all_conf_means else 0,
                "mean_of_stdevs": round(statistics.mean(all_conf_stdevs), 3) if all_conf_stdevs else 0,
                "max_inter_ticker_stdev": round(statistics.stdev(all_conf_means), 3) if len(all_conf_means) > 1 else 0,
                "note": "跨标的之间置信度均值差异越大，说明模型对不同行业/基本面有区分度",
            },
            "latency_cross": {
                "mean_per_ticker": round(statistics.mean(all_latency_means), 1) if all_latency_means else 0,
                "range": (
                    f"{round(min(all_latency_means), 1)}-{round(max(all_latency_means), 1)}"
                    if all_latency_means else "N/A"
                ),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    """执行 N 次真实运行，存档并打印一致性 + 延迟汇总。"""
    parser = argparse.ArgumentParser(description="AgentCouncil 真实评测：决策一致性 + 延迟")
    parser.add_argument("--runs", type=int, default=5, help="运行次数（默认 5）")
    parser.add_argument("--query", default="600519.SH", help="评测标的（默认 600519.SH）")
    parser.add_argument(
        "--queries-file",
        default=None,
        help="多标的评测文件（JSON 数组，如 eval/queries.json）。指定时 --query 和 --runs 对每个标的分别生效。",
    )
    args = parser.parse_args(argv)

    if config_module.CONFIG.mock_mode:
        print("错误：评测必须真实模式（MOCK_MODE=false）", file=sys.stderr)
        return 1
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 确定标的列表
    tickers: list[str]
    if args.queries_file:
        qf_path: Path = Path(args.queries_file)
        if not qf_path.is_absolute():
            qf_path = Path.cwd() / qf_path
        if not qf_path.is_file():
            print(f"错误：标文件不存在：{qf_path}", file=sys.stderr)
            return 1
        tickers = json.loads(qf_path.read_text(encoding="utf-8"))
        if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
            print("错误：标文件需为 JSON 字符串数组，如 [\"600519.SH\", \"000001.SZ\"]", file=sys.stderr)
            return 1
    else:
        tickers = [args.query]

    all_metrics: list[dict[str, Any]] = []
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"=== 评测标的：{ticker} × {args.runs} 次 ===")
        print(f"{'='*60}")
        runs: list[dict[str, Any]] = []
        for i in range(1, args.runs + 1):
            print(f"--- 第 {i}/{args.runs} 次运行（真实计时） ---", flush=True)
            artifact: dict[str, Any] = run_once(ticker, i)
            decision = artifact["final_decision"]
            if decision is None or decision.get("status") != "ok":
                print(f"  警告：第 {i} 次运行未产出合法最终决策，跳过", file=sys.stderr)
                continue
            runs.append(artifact)
            out_path: Path = RUNS_DIR / f"run_{len(runs):03d}.json"
            out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"  评级 {decision['rating']}｜置信度 {decision['confidence']}｜"
                f"端到端 {artifact['wall_seconds']}s｜分析师并行 {artifact['stage_seconds']['analyst_parallel']}s｜"
                f"辩论 {artifact['stage_seconds']['debate']}s｜错误留痕 {len(artifact['pipeline_errors'])} 条｜"
                f"Tokens {artifact['token_usage']['total_tokens']} 个",
                flush=True,
            )

        if not runs:
            print(f"  跳过 {ticker}：无有效运行", file=sys.stderr)
            continue

        metrics: dict[str, Any] = compute_metrics(runs)
        # 多标的情况下按标的单独存档
        if len(tickers) > 1:
            safe_name = ticker.replace(".", "_")
            (RUNS_DIR / f"metrics_{safe_name}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        all_metrics.append(metrics)
        print(f"\n--- {ticker} 评测摘要 ---")
        c = metrics["consistency"]
        lat = metrics["latency_seconds"]
        print(f"  评级分布 {c['rating_counts']}，一致率 {c['consistency_ratio']:.0%}")
        print(f"  置信度均值 {c['confidence_mean']}（标准差 {c['confidence_stdev']}）")
        print(f"  端到端均值 {lat['mean']}s（P95 {lat['p95']}s）")

    # 汇总存档
    if len(all_metrics) > 1:
        # 跨标的交叉汇总
        cross = _cross_ticker_summary(all_metrics)
        (RUNS_DIR / "metrics.json").write_text(
            json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n{'='*60}")
        print("=== 跨标的交叉评测汇总 ===")
        print(json.dumps(cross, ensure_ascii=False, indent=2))
    else:
        (RUNS_DIR / "metrics.json").write_text(
            json.dumps(all_metrics[0], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n=== 评测汇总（真实统计） ===")
        print(json.dumps(all_metrics[0], ensure_ascii=False, indent=2))

    print(f"\n运行产物已保存：{RUNS_DIR}")
    print("=== 评测运行完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())