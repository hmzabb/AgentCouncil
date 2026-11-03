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
from agents.schemas import AgentStatus  # noqa: E402
from graph.builder import build_council_graph  # noqa: E402

RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"

# 五档评级（决策一致性的统计空间）
_RATINGS: tuple[str, ...] = ("买入", "增持", "中性", "减持", "卖出")


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

    return {
        "runs": n,
        "query": runs[0]["query"],
        "consistency": {
            "rating_counts": rating_counts,
            "mode_rating": mode_rating,
            "consistency_ratio": round(rating_counts[mode_rating] / n, 3),
            "confidence_mean": round(statistics.mean(confidences), 3),
            "confidence_stdev": round(statistics.stdev(confidences), 3) if n > 1 else 0.0,
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
    }


def main(argv: list[str] | None = None) -> int:
    """执行 N 次真实运行，存档并打印一致性 + 延迟汇总。"""
    parser = argparse.ArgumentParser(description="AgentCouncil 真实评测：决策一致性 + 延迟")
    parser.add_argument("--runs", type=int, default=5, help="运行次数（默认 5）")
    parser.add_argument("--query", default="600519.SH", help="评测标的（默认 600519.SH）")
    args = parser.parse_args(argv)

    if config_module.CONFIG.mock_mode:
        print("错误：评测必须真实模式（MOCK_MODE=false）", file=sys.stderr)
        return 1
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== AgentCouncil 真实评测：{args.query} × {args.runs} 次（真实数据 + DeepSeek） ===")
    runs: list[dict[str, Any]] = []
    for i in range(1, args.runs + 1):
        print(f"--- 第 {i}/{args.runs} 次运行（真实计时） ---", flush=True)
        artifact: dict[str, Any] = run_once(args.query, i)
        decision = artifact["final_decision"]
        assert decision is not None and decision["status"] == "ok", "运行未产出合法最终决策"
        runs.append(artifact)
        out_path: Path = RUNS_DIR / f"run_{i:03d}.json"
        out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  评级 {decision['rating']}｜置信度 {decision['confidence']}｜"
            f"端到端 {artifact['wall_seconds']}s｜分析师并行 {artifact['stage_seconds']['analyst_parallel']}s｜"
            f"辩论 {artifact['stage_seconds']['debate']}s｜错误留痕 {len(artifact['pipeline_errors'])} 条",
            flush=True,
        )

    metrics: dict[str, Any] = compute_metrics(runs)
    (RUNS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 评测汇总（真实统计） ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n运行产物已保存：{RUNS_DIR}")
    print("=== 评测运行完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
