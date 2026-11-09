"""黄金集基准评测：对比 AgentCouncil 决策与真实券商研报一致预期。

设计原则：
- 所有预期评级基于市场公开可查的券商研报共识，每条均标注来源与日期；
- 评测不追求绝对「正确」，而是计算 AgentCouncil 与市场共识的偏离度
  （方向一致、距离一致、风险等级一致）；
- A股卖方研报普遍偏乐观（买入/增持为主），因此主要验证方向一致性
  与系统偏差检测能力（如是否存在系统性悲观/乐观偏向）。
- 每标的单次运行（控制成本），合计 5 标的全链路评测。

用法（项目根目录）：
    python -m eval.golden_test
    python -m eval.golden_test --output eval/runs/golden_metrics.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MOCK_MODE"] = "false"

import config as config_module  # noqa: E402
from agents.llm import enable_token_tracking as _enable_llm_tracking  # noqa: E402
from agents.llm import get_and_reset_token_records as _get_llm_token_records  # noqa: E402
from graph.builder import build_council_graph  # noqa: E402

GOLDEN_FILE: Path = Path(__file__).resolve().parent / "golden_set.json"
RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"

_RATINGS: tuple[str, ...] = ("买入", "增持", "中性", "减持", "卖出")
_RATING_ORDER: dict[str, int] = {r: i for i, r in enumerate(_RATINGS)}
_RISK_LEVELS: tuple[str, ...] = ("低", "中", "高")
_RISK_ORDER: dict[str, int] = {r: i for i, r in enumerate(_RISK_LEVELS)}


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def run_single(query: dict[str, Any]) -> dict[str, Any]:
    """对单个标的运行一次全链路。"""
    ticker: str = query["ticker"]
    graph = build_council_graph()
    thread_id: str = f"golden-{ticker}-{uuid.uuid4().hex[:6]}"
    graph_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    _enable_llm_tracking()

    t0: float = time.perf_counter()
    for _ in graph.stream({"query": ticker}, config=graph_config, stream_mode="updates"):
        pass
    wall: float = time.perf_counter() - t0

    final_state: dict[str, Any] = graph.get_state(graph_config).values
    token_records = _get_llm_token_records()
    decision = final_state.get("final_decision")

    return {
        "ticker": ticker,
        "industry": query.get("industry", ""),
        "wall_seconds": round(wall, 3),
        "analyst_reports": [_dump(r) for r in final_state.get("analyst_reports", [])],
        "debate_report": _dump(final_state.get("debate_report")),
        "risk_report": _dump(final_state.get("risk_report")),
        "final_decision": _dump(decision),
        "pipeline_errors": list(final_state.get("errors", [])),
        "token_total": sum(u["total_tokens"] for _, u in token_records),
    }


def compute_golden_metrics(
    results: list[dict[str, Any]], golden: dict[str, Any]
) -> dict[str, Any]:
    """对比 AgentCouncil 输出与黄金集预期。"""
    per_ticker: list[dict[str, Any]] = []
    rating_distances: list[int] = []
    direction_correct: int = 0
    risk_distances: list[int] = []

    for result in results:
        ticker: str = result["ticker"]
        expected = next(
            (q for q in golden["queries"] if q["ticker"] == ticker), None
        )
        if expected is None:
            continue

        decision = result["final_decision"]
        if not decision or decision.get("status") != "ok":
            per_ticker.append({
                "ticker": ticker,
                "error": "AgentCouncil 未产出合法决策",
                "rating_match": False,
            })
            continue

        actual_rating: str = decision["rating"]
        expected_rating: str = expected["expected_rating"]
        actual_risk: str = result.get("risk_report", {}).get("risk_level", "中") if result.get("risk_report") else "中"
        expected_risk: str = expected.get("expected_risk", "中")

        rating_idx: int = _RATING_ORDER.get(actual_rating, 2)
        expected_idx: int = _RATING_ORDER.get(expected_rating, 2)
        distance: int = abs(rating_idx - expected_idx)
        rating_distances.append(distance)

        risk_distance: int = abs(
            _RISK_ORDER.get(str(actual_risk), 1) - _RISK_ORDER.get(str(expected_risk), 1)
        )
        risk_distances.append(risk_distance)

        direction_ok: bool = (
            (actual_rating in ("买入", "增持") and expected_rating in ("买入", "增持"))
            or (actual_rating == "中性" and expected_rating == "中性")
            or (actual_rating in ("减持", "卖出") and expected_rating in ("减持", "卖出"))
        )
        if direction_ok:
            direction_correct += 1

        per_ticker.append({
            "ticker": ticker,
            "industry": expected.get("industry", ""),
            "expected_rating": expected_rating,
            "actual_rating": actual_rating,
            "expected_risk": expected_risk,
            "actual_risk": actual_risk,
            "rating_distance": distance,
            "direction_match": direction_ok,
            "confidence": decision.get("confidence", 0),
            "key_concerns": expected.get("key_concerns", []),
            "pipeline_errors": len(result.get("pipeline_errors", [])),
            "wall_seconds": result.get("wall_seconds", 0),
            "token_total": result.get("token_total", 0),
        })

    n: int = len(rating_distances)
    exact_match: int = sum(1 for d in rating_distances if d == 0)
    avg_distance: float = sum(rating_distances) / n if n > 0 else 0.0

    direction_acc: float = direction_correct / n if n > 0 else 0.0

    avg_risk_dist: float = sum(risk_distances) / len(risk_distances) if risk_distances else 0.0
    risk_exact: int = sum(1 for d in risk_distances if d == 0)
    risk_acc: float = risk_exact / len(risk_distances) if risk_distances else 0.0

    # 系统偏差检测：平均评级偏移方向（正值=偏乐观，负值=偏悲观）
    bias_signs: list[int] = []
    for result in per_ticker:
        if "rating_distance" in result:
            actual_idx = _RATING_ORDER.get(result["actual_rating"], 2)
            expected_idx = _RATING_ORDER.get(result["expected_rating"], 2)
            bias_signs.append(expected_idx - actual_idx)  # 正=Agent更悲观, 负=Agent更乐观
    systemic_bias: float = sum(bias_signs) / len(bias_signs) if bias_signs else 0.0

    return {
        "summary": {
            "n_stocks": n,
            "rating_exact_match": f"{exact_match}/{n}",
            "rating_exact_accuracy": round(exact_match / n, 3) if n else 0,
            "rating_avg_distance": round(avg_distance, 2),
            "rating_direction_accuracy": round(direction_acc, 3),
            "risk_exact_match": f"{risk_exact}/{n}",
            "risk_accuracy": round(risk_acc, 3),
            "risk_avg_distance": round(avg_risk_dist, 2),
            "systemic_bias": round(systemic_bias, 2),
            "systemic_bias_note": (
                "正值=AgentCouncil 评级比券商共识更悲观"
                if systemic_bias > 0
                else "负值=AgentCouncil 评级比券商共识更乐观"
                if systemic_bias < 0
                else "无系统性偏差"
            ),
            "limitation_note": (
                "注意：A股卖方研报普遍偏乐观，黄金集共识并不代表'正确'，"
                "仅作为方向对比基准。系统性偏差可能源于卖方乐观偏差而非模型错误。"
            ),
        },
        "per_ticker": per_ticker,
        "reference_date": golden.get("_reference_date", ""),
    }


def main() -> int:
    if config_module.CONFIG.mock_mode:
        print("错误：黄金集评测必须真实模式（MOCK_MODE=false）", file=sys.stderr)
        return 1

    golden: dict[str, Any] = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = golden["queries"]
    print(f"=== AgentCouncil 黄金集评测：{len(queries)} 只标的（真实券商共识对比） ===")
    print(f"参考日期：{golden.get('_reference_date', 'N/A')}")
    print(f"来源：{'; '.join(golden.get('_sources', [])[:2])}...")
    print()

    results: list[dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        ticker: str = q["ticker"]
        industry: str = q.get("industry", "")
        expected: str = q["expected_rating"]
        print(f"--- [{i}/{len(queries)}] {ticker}（{industry}，预期 {expected}）---", flush=True)
        start = time.perf_counter()
        result = run_single(q)
        elapsed = time.perf_counter() - start
        decision = result["final_decision"]
        print(
            f"  结果: 评级 {decision.get('rating', 'N/A')} | "
            f"置信度 {decision.get('confidence', 'N/A')} | "
            f"风险 {result.get('risk_report', {}).get('risk_level', 'N/A')} | "
            f"耗时 {elapsed:.1f}s | "
            f"Tokens {result['token_total']}",
            flush=True,
        )
        results.append(result)

    metrics = compute_golden_metrics(results, golden)
    print("\n=== 黄金集评测结果 ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out_path: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS_DIR / "golden_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())