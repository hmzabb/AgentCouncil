"""延迟基线快照：记录当前系统在 Mock 模式的各阶段耗时基线。

用途：
- CI 中每次 PR 执行一次基线对比，延迟增加超过阈值时告警
- 检测 LLM prompt 增长或工具调用变慢导致的性能回退
- 支持真模对比：分别保存 mock/real 两种模式的基线

用法：
    python -m eval.baseline_capture             # Mock 模式基线（零 LLM 成本）
    python -m eval.baseline_capture --mode real  # 真实模式基线（需真实配置）
    python -m eval.baseline_capture --compare    # 对比当前运行与已存基线
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

BASELINE_DIR: Path = Path(__file__).resolve().parent / "baselines"
BASELINE_FILE: Path = BASELINE_DIR / "latency_baseline.json"

# 默认标的和运行次数（基线需要足够的统计样本）
DEFAULT_TICKER: str = "600519.SH"
DEFAULT_RUNS: int = 5
# 延迟回退告警阈值（%）：超过此值触发 warning
DEGRADATION_WARNING_PCT: float = 20.0


def _capture_mock_baseline(ticker: str, runs: int) -> dict[str, Any]:
    """在 Mock 模式下运行 N 次并记录各阶段耗时。"""
    from graph.builder import build_council_graph  # noqa: E402

    os.environ["MOCK_MODE"] = "true"
    import config as cfg  # noqa: E402
    cfg.reload_config()

    latencies: list[dict[str, float]] = []
    for i in range(runs):
        graph = build_council_graph()
        thread_id = f"baseline-mock-{i:03d}-{uuid.uuid4().hex[:6]}"
        graph_config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        for _ in graph.stream({"query": ticker}, config=graph_config, stream_mode="updates"):
            pass
        wall = time.perf_counter() - t0

        latencies.append({
            "run": i + 1,
            "wall_seconds": round(wall, 3),
        })

    all_wall = [item["wall_seconds"] for item in latencies]
    return {
        "mode": "mock",
        "ticker": ticker,
        "runs": runs,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "wall_mean": round(statistics.mean(all_wall), 3),
            "wall_stdev": round(statistics.stdev(all_wall), 3) if runs > 1 else 0.0,
            "wall_p95": round(sorted(all_wall)[max(0, round(0.95 * runs) - 1)], 3),
            "wall_min": round(min(all_wall), 3),
            "wall_max": round(max(all_wall), 3),
        },
        "samples": latencies,
    }


def _capture_real_baseline(ticker: str, runs: int) -> dict[str, Any]:
    """在真实模式下运行 N 次并记录各阶段耗时。"""
    os.environ["MOCK_MODE"] = "false"
    import config as cfg  # noqa: E402
    cfg.reload_config()

    if cfg.CONFIG.mock_mode:
        return {"error": "真实模式不可用（检查 .env 配置）"}

    from graph.builder import build_council_graph  # noqa: E402

    latencies: list[dict[str, float]] = []
    for i in range(runs):
        graph = build_council_graph()
        thread_id = f"baseline-real-{i:03d}-{uuid.uuid4().hex[:6]}"
        graph_config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        last = t0
        for chunk in graph.stream({"query": ticker}, config=graph_config, stream_mode="updates"):
            now = time.perf_counter()
            node_name = "+".join(sorted(chunk.keys()))
            stages[node_name] = round(now - last, 3)
            last = now
        wall = time.perf_counter() - t0

        latencies.append({
            "run": i + 1,
            "wall_seconds": round(wall, 3),
            "stages": stages,
        })

    all_wall = [item["wall_seconds"] for item in latencies]
    return {
        "mode": "real",
        "ticker": ticker,
        "runs": runs,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "wall_mean": round(statistics.mean(all_wall), 3),
            "wall_stdev": round(statistics.stdev(all_wall), 3) if runs > 1 else 0.0,
            "wall_p95": round(sorted(all_wall)[max(0, round(0.95 * runs) - 1)], 3),
            "wall_min": round(min(all_wall), 3),
            "wall_max": round(max(all_wall), 3),
        },
        "samples": latencies,
    }


def _compare_with_baseline(current: dict[str, Any]) -> dict[str, Any]:
    """对比当前指标与已保存基线的差异。"""
    if not BASELINE_FILE.is_file():
        return {"error": f"基线文件不存在: {BASELINE_FILE}，请先运行 baseline_capture"}
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))

    b_summary = baseline.get("summary", {})
    c_summary = current.get("summary", {})
    b_mean = b_summary.get("wall_mean", 0)
    c_mean = c_summary.get("wall_mean", 0)

    if b_mean <= 0:
        return {"error": "基线数据无效"}

    change_pct = round((c_mean - b_mean) / b_mean * 100, 1)
    degraded = change_pct > DEGRADATION_WARNING_PCT

    return {
        "baseline_mean_s": b_mean,
        "current_mean_s": c_mean,
        "change_pct": change_pct,
        "degraded": degraded,
        "warning": (
            f"延迟回退 {change_pct}% > 阈值 {DEGRADATION_WARNING_PCT}%："
            f"请检查最近 prompt/依赖/工具调用变化"
        ) if degraded else None,
        "baseline_captured_at": baseline.get("captured_at", "N/A"),
        "current_captured_at": current.get("captured_at", "N/A"),
        "threshold_pct": DEGRADATION_WARNING_PCT,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="延迟基线捕获与对比")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="运行模式")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="标的")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="运行次数")
    parser.add_argument("--compare", action="store_true", help="对比当前与已存基线")
    args = parser.parse_args()

    print(f"=== 延迟基线捕获（{args.mode} 模式，{args.ticker} × {args.runs} 次） ===")

    if args.mode == "mock":
        current = _capture_mock_baseline(args.ticker, args.runs)
    else:
        current = _capture_real_baseline(args.ticker, args.runs)

    if "error" in current:
        print(f"错误: {current['error']}", file=sys.stderr)
        return 1

    s = current["summary"]
    print(
        f"\n{args.mode.upper()} 基线: 均值 {s['wall_mean']}s | "
        f"P95 {s['wall_p95']}s | 标准差 {s['wall_stdev']}s | "
        f"范围 {s['wall_min']}-{s['wall_max']}s"
    )

    if not args.compare:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n基线已保存: {BASELINE_FILE}")
    else:
        comparison = _compare_with_baseline(current)
        print("\n=== 基线对比 ===")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        if comparison.get("degraded"):
            print(f"\n⚠️ 警告: {comparison['warning']}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())