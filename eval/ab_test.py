"""A/B 对比框架：同一批标的、两套配置、重复采样 + 显著性统计。

解决的问题：单次运行只能说明「这一次结果如何」，无法回答
「改了配置（如辩论轮数、工具自采、缓存）到底有没有变好、变化是否只是噪声」。
本脚本用配对采样回答这个问题。

设计要点：
- 配对（paired）：每个样本是同一个 (标的, 重复序号) 在配置 A 与配置 B 下的
  两次运行，只有「同一条件」的两次结果才可比，避免标的难度差异污染结论；
- 重复（repeats）：同配置同标的跑 N 次，量化模型自身随机性
  （LLM 采样导致同输入不同输出），一致性本身就是质量指标；
- 显著性：
  * 方向匹配率（二分指标）→ McNemar 精确检验（手写二项精确双侧 p 值，
    不引入 scipy 依赖，n 较小时仍准确）；
  * 延迟 / 评级距离（连续指标）→ 配对差值 bootstrap 95% 置信区间
    （固定随机种子，结果可复现；CI 跨 0 即视为无可观测差异）；
- 口径一致：方向匹配、评级距离的判定与 eval/golden_test.py 完全相同
  （评级五档有序、方向按 买/增/中/减/卖 三组归类），保证两套评测可比。

用法（项目根目录）：
    # mock 链路（零成本，CI 用；链路真实跑通但数据为确定性占位）
    python -m eval.ab_test --repeats 3 \
        --a debate_max_rounds=1 --b debate_max_rounds=3

    # 真实链路（AkShare + Milvus + DeepSeek，成本高，建议先小样本）
    python -m eval.ab_test --real --repeats 2 --tickers 600519.SH,002594.SZ \
        --a debate_max_rounds=1 --b debate_max_rounds=3

参数：--a / --b 为逗号分隔的 key=value，key 是环境变量名
（见 config.py 与 .env.example，如 debate_max_rounds、risk_veto_rsi_floor）。
结果写入 eval/runs/ab_comparison.json（含原始配对样本，可复核）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from math import comb
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN_FILE: Path = Path(__file__).resolve().parent / "golden_set.json"
RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"

_RATINGS: tuple[str, ...] = ("买入", "增持", "中性", "减持", "卖出")
_RATING_ORDER: dict[str, int] = {r: i for i, r in enumerate(_RATINGS)}
_BOOTSTRAP_SEED: int = 20261108  # 固定种子：bootstrap CI 可复现
_BOOTSTRAP_ROUNDS: int = 10000


# ---------------------------------------------------------------- 配置变体


def _parse_overrides(text: str) -> dict[str, str]:
    """把 `k=v,k=v` 解析成 dict；格式非法直接报错（避免静默跑错配置）。"""
    overrides: dict[str, str] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"配置项格式应为 key=value，收到：{part!r}")
        key, value = part.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _apply_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """应用环境变量覆盖并重载配置，返回被覆盖前的旧值（用于还原）。"""
    import config as config_module

    previous: dict[str, str] = {}
    for key, value in overrides.items():
        if key in os.environ:
            previous[key] = os.environ[key]
        os.environ[key] = value
    config_module.reload_config()
    return previous


def _restore(previous: dict[str, str], overrides: dict[str, str]) -> None:
    """还原环境变量并重载配置（避免变体配置泄漏到后续运行）。"""
    import config as config_module

    for key in overrides:
        if key in previous:
            os.environ[key] = previous[key]
        else:
            os.environ.pop(key, None)
    config_module.reload_config()


# ---------------------------------------------------------------- 单次运行


def _run_once(ticker: str, real_mode: bool) -> dict[str, Any]:
    """运行一次全链路，返回本次运行的确定性观测值。"""
    from agents.llm import enable_token_tracking, get_and_reset_token_records
    from graph.builder import build_council_graph

    graph = build_council_graph()
    thread_id: str = f"ab-{ticker}-{uuid.uuid4().hex[:8]}"
    graph_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if real_mode:
        enable_token_tracking()

    t0: float = time.perf_counter()
    for _ in graph.stream({"query": ticker}, config=graph_config, stream_mode="updates"):
        pass
    wall: float = round(time.perf_counter() - t0, 3)

    state: dict[str, Any] = graph.get_state(graph_config).values
    decision = state.get("final_decision")
    risk = state.get("risk_report")
    debate = state.get("debate_report")
    tokens: int = sum(u["total_tokens"] for _, u in get_and_reset_token_records()) if real_mode else 0

    return {
        "ticker": ticker,
        "status": decision.status.value if decision is not None else "missing",
        "rating": decision.rating if decision is not None else None,
        "confidence": decision.confidence if decision is not None else 0.0,
        "risk_level": risk.risk_level if risk is not None else None,
        "blocked": bool(risk.blocked) if risk is not None else False,
        "debate_rounds": debate.total_rounds if debate is not None else 0,
        "tools_used": sorted(
            {t for r in state.get("analyst_reports", []) for t in (r.tools_used or [])}
        ),
        "wall_seconds": wall,
        "tokens": tokens,
        "pipeline_errors": len(state.get("errors", [])),
    }


def _run_variant(
    name: str, overrides: dict[str, str], tickers: list[str], repeats: int, real_mode: bool
) -> list[dict[str, Any]]:
    """应用变体配置后跑「标的 × 重复」全部样本，再还原配置。"""
    print(f"\n--- 变体 {name}：{overrides or '（默认配置）'} ---", flush=True)
    previous: dict[str, str] = _apply_overrides(overrides)
    samples: list[dict[str, Any]] = []
    try:
        for i in range(repeats):
            for ticker in tickers:
                sample = _run_once(ticker, real_mode)
                sample["variant"] = name
                sample["repeat"] = i
                samples.append(sample)
                print(
                    f"  {name} r{i + 1} {ticker}：评级 {sample['rating']} | "
                    f"风险 {sample['risk_level']} | 轮次 {sample['debate_rounds']} | "
                    f"{sample['wall_seconds']}s | tokens {sample['tokens']}",
                    flush=True,
                )
    finally:
        _restore(previous, overrides)
    return samples


# ---------------------------------------------------------------- 统计工具


def _binom_two_sided(k: int, n: int) -> float:
    """精确二项双侧 p 值（用于 McNemar：零假设 p=0.5）。"""
    if n == 0:
        return 1.0
    probs: list[float] = [comb(n, i) / (2 ** n) for i in range(n + 1)]
    observed: float = probs[k]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def mcnemar_exact(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """McNemar 精确检验：配对二分指标（如方向匹配）在 A/B 下是否显著不同。"""
    b: int = sum(1 for a_ok, b_ok in pairs if a_ok and not b_ok)  # A 对 B 错
    c: int = sum(1 for a_ok, b_ok in pairs if not a_ok and b_ok)  # A 错 B 对
    p: float = _binom_two_sided(min(b, c), b + c)
    return {
        "n_pairs": len(pairs),
        "a_only_correct": b,
        "b_only_correct": c,
        "both_correct": sum(1 for a_ok, b_ok in pairs if a_ok and b_ok),
        "both_wrong": sum(1 for a_ok, b_ok in pairs if not a_ok and not b_ok),
        "p_value": round(p, 4),
        "significant_at_0.05": p < 0.05,
        "test": "McNemar exact (binomial two-sided)",
        "note": "样本量小时 p 值天然偏大，结论需结合置信区间与样本量一起看",
    }


def bootstrap_ci(diffs: list[float]) -> dict[str, Any]:
    """配对差值的 bootstrap 95% 置信区间（固定种子，可复现）。"""
    if not diffs:
        return {"n": 0, "mean_diff": 0.0, "ci95": [0.0, 0.0], "crosses_zero": True}
    rng = random.Random(_BOOTSTRAP_SEED)
    n: int = len(diffs)
    means: list[float] = []
    for _ in range(_BOOTSTRAP_ROUNDS):
        means.append(mean(rng.choices(diffs, k=n)))
    means.sort()
    lo: float = means[int(0.025 * _BOOTSTRAP_ROUNDS)]
    hi: float = means[int(0.975 * _BOOTSTRAP_ROUNDS) - 1]
    return {
        "n": n,
        "mean_diff": round(mean(diffs), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "crosses_zero": lo <= 0.0 <= hi,
        "method": f"paired bootstrap, {_BOOTSTRAP_ROUNDS} resamples, seed={_BOOTSTRAP_SEED}",
    }


def _direction_ok(rating: str | None, expected: str) -> bool:
    """方向匹配口径（与 golden_test 完全一致）。"""
    if rating is None:
        return False
    return (
        (rating in ("买入", "增持") and expected in ("买入", "增持"))
        or (rating == "中性" and expected == "中性")
        or (rating in ("减持", "卖出") and expected in ("减持", "卖出"))
    )


def _describe(values: list[float]) -> dict[str, Any]:
    """均值 / 中位数 / 标准差 / 极值。"""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "std": round(pstdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


# ---------------------------------------------------------------- 汇总分析


def analyze(
    samples_a: list[dict[str, Any]],
    samples_b: list[dict[str, Any]],
    expected: dict[str, str],
) -> dict[str, Any]:
    """对 A/B 配对样本做统计汇总（全部指标均为真实运行结果的聚合）。"""
    key = lambda s: (s["ticker"], s["repeat"])  # noqa: E731 —— 配对键
    map_b: dict[tuple[str, int], dict[str, Any]] = {key(s): s for s in samples_b}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (s, map_b[key(s)]) for s in samples_a if key(s) in map_b
    ]

    direction_pairs: list[tuple[bool, bool]] = [
        (_direction_ok(a["rating"], expected.get(a["ticker"], "中性")),
         _direction_ok(b["rating"], expected.get(b["ticker"], "中性")))
        for a, b in pairs
    ]
    latency_diffs: list[float] = [b["wall_seconds"] - a["wall_seconds"] for a, b in pairs]
    rounds_diffs: list[float] = [b["debate_rounds"] - a["debate_rounds"] for a, b in pairs]

    def consistency(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """同配置重复运行的评级一致率（众数占比）与平均评级距离。"""
        by_ticker: dict[str, list[str]] = {}
        for s in samples:
            if s["rating"]:
                by_ticker.setdefault(s["ticker"], []).append(s["rating"])
        ratios: list[float] = []
        for ratings in by_ticker.values():
            top: int = max(ratings.count(r) for r in set(ratings))
            ratios.append(top / len(ratings))
        return {
            "repeats_per_ticker": sorted({len(v) for v in by_ticker.values()}),
            "rating_agreement_mean": round(mean(ratios), 3) if ratios else 0.0,
            "per_ticker": {
                t: {"ratings": v, "mode_ratio": round(max(v.count(r) for r in set(v)) / len(v), 3)}
                for t, v in by_ticker.items()
            },
        }

    def metric_block(samples: list[dict[str, Any]]) -> dict[str, Any]:
        ratings: list[str | None] = [s["rating"] for s in samples]
        distances: list[float] = [
            abs(_RATING_ORDER.get(s["rating"] or "中性", 2) - _RATING_ORDER.get(expected.get(s["ticker"], "中性"), 2))
            for s in samples
        ]
        return {
            "samples": len(samples),
            "direction_accuracy": round(
                mean(1.0 if _direction_ok(s["rating"], expected.get(s["ticker"], "中性")) else 0.0
                     for s in samples), 3) if samples else 0.0,
            "rating_avg_distance": round(mean(distances), 3) if distances else 0.0,
            "rating_distribution": {
                r: ratings.count(r) for r in _RATINGS if r in ratings
            },
            "confidence": _describe([s["confidence"] for s in samples]),
            "latency_seconds": _describe([s["wall_seconds"] for s in samples]),
            "debate_rounds": _describe([float(s["debate_rounds"]) for s in samples]),
            "tokens": _describe([float(s["tokens"]) for s in samples]),
            "blocked_rate": round(mean(1.0 if s["blocked"] else 0.0 for s in samples), 3) if samples else 0.0,
            "pipeline_error_rate": round(
                mean(1.0 if s["pipeline_errors"] else 0.0 for s in samples), 3) if samples else 0.0,
            "consistency": consistency(samples),
        }

    comparison: dict[str, Any] = {
        "n_pairs": len(pairs),
        "direction_match": mcnemar_exact(direction_pairs),
        "latency_diff_b_minus_a": bootstrap_ci(latency_diffs),
        "debate_rounds_diff_b_minus_a": bootstrap_ci(rounds_diffs),
    }

    verdicts: list[str] = []
    if comparison["direction_match"]["significant_at_0.05"]:
        verdicts.append("方向匹配率存在统计显著差异（McNemar p<0.05）")
    else:
        verdicts.append("方向匹配率无统计显著差异（McNemar p>=0.05 或样本不足）")
    lat: dict[str, Any] = comparison["latency_diff_b_minus_a"]
    if lat["n"] and not lat["crosses_zero"]:
        verdicts.append(
            f"延迟存在显著差异：B 相对 A 均值变化 {lat['mean_diff']}s"
            f"（95%CI [{lat['ci95'][0]}, {lat['ci95'][1]}]）"
        )
    else:
        verdicts.append("延迟无可观测差异（bootstrap 95%CI 跨 0）")
    rounds: dict[str, Any] = comparison["debate_rounds_diff_b_minus_a"]
    if rounds["n"] and not rounds["crosses_zero"]:
        verdicts.append(
            f"辩论轮次随配置变化 {rounds['mean_diff']} 轮"
            f"（95%CI [{rounds['ci95'][0]}, {rounds['ci95'][1]}]）"
        )

    return {
        "variant_a": metric_block(samples_a),
        "variant_b": metric_block(samples_b),
        "comparison": comparison,
        "verdict": verdicts,
        "raw_pairs": [
            {
                "ticker": a["ticker"], "repeat": a["repeat"],
                "a_rating": a["rating"], "b_rating": b["rating"],
                "a_seconds": a["wall_seconds"], "b_seconds": b["wall_seconds"],
                "a_rounds": a["debate_rounds"], "b_rounds": b["debate_rounds"],
                "a_tools": a["tools_used"], "b_tools": b["tools_used"],
            }
            for a, b in pairs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentCouncil A/B 对比（配对采样 + 显著性统计）")
    parser.add_argument("--a", default="debate_max_rounds=1", help="A 组配置，k=v,k=v")
    parser.add_argument("--b", default="debate_max_rounds=3", help="B 组配置，k=v,k=v")
    parser.add_argument("--repeats", type=int, default=3, help="每标的每配置重复次数")
    parser.add_argument("--tickers", default="", help="逗号分隔标的；默认取黄金集前 2 只")
    parser.add_argument("--real", action="store_true", help="真实链路（默认 mock，零成本）")
    parser.add_argument("--output", default="", help="结果文件路径（默认 runs/ab_comparison.json）")
    args = parser.parse_args(argv)

    if args.repeats < 2:
        print("错误：repeats 至少为 2，否则无法量化随机性与配对差异", file=sys.stderr)
        return 1

    # 模式必须在导入 config 前确定（config 导入时加载 .env，dotenv 不覆盖已有变量）
    os.environ["MOCK_MODE"] = "false" if args.real else "true"

    golden: dict[str, Any] = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    expected: dict[str, str] = {q["ticker"]: q["expected_rating"] for q in golden["queries"]}
    tickers: list[str] = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else [q["ticker"] for q in golden["queries"][:2]]
    )

    overrides_a: dict[str, str] = _parse_overrides(args.a)
    overrides_b: dict[str, str] = _parse_overrides(args.b)

    print("=== AgentCouncil A/B 对比（配对采样） ===")
    print(f"模式：{'真实链路（AkShare/Milvus/DeepSeek）' if args.real else 'mock 链路（确定性占位数据）'}")
    print(f"标的：{', '.join(tickers)}　重复：{args.repeats} 次/标的/组")
    print(f"A：{overrides_a}　B：{overrides_b}")

    samples_a: list[dict[str, Any]] = _run_variant("A", overrides_a, tickers, args.repeats, args.real)
    samples_b: list[dict[str, Any]] = _run_variant("B", overrides_b, tickers, args.repeats, args.real)

    report: dict[str, Any] = analyze(samples_a, samples_b, expected)
    report.update({
        "mode": "real" if args.real else "mock",
        "tickers": tickers,
        "repeats": args.repeats,
        "config_a": overrides_a,
        "config_b": overrides_b,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caveat": (
            "mock 链路使用确定性占位数据，方向匹配率/评级距离仅用于验证统计框架本身，"
            "不代表决策质量；质量结论须用 --real 在真实数据链路下得出"
            if not args.real
            else "真实链路结果仍受 LLM 采样随机性影响，重复次数越多统计结论越稳"
        ),
    })

    print("\n=== 结论 ===")
    for line in report["verdict"]:
        print(f"- {line}")
    print(f"- A 方向匹配率 {report['variant_a']['direction_accuracy']}，"
          f"B 方向匹配率 {report['variant_b']['direction_accuracy']}")
    print(f"- A 延迟均值 {report['variant_a']['latency_seconds'].get('mean')}s，"
          f"B 延迟均值 {report['variant_b']['latency_seconds'].get('mean')}s")
    print(f"- A 评级一致率 {report['variant_a']['consistency']['rating_agreement_mean']}，"
          f"B 评级一致率 {report['variant_b']['consistency']['rating_agreement_mean']}")
    print(f"- 说明：{report['caveat']}")

    out_path: Path = Path(args.output) if args.output else RUNS_DIR / "ab_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())