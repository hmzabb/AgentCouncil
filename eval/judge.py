"""LLM-as-Judge 评审：角色报告质量（事实性/相关性/无幻觉）+ 辩论有效性。

设计（所有分数来自真实 LLM 评审，禁止编造）：
- 评审模型与主链路同为 DeepSeek（temperature 0.2，structured_call 严格
  校验），评审 prompt 独立于生产 prompt（critic 视角，避免自我偏袒）；
- 报告评审：给定「分析师收到的全部工具数据文本」与报告全文，评委对
  事实性 / 相关性 / 无幻觉打 0~5 分，并逐条判定报告核心论点的
  grounded（是否由给定数据支持或合理推断）——这就是幻觉率的
  论点级度量；
- 辩论评审：给定逐轮多空论点，评委打 多方交锋度 / 空方交锋度 /
  对立覆盖度（是否真覆盖对立观点、非一边倒）0~5 分；
- fail-open：评委调用失败（网络/输出不合法）时记录失败并计 0 分 +
  留痕，不伪造评审结果。

用法：
    python -m eval.judge               # 评审 eval/runs/run_*.json 全部运行
    python -m eval.judge --runs 1 2 3  # 只评审指定运行
输出：eval/runs/judge_results.json + 控制台汇总。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 固定真实模式：评审与分析链路同环境
import os  # noqa: E402

os.environ.setdefault("MOCK_MODE", "false")

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

import config as config_module  # noqa: E402
from agents.analyst import ANALYST_ROLES, AnalystAgent  # noqa: E402
from agents.llm import SYSTEM_ROLE_HINT, structured_call  # noqa: E402
from agents.schemas import AnalystReport, DebateRound  # noqa: E402

RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"

# ------------------------------------------------------------------ 评审 schema


class ClaimVerdict(BaseModel):
    """报告单条核心论点的有据判定。"""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="被评审的核心论点（逐字引用报告）")
    grounded: bool = Field(description="该论点是否由给定数据直接支持或可合理推断")
    reason: str = Field(description="判定理由（指出支持/不支持该论点的数据）")


class ReportQualityVerdict(BaseModel):
    """分析师报告质量评审结果。"""

    model_config = ConfigDict(extra="forbid")

    factuality: int = Field(ge=0, le=5, description="事实性：数据引用是否与给定数据一致（0~5）")
    relevance: int = Field(ge=0, le=5, description="相关性：内容是否紧扣该角色职责与查询标的（0~5）")
    no_hallucination: int = Field(ge=0, le=5, description="无幻觉：有无编造给定数据之外的具体数字/事实（0~5）")
    claims: list[ClaimVerdict] = Field(min_length=1, description="逐条核心论点的有据判定")
    comment: str = Field(description="总体评审意见（中文，50~150 字）")


class DebateQualityVerdict(BaseModel):
    """多空辩论有效性评审结果。"""

    model_config = ConfigDict(extra="forbid")

    bull_engagement: int = Field(ge=0, le=5, description="多方交锋度：是否实质回应/反驳空方而非自说自话（0~5）")
    bear_engagement: int = Field(ge=0, le=5, description="空方交锋度：是否实质回应/反驳多方而非自说自话（0~5）")
    opposing_coverage: int = Field(ge=0, le=5, description="对立覆盖度：多空是否真覆盖对立观点、非一边倒（0~5）")
    comment: str = Field(description="总体评审意见（中文，50~150 字）")


# ------------------------------------------------------------------ 评审 prompt

_REPORT_JUDGE_SYSTEM: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：评测系统评审专家（LLM-as-Judge）。任务：对一份{role}分析师的报告做质量评审。"
    + "\n评审基准：报告必须只基于「给定数据」；明确标注数据缺失不算幻觉。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- factuality: 0~5 整数，事实性（数据引用是否与给定数据一致）；"
    + "\n- relevance: 0~5 整数，相关性（是否紧扣{role}职责与查询标的、不越界）；"
    + "\n- no_hallucination: 0~5 整数，无幻觉（有无编造给定数据之外的具体数字/事实）；"
    + "\n- claims: 数组，逐条评审报告核心论点，每项含 claim(字符串，逐字引用)、"
    + "grounded(布尔，该论点是否由给定数据直接支持或可合理推断)、reason(字符串，判定理由)；"
    + "\n- comment: 字符串，总体评审意见（50~150 字）。"
    + "\n要求：中文；评审严格，不因表述流畅而放宽事实核查；分数必须与评论一致。"
)

_DEBATE_JUDGE_SYSTEM: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：评测系统评审专家（LLM-as-Judge）。任务：评审多轮多空辩论的有效性。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- bull_engagement: 0~5 整数，多方交锋度（是否实质回应/反驳空方观点，而非自说自话）；"
    + "\n- bear_engagement: 0~5 整数，空方交锋度（是否实质回应/反驳多方观点）；"
    + "\n- opposing_coverage: 0~5 整数，对立覆盖度（多空是否真正对立、双方观点都被充分展开，"
    + "不存在一边倒或各说各话）；"
    + "\n- comment: 字符串，总体评审意见（50~150 字）。"
    + "\n要求：中文；评分依据辩论原文，逐轮考察双方是否引用证据回应对方上一轮观点。"
)


# ------------------------------------------------------------------ 评审执行


def _safe_judge(schema_cls: type[Any], system: str, human: str) -> tuple[Any | None, str | None]:
    """fail-open 评审调用：失败返回 (None, 错误信息)，绝不伪造评审结果。"""
    try:
        return structured_call(schema_cls, system=system, human=human), None
    except Exception as exc:  # noqa: BLE001 —— 评审失败留痕，不计入有效分数
        return None, f"{type(exc).__name__}: {exc}"


def judge_report(role: str, report: AnalystReport, data_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """评审一份分析师报告（给定数据 + 报告全文 → 质量分 + 论点有据判定）。"""
    system = _REPORT_JUDGE_SYSTEM.format(role=role)
    human = (
        f"给定数据（分析师收到的全部工具数据）：\n{data_text}\n\n"
        f"分析师报告全文：\n摘要：{report.summary}\n"
        + "".join(f"论点 {i + 1}：{k}\n" for i, k in enumerate(report.key_points))
        + "\n请输出评审 JSON。"
    )
    verdict, error = _safe_judge(ReportQualityVerdict, system=system, human=human)
    if verdict is None:
        return None, error
    return verdict.model_dump(mode="json"), None


def judge_debate(history: list[DebateRound]) -> tuple[dict[str, Any] | None, str | None]:
    """评审多轮辩论的有效性（逐轮多空论点 → 交锋度/对立覆盖度）。"""
    rounds_text = "\n".join(
        f"第 {h.round_no} 轮多方：{h.bull_argument}\n第 {h.round_no} 轮空方：{h.bear_argument}\n"
        for h in history
    )
    human = f"辩论记录：\n{rounds_text}\n请输出评审 JSON。"
    verdict, error = _safe_judge(DebateQualityVerdict, system=_DEBATE_JUDGE_SYSTEM, human=human)
    if verdict is None:
        return None, error
    return verdict.model_dump(mode="json"), None


# ------------------------------------------------------------------ 驱动：加载运行产物并评审


def _collect_data_contexts(query: str) -> dict[str, str]:
    """为每个角色重建「给定数据」文本（与分析链路完全相同的工具取数）。

    当日数据稳定：三个角色各取一次，全部运行共用（运行间隔数分钟内）。
    工具集合使用角色默认集；报告若记录了自己的 tools_used，
    评审时优先用报告自身的工具集重建（见 _context_for_report）。
    """
    contexts: dict[str, str] = {}
    for role in ANALYST_ROLES:
        agent = AnalystAgent(role=role, confidence=0.70)
        context, tool_errors, _sources = agent.collect_context(query)
        text = agent.describe_context(context)
        if tool_errors:
            text += f"\n（工具失败留痕：{'；'.join(tool_errors)}）"
        contexts[role] = text
    return contexts


def _context_for_report(report: Any, role: str, query: str, cache: dict[str, str]) -> str:
    """按报告实际使用的工具集重建「给定数据」文本（未记录则用角色默认集缓存）。"""
    tools: list[str] = list(getattr(report, "tools_used", []) or [])
    if not tools:
        return cache.get(role, "")
    key: str = f"{role}::{','.join(tools)}"
    if key in cache:
        return cache[key]
    agent = AnalystAgent(role=role, confidence=0.70)
    context, tool_errors, _sources = agent.collect_context(query, tools=tools)
    text = agent.describe_context(context)
    if tool_errors:
        text += f"\n（工具失败留痕：{'；'.join(tool_errors)}）"
    cache[key] = text
    return text


def judge_runs(run_ids: list[int], query: str) -> dict[str, Any]:
    """评审指定运行产物，返回汇总结果 dict（评审失败如实留痕）。"""
    data_contexts: dict[str, str] = _collect_data_contexts(query)
    results: dict[str, Any] = {
        "judge_model": config_module.CONFIG.deepseek_model,
        "reports": [],   # 每份报告一条评审
        "debates": [],   # 每次运行一条辩论评审
        "failures": [],
    }
    for run_id in run_ids:
        artifact_path: Path = RUNS_DIR / f"run_{run_id:03d}.json"
        if not artifact_path.exists():
            print(f"跳过 run_{run_id:03d}：产物不存在 {artifact_path}")
            continue
        artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))
        for report in artifact["analyst_reports"]:
            parsed = AnalystReport.model_validate(report)
            verdict, error = judge_report(
                report["role"],
                parsed,
                _context_for_report(parsed, report["role"], query, data_contexts),
            )
            entry: dict[str, Any] = {"run_id": run_id, "role": report["role"]}
            if verdict is None:
                entry["error"] = error
                results["failures"].append(entry)
            else:
                entry.update(verdict)
                results["reports"].append(entry)
            print(
                f"  run{run_id:03d}/{report['role']}："
                + (f"事实性 {verdict['factuality']} 相关性 {verdict['relevance']} "
                   f"无幻觉 {verdict['no_hallucination']} 论点有据 "
                   f"{sum(1 for c in verdict['claims'] if c['grounded'])}/{len(verdict['claims'])}"
                   if verdict else f"评审失败: {error}")
            )
        history: list[DebateRound] = [DebateRound.model_validate(h) for h in artifact["debate_history"]]
        verdict, error = judge_debate(history)
        entry = {"run_id": run_id}
        if verdict is None:
            entry["error"] = error
            results["failures"].append(entry)
        else:
            entry.update(verdict)
            results["debates"].append(entry)
        print(
            f"  run{run_id:03d}/辩论："
            + (f"多方交锋 {verdict['bull_engagement']} 空方交锋 {verdict['bear_engagement']} "
               f"对立覆盖 {verdict['opposing_coverage']}"
               if verdict else f"评审失败: {error}")
        )
    return results


def _summarize(results: dict[str, Any]) -> dict[str, Any]:
    """汇总评审分数（均值与极值，全部真实计算）。"""
    def agg(items: list[dict[str, Any]], *keys: str) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(items)}
        for key in keys:
            values = [float(i[key]) for i in items if key in i]
            if values:
                out[key] = {
                    "mean": round(sum(values) / len(values), 2),
                    "min": min(values),
                    "max": max(values),
                }
        return out

    report_scores = agg(results["reports"], "factuality", "relevance", "no_hallucination")
    claims = [c for r in results["reports"] for c in r.get("claims", [])]
    grounding = (
        {"total": len(claims), "grounded": sum(1 for c in claims if c["grounded"]),
         "ratio": round(sum(1 for c in claims if c["grounded"]) / len(claims), 3)}
        if claims else {"total": 0, "grounded": 0, "ratio": 0.0}
    )
    debate_scores = agg(results["debates"], "bull_engagement", "bear_engagement", "opposing_coverage")
    return {
        "report_quality": report_scores,
        "claim_grounding": grounding,
        "debate_quality": debate_scores,
        "judge_failures": len(results["failures"]),
    }


def main(argv: list[str] | None = None) -> int:
    """评审入口：加载运行产物 → LLM-as-Judge → 保存并打印汇总。"""
    parser = argparse.ArgumentParser(description="AgentCouncil LLM-as-Judge 评审")
    parser.add_argument("--runs", nargs="+", type=int, default=None, help="指定运行编号（默认全部）")
    parser.add_argument("--query", default="600519.SH", help="与评测运行一致的查询")
    args = parser.parse_args(argv)

    if not config_module.CONFIG.deepseek_api_key:
        print("错误：未配置 DEEPSEEK_API_KEY（评审需要真实 LLM 调用）", file=sys.stderr)
        return 1
    run_ids: list[int] = args.runs or sorted(
        int(p.stem.split("_")[1]) for p in RUNS_DIR.glob("run_*.json")
    )
    if not run_ids:
        print(f"错误：{RUNS_DIR} 下没有 run_*.json（请先运行 python -m eval.run_eval）", file=sys.stderr)
        return 1

    print(f"=== LLM-as-Judge 评审（{len(run_ids)} 次运行，模型 {config_module.CONFIG.deepseek_model}） ===")
    results: dict[str, Any] = judge_runs(run_ids, args.query)
    summary: dict[str, Any] = _summarize(results)
    results["summary"] = summary
    out_path: Path = RUNS_DIR / "judge_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评审结果已保存：{out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=== 评审完成（评审失败次数已如实留痕） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())