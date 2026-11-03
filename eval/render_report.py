"""评测报告渲染：每次运行 → Markdown + 自包含 HTML（含各角色报告/辩论/风控/最终决策）。

输入：eval/runs/run_NNN.json（run_eval.py 产物）、metrics.json、
judge_results.json（judge.py 产物，可选）。
输出：eval/reports/run_NNN.md + run_NNN.html、SUMMARY.md + SUMMARY.html。
HTML 为单文件自包含（内联 CSS，无外部依赖），可直接浏览器打开。

用法（项目根目录）：
    python -m eval.render_report
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS_DIR: Path = Path(__file__).resolve().parent / "runs"
REPORTS_DIR: Path = Path(__file__).resolve().parent / "reports"

# 评级 → 徽章配色（HTML）
_RATING_COLORS: dict[str, str] = {
    "买入": "#1a7f37", "增持": "#2da44e", "中性": "#9a6700",
    "减持": "#cf222e", "卖出": "#a40e26",
}
_RISK_COLORS: dict[str, str] = {"高": "#cf222e", "中": "#bc4c00", "低": "#1a7f37"}

_CSS: str = """
body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 980px;
       margin: 24px auto; padding: 0 16px; color: #1f2328; line-height: 1.6; }
h1 { border-bottom: 2px solid #d0d7de; padding-bottom: 8px; }
h2 { margin-top: 28px; border-left: 4px solid #0969da; padding-left: 8px; }
h3 { margin-top: 20px; }
.card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 16px;
        margin: 12px 0; background: #ffffff; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         color: #fff; font-weight: 600; font-size: 14px; }
.meta { color: #57606a; font-size: 13px; }
.vs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.vs .side { border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 12px; }
.vs .bull { border-left: 4px solid #1a7f37; }
.vs .bear { border-left: 4px solid #cf222e; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 14px; }
th { background: #f6f8fa; }
ul { margin: 4px 0; }
.error { color: #cf222e; }
.score { font-weight: 700; }
"""


def _load_json(path: Path) -> dict[str, Any] | None:
    """加载 JSON（不存在返回 None）。"""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _esc(text: Any) -> str:
    """HTML 转义。"""
    return html.escape(str(text or ""))


def _run_judge(judge_results: dict[str, Any] | None, run_id: int) -> dict[str, Any]:
    """提取某次运行的评审结果（报告按角色索引 + 辩论评审）。"""
    out: dict[str, Any] = {"reports": {}, "debate": None}
    if not judge_results:
        return out
    for entry in judge_results.get("reports", []):
        if entry.get("run_id") == run_id:
            out["reports"][entry["role"]] = entry
    for entry in judge_results.get("debates", []):
        if entry.get("run_id") == run_id:
            out["debate"] = entry
    return out


# ------------------------------------------------------------------ Markdown


def _md_report_scores(judge: dict[str, Any] | None) -> str:
    """报告评审分数（Markdown 行）。"""
    if not judge or "error" in judge:
        return f"（评审失败：{judge.get('error')}）" if judge else ""
    claims = judge.get("claims", [])
    grounded = sum(1 for c in claims if c["grounded"])
    return (
        f"事实性 **{judge['factuality']}/5** ｜相关性 **{judge['relevance']}/5** ｜"
        f"无幻觉 **{judge['no_hallucination']}/5** ｜论点有据 **{grounded}/{len(claims)}**\n\n"
        f"> 评审意见：{judge['comment']}"
    )


def run_to_markdown(run: dict[str, Any], judge_results: dict[str, Any] | None) -> str:
    """渲染单次运行 → Markdown。"""
    rid: int = run["run_id"]
    decision: dict[str, Any] = run["final_decision"]
    judge = _run_judge(judge_results, rid)
    lines: list[str] = [
        f"# AgentCouncil 运行报告 #{rid:03d}",
        "",
        f"- 标的：`{run['query']}`",
        f"- 时间：{run['started_at']}",
        f"- 端到端耗时：**{run['wall_seconds']}s**（分析师并行 {run['stage_seconds']['analyst_parallel']}s / "
        f"辩论 {run['stage_seconds']['debate']}s / 风控 {run['stage_seconds']['risk_review']}s / "
        f"经理 {run['stage_seconds']['manager_summarize']}s）",
        "",
        "## 三名分析师报告",
        "",
    ]
    for report in run["analyst_reports"]:
        role_judge = judge["reports"].get(report["role"])
        lines += [
            f"### {report['role']}分析师（置信度 {report['confidence']}，状态 {report['status']}）",
            "",
            f"**摘要**：{report['summary']}",
            "",
            "**核心论点**：",
            "",
        ]
        lines += [f"- {kp}" for kp in report["key_points"]]
        lines += ["", f"**来源**：{'；'.join(report['sources']) if report['sources'] else '（无）'}", ""]
        if report["tool_errors"]:
            lines += [f"**工具留痕**：{'；'.join(report['tool_errors'])}", ""]
        lines += [f"**LLM-as-Judge 评审**：{_md_report_scores(role_judge)}", ""]

    lines += ["## 多空辩论（逐轮）", ""]
    for h in run["debate_history"]:
        lines += [f"### 第 {h['round_no']} 轮", "", f"**多方**：{h['bull_argument']}", "", f"**空方**：{h['bear_argument']}", ""]
    debate = run["debate_report"] or {}
    lines += [f"**收敛结论**（{debate.get('total_rounds')} 轮）：{debate.get('conclusion', '')}", ""]
    if debate.get("open_questions"):
        lines += ["**未决问题**：", ""] + [f"- {q}" for q in debate["open_questions"]] + [""]
    if judge["debate"] and "error" not in judge["debate"]:
        d = judge["debate"]
        lines += [f"**辩论评审**：多方交锋 {d['bull_engagement']}/5 ｜空方交锋 {d['bear_engagement']}/5 ｜"
                  f"对立覆盖 {d['opposing_coverage']}/5\n\n> {d['comment']}", ""]

    risk = run["risk_report"] or {}
    lines += [f"## 风控评审（整体风险：**{risk.get('risk_level')}**，blocked={risk.get('blocked')}）", ""]
    for item in risk.get("risks", []):
        lines += [f"- {item['level']}风险：**{item['title']}**——{item['description']}（缓释：{item['mitigation']}）"]
    lines += ["", f"> {risk.get('comment', '')}", ""]

    lines += [
        f"## 最终决策：{decision['rating']}（置信度 {decision['confidence']}）",
        "",
        f"**摘要**：{decision['summary']}",
        "",
        f"**决策理由**：{decision['reasoning']}",
        "",
        "**采纳论点**：",
        "",
    ]
    lines += [f"- {kp}" for kp in decision.get("key_points", [])]
    lines += ["", "**风险提示**：", ""]
    lines += [f"- {hint}" for hint in decision.get("risk_hints", [])]
    lines += ["", f"**来源汇总**：{'；'.join(decision.get('sources', []))}", ""]
    if decision.get("pipeline_errors"):
        lines += [f"**管线错误留痕**：{'；'.join(decision['pipeline_errors'])}", ""]
    return "\n".join(lines)


def summary_to_markdown(metrics: dict[str, Any], judge_results: dict[str, Any] | None) -> str:
    """渲染评测汇总 → Markdown。"""
    c = metrics["consistency"]
    l = metrics["latency_seconds"]
    lines: list[str] = [
        "# AgentCouncil 真实评测汇总",
        "",
        f"- 标的：`{metrics['query']}`　运行次数：{metrics['runs']}（真实数据 + DeepSeek，逐次串行）",
        "",
        "## 1. 决策一致性",
        "",
        f"- 评级分布：{c['rating_counts']}",
        f"- 众数评级：**{c['mode_rating']}**，一致率 {c['consistency_ratio']:.0%}（{c['rating_counts'][c['mode_rating']]}/{metrics['runs']}）",
        f"- 置信度：均值 {c['confidence_mean']}，标准差 {c['confidence_stdev']}",
        "",
        "## 2. 延迟（真实墙钟，秒）",
        "",
        f"| 指标 | 端到端 | 分析师并行 | 辩论 | 风控 | 经理 |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| 均值 | {l['mean']} | {l['by_stage']['analyst_parallel']['mean']} | {l['by_stage']['debate']['mean']} | {l['by_stage']['risk_review']['mean']} | {l['by_stage']['manager_summarize']['mean']} |",
        f"| P95/最大 | {l['p95']} | {l['by_stage']['analyst_parallel']['max']} | {l['by_stage']['debate']['max']} | {l['by_stage']['risk_review']['max']} | {l['by_stage']['manager_summarize']['max']} |",
        f"| 最小 | {l['min']} | - | - | - | - |",
        "",
        f"（逐次样本：{l['samples']}）",
        "",
        "## 3. 确定性质量指标",
        "",
        f"- 引用来源比例（报告带工具来源）：{metrics['deterministic_quality']['citation_ratio']:.0%}",
        f"- 辩论双侧论点比例（多空均非空且不同）：{metrics['deterministic_quality']['debate_both_sides_ratio']:.0%}",
        f"- 管线错误留痕总数：{metrics['deterministic_quality']['pipeline_error_count']}",
        "",
    ]
    if judge_results:
        s = judge_results.get("summary", {})
        rq = s.get("report_quality", {})
        cg = s.get("claim_grounding", {})
        dq = s.get("debate_quality", {})
        lines += [
            "## 4. LLM-as-Judge 评审（DeepSeek 评审，真实调用）",
            "",
            f"- 报告质量（{rq.get('n', 0)} 份）：事实性 {rq['factuality']['mean']}/5，"
            f"相关性 {rq['relevance']['mean']}/5，无幻觉 {rq['no_hallucination']['mean']}/5",
            f"- 幻觉率（论点级）：有据 {cg.get('grounded', 0)}/{cg.get('total', 0)}，"
            f"有据率 {cg.get('ratio', 0):.0%}（无据率 {1 - cg.get('ratio', 0):.0%}）",
            f"- 辩论有效性（{dq.get('n', 0)} 次）：多方交锋 {dq['bull_engagement']['mean']}/5，"
            f"空方交锋 {dq['bear_engagement']['mean']}/5，对立覆盖 {dq['opposing_coverage']['mean']}/5",
            f"- 评审失败次数：{s.get('judge_failures', 0)}",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ HTML


def _html_report_scores(judge: dict[str, Any] | None) -> str:
    """报告评审分数（HTML 片段）。"""
    if not judge:
        return '<span class="meta">（无评审结果）</span>'
    if "error" in judge:
        return f'<span class="error">（评审失败：{_esc(judge["error"])}）</span>'
    claims = judge.get("claims", [])
    grounded = sum(1 for c in claims if c["grounded"])
    return (
        f'<span class="score">事实性 {judge["factuality"]}/5</span> ｜ '
        f'<span class="score">相关性 {judge["relevance"]}/5</span> ｜ '
        f'<span class="score">无幻觉 {judge["no_hallucination"]}/5</span> ｜ '
        f'<span class="score">论点有据 {grounded}/{len(claims)}</span>'
        f'<div class="meta">评审意见：{_esc(judge["comment"])}</div>'
    )


def run_to_html(run: dict[str, Any], judge_results: dict[str, Any] | None) -> str:
    """渲染单次运行 → 自包含 HTML。"""
    rid: int = run["run_id"]
    decision: dict[str, Any] = run["final_decision"]
    judge = _run_judge(judge_results, rid)
    rating_color = _RATING_COLORS.get(decision["rating"], "#57606a")

    parts: list[str] = [
        f"<h1>AgentCouncil 运行报告 #{rid:03d}</h1>",
        f'<p class="meta">标的：{_esc(run["query"])} ｜ 时间：{_esc(run["started_at"])} ｜ '
        f'端到端 <b>{run["wall_seconds"]}s</b>（分析师并行 {run["stage_seconds"]["analyst_parallel"]}s / '
        f'辩论 {run["stage_seconds"]["debate"]}s / 风控 {run["stage_seconds"]["risk_review"]}s / '
        f'经理 {run["stage_seconds"]["manager_summarize"]}s）</p>',
        "<h2>三名分析师报告</h2>",
    ]
    for report in run["analyst_reports"]:
        role_judge = judge["reports"].get(report["role"])
        kps = "".join(f"<li>{_esc(kp)}</li>" for kp in report["key_points"])
        sources = "；".join(report["sources"]) if report["sources"] else "（无）"
        tool_errors = (
            f'<p class="error">工具留痕：{_esc("；".join(report["tool_errors"]))}</p>'
            if report["tool_errors"] else ""
        )
        parts.append(
            f'<div class="card"><h3>{_esc(report["role"])}分析师 '
            f'<span class="badge" style="background:#0969da">置信度 {report["confidence"]}</span></h3>'
            f"<p><b>摘要</b>：{_esc(report['summary'])}</p>"
            f"<p><b>核心论点</b>：</p><ul>{kps}</ul>"
            f'<p class="meta"><b>来源</b>：{_esc(sources)}</p>{tool_errors}'
            f"<p><b>LLM-as-Judge</b>：{_html_report_scores(role_judge)}</p></div>"
        )

    parts.append("<h2>多空辩论（逐轮）</h2>")
    for h in run["debate_history"]:
        parts.append(
            f'<div class="card"><h3>第 {h["round_no"]} 轮</h3><div class="vs">'
            f'<div class="side bull"><b>多方</b>：{_esc(h["bull_argument"])}</div>'
            f'<div class="side bear"><b>空方</b>：{_esc(h["bear_argument"])}</div>'
            "</div></div>"
        )
    debate = run["debate_report"] or {}
    open_qs = "".join(f"<li>{_esc(q)}</li>" for q in debate.get("open_questions", []))
    debate_judge_html = ""
    if judge["debate"] and "error" not in judge["debate"]:
        d = judge["debate"]
        debate_judge_html = (
            f'<p><b>辩论评审</b>：<span class="score">多方交锋 {d["bull_engagement"]}/5</span> ｜ '
            f'<span class="score">空方交锋 {d["bear_engagement"]}/5</span> ｜ '
            f'<span class="score">对立覆盖 {d["opposing_coverage"]}/5</span>'
            f'<div class="meta">{_esc(d["comment"])}</div></p>'
        )
    parts.append(
        f'<div class="card"><p><b>收敛结论</b>（{debate.get("total_rounds")} 轮）：'
        f'{_esc(debate.get("conclusion", ""))}</p>'
        + (f"<p><b>未决问题</b>：</p><ul>{open_qs}</ul>" if open_qs else "")
        + debate_judge_html + "</div>"
    )

    risk = run["risk_report"] or {}
    risk_items = "".join(
        f'<li><b style="color:{_RISK_COLORS.get(item["level"], "#57606a")}">{item["level"]}风险</b>：'
        f"<b>{_esc(item['title'])}</b>——{_esc(item['description'])}"
        f"（缓释：{_esc(item['mitigation'])}）</li>"
        for item in risk.get("risks", [])
    )
    parts.append(
        f'<div class="card"><h2>风控评审</h2><p>整体风险：'
        f'<span class="badge" style="background:{_RISK_COLORS.get(risk.get("risk_level"), "#57606a")}">'
        f'{_esc(risk.get("risk_level"))}</span>（blocked={risk.get("blocked")}）</p>'
        f"<ul>{risk_items}</ul><p class=\"meta\">评审意见：{_esc(risk.get('comment', ''))}</p></div>"
    )

    kps = "".join(f"<li>{_esc(kp)}</li>" for kp in decision.get("key_points", []))
    hints = "".join(f"<li>{_esc(hint)}</li>" for hint in decision.get("risk_hints", []))
    sources = "；".join(decision.get("sources", []))
    errors_html = (
        f'<p class="error">管线错误留痕：{_esc("；".join(decision["pipeline_errors"]))}</p>'
        if decision.get("pipeline_errors") else ""
    )
    parts.append(
        f'<div class="card"><h2>最终决策 '
        f'<span class="badge" style="background:{rating_color}">{_esc(decision["rating"])}</span> '
        f'<span class="badge" style="background:#57606a">置信度 {decision["confidence"]}</span></h2>'
        f"<p><b>摘要</b>：{_esc(decision['summary'])}</p>"
        f"<p><b>决策理由</b>：{_esc(decision['reasoning'])}</p>"
        f"<p><b>采纳论点</b>：</p><ul>{kps}</ul>"
        f"<p><b>风险提示</b>：</p><ul>{hints}</ul>"
        f'<p class="meta"><b>来源汇总</b>：{_esc(sources)}</p>{errors_html}</div>'
    )
    return f"<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">" \
        f"<title>AgentCouncil 运行报告 #{rid:03d}</title><style>{_CSS}</style></head>" \
        f"<body>{''.join(parts)}</body></html>"


def summary_to_html(metrics: dict[str, Any], judge_results: dict[str, Any] | None) -> str:
    """渲染评测汇总 → 自包含 HTML。"""
    c = metrics["consistency"]
    l = metrics["latency_seconds"]
    rating_rows = "".join(
        f'<tr><td>{_esc(k)}</td><td>{v}</td></tr>' for k, v in c["rating_counts"].items()
    )
    judge_section = ""
    if judge_results:
        s = judge_results.get("summary", {})
        rq = s.get("report_quality", {})
        cg = s.get("claim_grounding", {})
        dq = s.get("debate_quality", {})
        judge_section = (
            "<h2>4. LLM-as-Judge 评审（DeepSeek 真实评审）</h2><div class=\"card\"><table>"
            f"<tr><th>维度</th><th>分数</th></tr>"
            f"<tr><td>报告事实性（{rq.get('n', 0)} 份）</td><td class=\"score\">{rq['factuality']['mean']}/5</td></tr>"
            f"<tr><td>报告相关性</td><td class=\"score\">{rq['relevance']['mean']}/5</td></tr>"
            f"<tr><td>报告无幻觉</td><td class=\"score\">{rq['no_hallucination']['mean']}/5</td></tr>"
            f"<tr><td>论点有据率（幻觉率=1-有据率）</td><td class=\"score\">{cg.get('ratio', 0):.0%}"
            f"（{cg.get('grounded', 0)}/{cg.get('total', 0)} 条）</td></tr>"
            f"<tr><td>辩论多方交锋（{dq.get('n', 0)} 次）</td><td class=\"score\">{dq['bull_engagement']['mean']}/5</td></tr>"
            f"<tr><td>辩论空方交锋</td><td class=\"score\">{dq['bear_engagement']['mean']}/5</td></tr>"
            f"<tr><td>辩论对立覆盖</td><td class=\"score\">{dq['opposing_coverage']['mean']}/5</td></tr>"
            f"<tr><td>评审失败次数</td><td>{s.get('judge_failures', 0)}</td></tr>"
            "</table></div>"
        )
    body = (
        f"<h1>AgentCouncil 真实评测汇总</h1>"
        f'<p class="meta">标的：{_esc(metrics["query"])} ｜ 运行次数：{metrics["runs"]}'
        f"（真实数据 + DeepSeek，逐次串行）</p>"
        f'<div class="card"><h2>1. 决策一致性</h2><table>'
        f"<tr><th>评级</th><th>次数</th></tr>{rating_rows}</table>"
        f"<p>众数评级：<b>{_esc(c['mode_rating'])}</b> ｜ 一致率："
        f"<span class=\"score\">{c['consistency_ratio']:.0%}</span> ｜ "
        f"置信度均值 {c['confidence_mean']}（标准差 {c['confidence_stdev']}）</p></div>"
        f'<div class="card"><h2>2. 延迟（真实墙钟，秒）</h2><table>'
        "<tr><th>指标</th><th>端到端</th><th>分析师并行</th><th>辩论</th><th>风控</th><th>经理</th></tr>"
        f"<tr><td>均值</td><td>{l['mean']}</td><td>{l['by_stage']['analyst_parallel']['mean']}</td>"
        f"<td>{l['by_stage']['debate']['mean']}</td><td>{l['by_stage']['risk_review']['mean']}</td>"
        f"<td>{l['by_stage']['manager_summarize']['mean']}</td></tr>"
        f"<tr><td>P95/最大</td><td class=\"score\">{l['p95']}</td><td>{l['by_stage']['analyst_parallel']['max']}</td>"
        f"<td>{l['by_stage']['debate']['max']}</td><td>{l['by_stage']['risk_review']['max']}</td>"
        f"<td>{l['by_stage']['manager_summarize']['max']}</td></tr>"
        f"<tr><td>最小</td><td>{l['min']}</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>"
        f"</table><p class=\"meta\">逐次样本：{l['samples']}</p></div>"
        f'<div class="card"><h2>3. 确定性质量指标</h2><ul>'
        f"<li>引用来源比例（报告带工具来源）：<span class=\"score\">"
        f"{metrics['deterministic_quality']['citation_ratio']:.0%}</span></li>"
        f"<li>辩论双侧论点比例：<span class=\"score\">"
        f"{metrics['deterministic_quality']['debate_both_sides_ratio']:.0%}</span></li>"
        f"<li>管线错误留痕总数：{metrics['deterministic_quality']['pipeline_error_count']}</li>"
        f"</ul></div>"
        f"{judge_section}"
    )
    return f"<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">" \
        "<title>AgentCouncil 真实评测汇总</title><style>" + _CSS + "</style></head>" \
        f"<body>{body}</body></html>"


# ------------------------------------------------------------------ 驱动


def main() -> int:
    """加载评测产物并渲染全部报告。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_files = sorted(RUNS_DIR.glob("run_*.json"))
    if not run_files:
        print(f"错误：{RUNS_DIR} 下没有 run_*.json（请先运行 python -m eval.run_eval）", file=sys.stderr)
        return 1
    metrics = _load_json(RUNS_DIR / "metrics.json") or {}
    judge_results = _load_json(RUNS_DIR / "judge_results.json")

    for path in run_files:
        run: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        rid: int = run["run_id"]
        (REPORTS_DIR / f"run_{rid:03d}.md").write_text(
            run_to_markdown(run, judge_results), encoding="utf-8"
        )
        (REPORTS_DIR / f"run_{rid:03d}.html").write_text(
            run_to_html(run, judge_results), encoding="utf-8"
        )
        print(f"已渲染 run_{rid:03d}.md / run_{rid:03d}.html")
    if metrics:
        (REPORTS_DIR / "SUMMARY.md").write_text(
            summary_to_markdown(metrics, judge_results), encoding="utf-8"
        )
        (REPORTS_DIR / "SUMMARY.html").write_text(
            summary_to_html(metrics, judge_results), encoding="utf-8"
        )
        print("已渲染 SUMMARY.md / SUMMARY.html")
    print(f"报告目录：{REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
