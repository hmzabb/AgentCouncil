"""基金经理汇总：聚合全链路产出，给出评级 + 置信度 + 摘要 + 风险提示。

产出路径（按模式）：
- MOCK_MODE=true：确定性评级映射（冒烟测试用）；
- MOCK_MODE=false：DeepSeek LLM 综合分析师报告 / 辩论 / 风控产出决策；
  LLM 失败时降级为确定性映射 + 错误留痕。
代码兜底（两种模式都生效）：
- 无成功分析师报告 → 评级「中性」、置信度 0.0、提示补充调研；
- 风控一票权：blocked → 评级上限「卖出」；风险等级 高 → 评级上限「减持」
  （LLM 给出的评级不得越过风控上限，越界则截断并在 reasoning 中说明）；
- sources / pipeline_errors 由代码真实汇总，不让 LLM 产出。
"""

from __future__ import annotations

import logging
from typing import Any

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.base import BaseAgent
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import (
    AgentStatus,
    AnalystReport,
    DebateReport,
    FinalDecision,
    FinalDecisionLlmOutput,
    RiskReviewReport,
)
from graph.state import CouncilState

logger = logging.getLogger(__name__)

# 评级强度序（用于风控一票权上限截断）
_RATING_ORDER: tuple[str, ...] = ("买入", "增持", "中性", "减持", "卖出")

_MANAGER_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：基金经理（唯一决策者）。任务：综合分析师报告、多空辩论与风控评审，"
    + "对「{query}」给出最终投资决策。"
    + "\n职责边界：只基于上游报告做综合与权衡，不重复做数据层面的分析，"
    + "不替分析师补充论点；投资评级由你给出。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- rating: 字符串，投资评级，取值仅限 买入/增持/中性/减持/卖出；"
    + "\n- confidence: 0~1 的数字，综合置信度；"
    + "\n- summary: 字符串，投资分析摘要（100~200 字）；"
    + "\n- reasoning: 字符串，决策理由（说明如何权衡多方证据与风险）；"
    + "\n- key_points: 字符串数组，采纳的核心论点；"
    + "\n- risk_hints: 字符串数组，风险提示。"
    + "\n要求：中文；基于给定材料，不编造；不要输出【stub】字样。"
)


class ManagerAgent(BaseAgent[FinalDecision]):
    """基金经理：LLM 综合决策 + 风控一票权 + 代码兜底映射。"""

    name = "manager_summarize"

    def execute(self, state: CouncilState) -> FinalDecision:
        query: str = state["query"]
        reports: list[AnalystReport] = state.get("analyst_reports", [])
        debate: DebateReport | None = state.get("debate_report")
        risk: RiskReviewReport | None = state.get("risk_report")
        errors: list[str] = list(state.get("errors", []))

        ok_reports = [r for r in reports if r.status is AgentStatus.OK]
        if not ok_reports:
            return FinalDecision(
                agent=self.name,
                status=AgentStatus.OK,
                rating="中性",
                confidence=0.0,
                summary=f"【stub】{query}：分析师阶段无有效产出，暂无法形成投资分析。",
                reasoning=f"【stub】可用分析师报告为 0（共 {len(reports)} 份，全部失败），需补充调研。",
                key_points=[],
                risk_hints=["信息不足，建议补充调研后再评估。"],
                pipeline_errors=errors,
            )

        # ---------- LLM 主路径（真实模式） / 确定性映射（mock 或降级）----------
        note: str | None = None
        llm_error: str | None = None
        if config_module.CONFIG.mock_mode:
            rating, confidence, summary, reasoning, key_points = self._deterministic_decision(
                query, ok_reports, debate, risk
            )
            risk_hints = self._risk_hints(risk)
        else:
            try:
                out: FinalDecisionLlmOutput = self._llm_decision(query, ok_reports, debate, risk)
                rating, confidence = out.rating, out.confidence
                summary, reasoning, key_points = out.summary, out.reasoning, out.key_points
                risk_hints = out.risk_hints
                note = "真实数据 + DeepSeek 生成"
            except Exception as exc:  # noqa: BLE001 —— fail-open：LLM 失败降级映射
                logger.warning("经理 LLM 失败（降级确定性映射）：%s", exc)
                llm_error = f"LLM 调用失败（降级确定性映射）: {type(exc).__name__}: {exc}"
                errors.append(llm_error)
                rating, confidence, summary, reasoning, key_points = self._deterministic_decision(
                    query, ok_reports, debate, risk
                )
                risk_hints = self._risk_hints(risk)
                note = "【降级模板】LLM 调用失败"

        # ---------- 代码兜底：风控一票权（LLM 评级不得越过风控上限）----------
        if risk is not None and risk.blocked:
            cap = "卖出"
        elif risk is not None and risk.risk_level == "高":
            cap = "减持"
        else:
            cap = None
        if cap is not None and _RATING_ORDER.index(rating) < _RATING_ORDER.index(cap):
            reasoning = (
                f"{reasoning}（风控一票权：风险等级为{risk.risk_level}，"
                f"评级上限 {cap}，原评级 {rating} 已截断）"
            )
            rating = cap

        sources: list[str] = self._collect_sources(ok_reports, debate)
        kwargs: dict[str, Any] = {}
        if note is not None:  # note 为 str 字段：mock 路径不传，用 schema 默认值
            kwargs["note"] = note
        return FinalDecision(
            agent=self.name,
            status=AgentStatus.OK,
            **kwargs,
            rating=rating,
            confidence=confidence,
            summary=summary,
            reasoning=reasoning,
            key_points=key_points,
            risk_hints=risk_hints,
            sources=sources,
            pipeline_errors=errors,
        )

    # ---------- LLM 主路径 ----------

    def _llm_decision(
        self,
        query: str,
        ok_reports: list[AnalystReport],
        debate: DebateReport | None,
        risk: RiskReviewReport | None,
    ) -> FinalDecisionLlmOutput:
        """构造决策提示词（各阶段报告压缩）并调用 LLM。"""
        analyst_text = "\n".join(
            f"【{r.role}分析师｜置信度 {r.confidence}】{r.summary[:250]}\n"
            f"  论点：{'；'.join(k[:80] for k in r.key_points[:3])}"
            for r in ok_reports
        )
        debate_text = (
            f"结论：{debate.conclusion[:300]}\n"
            f"未决问题：{'；'.join(debate.open_questions[:3])}"
            if debate else "（辩论报告缺失）"
        )
        risk_text = (
            f"整体风险等级：{risk.risk_level}；"
            + "；".join(f"{r.title}({r.level})" for r in risk.risks[:6])
            + f"；评审意见：{risk.comment[:150]}"
            if risk else "（风控报告缺失）"
        )
        system = _MANAGER_SYSTEM_PROMPT.format(query=query)
        human = (
            f"分析标的：{query}\n"
            f"分析师报告：\n{analyst_text}\n"
            f"多空辩论：{debate_text}\n"
            f"风控评审：{risk_text}\n"
            f"请输出最终决策 JSON。"
        )
        return structured_call(FinalDecisionLlmOutput, system=system, human=human)

    # ---------- 确定性映射（mock / LLM 降级）----------

    def _deterministic_decision(
        self,
        query: str,
        ok_reports: list[AnalystReport],
        debate: DebateReport | None,
        risk: RiskReviewReport | None,
    ) -> tuple[str, float, str, str, list[str]]:
        """确定性评级映射 + 置信度均值（真实计算，非编造）。"""
        confidence: float = round(sum(r.confidence for r in ok_reports) / len(ok_reports), 2)
        if risk is not None and risk.blocked:
            rating = "卖出"
        elif risk is not None and risk.risk_level == "高":
            rating = "减持"
        elif risk is not None and risk.risk_level == "低":
            rating = "增持"
        else:
            rating = "中性"
        debate_rounds: int = debate.total_rounds if debate is not None else 0
        summary = (
            f"【stub】{query}：综合 {len(ok_reports)} 位分析师"
            f"（{'、'.join(r.role for r in ok_reports)}）与 {debate_rounds} 轮多空辩论、"
            f"风控评审，给出评级「{rating}」（置信度 {confidence}）。"
        )
        reasoning = (
            f"【stub】置信度取成功分析师置信度均值（真实计算）；"
            f"评级由风控风险等级映射；辩论结论：{debate.conclusion if debate else '缺失'}。"
        )
        key_points = [r.key_points[0] for r in ok_reports if r.key_points]
        return rating, confidence, summary, reasoning, key_points

    def _risk_hints(self, risk: RiskReviewReport | None) -> list[str]:
        """风险提示：取风控清单的等级 + 缓释措施（确定性）。"""
        if risk is None:
            return ["风控评审缺失，请谨慎决策。"]
        return [f"{item.level}风险：{item.title}——{item.mitigation}" for item in risk.risks]

    def _collect_sources(
        self, ok_reports: list[AnalystReport], debate: DebateReport | None
    ) -> list[str]:
        """来源汇总（去重、保持顺序，代码真实合并）。"""
        sources: list[str] = []
        for report in ok_reports:
            for src in report.sources:
                if src not in sources:
                    sources.append(src)
        if debate is not None:
            for src in debate.sources:
                if src not in sources:
                    sources.append(src)
        return sources

    def fallback(self, error: Exception) -> FinalDecision:
        """失败降级：经理自身失败时给出最保守输出（fail-open）。"""
        return FinalDecision(
            agent=self.name,
            status=AgentStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            rating="中性",
            confidence=0.0,
            summary=f"【stub】经理汇总失败，无法形成投资分析：{error}",
            reasoning=f"经理汇总失败：{error}",
            key_points=[],
        )
