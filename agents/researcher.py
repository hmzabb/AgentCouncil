"""研究员辩论编排：每轮驱动独立的多方 Agent 与空方 Agent 交锋；最终轮产出收敛报告。

由 researcher_debate 节点按轮次驱动（conditional_edges + max_rounds 防死循环）。
本轮结构（真实交锋，非单次双产出）：
1. 取研报检索证据（research_rag，fail-open）；
2. 多方 Agent（BullAgent）独立 LLM 调用产出做多论证；
3. 空方 Agent（BearAgent）在拿到同轮多方论点后独立 LLM 调用，逐条反驳；
4. 双方错误分别留痕，合并进 DebateRound.error。
产出路径（按模式）：
- MOCK_MODE=true：双方各出确定性占位模板（冒烟测试用）；
- MOCK_MODE=false：各自 DeepSeek 调用，单方失败 fail-open 降级模板 + 留痕。
收敛报告（synthesize）同理：LLM 汇总多空历史，失败降级模板。
"""

from __future__ import annotations

import logging
from typing import Any

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.base import BaseAgent
from agents.debate import BearAgent, BullAgent, format_evidence, format_history
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import (
    AgentStatus,
    DebateReport,
    DebateRound,
    DebateSideOutput,
    DebateSynthesisLlmOutput,
)
from graph.state import CouncilState
from tools.mock_tools import call_tool

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：多空辩论主持人，负责对「{query}」的 {rounds} 轮辩论做收敛总结。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- conclusion: 字符串，多空辩论收敛结论（100~200 字，指出哪方论据更有力）；"
    + "\n- open_questions: 字符串数组，仍待数据验证的未决问题。"
    + "\n要求：中文；基于给定辩论历史，不编造；不要输出【stub】字样。"
)


class ResearcherAgent(BaseAgent[DebateRound]):
    """研究员编排器：驱动独立多方/空方 Agent 完成一轮交锋（最终轮另产出收敛报告）。"""

    def __init__(self, round_no: int) -> None:
        """初始化本轮研究员。

        参数：
            round_no: 轮次序号（≥1，由节点维护计数）。
        """
        self.name = "researcher_debate"
        self.round_no = round_no

    def execute(self, state: CouncilState) -> DebateRound:
        """执行一轮辩论：多方先发言，空方在拿到多方论点后独立反驳。"""
        query: str = state["query"]
        # 工具：研报检索（真实模式走 Milvus；mock 走占位数据），双方共用同一证据集
        rag = call_tool("research_rag", query=f"{query} 多空辩论", top_k=2)
        chunks: list[dict[str, Any]] = rag.data["chunks"] if rag.ok and rag.data else []
        tool_error: str | None = (
            None if (rag.ok and rag.data) else f"research_rag: {rag.error or '无检索结果'}"
        )
        evidence_text: str = format_evidence(chunks)
        history_text: str = format_history(state.get("debate_history", []))

        # 独立 Agent：两次互不共享生成上下文的 LLM 调用（mock 模式各自返回占位模板）
        bull: DebateSideOutput = BullAgent(self.round_no).run(query, evidence_text, history_text)
        bear: DebateSideOutput = BearAgent(self.round_no).run(
            query, evidence_text, history_text, bull_argument=bull.argument
        )

        errors: list[str] = [
            e for e in (tool_error, bull.error, bear.error) if e
        ]
        return DebateRound(
            round_no=self.round_no,
            bull_argument=self._annotate_evidence(bull.argument, chunks, tool_error),
            bear_argument=bear.argument,
            bull_points=list(bull.points),
            bear_points=list(bear.points),
            error="；".join(errors) if errors else None,
        )

    @staticmethod
    def _annotate_evidence(
        argument: str, chunks: list[dict[str, Any]], tool_error: str | None
    ) -> str:
        """占位/降级论点缺少数据支撑，追加本轮检索情况说明（真实 LLM 论点原样返回）。"""
        if not (argument.startswith("【stub】") or argument.startswith("【降级模板】")):
            return argument
        if chunks:
            first = str(chunks[0]["chunk_text"])[:60]
            return argument + f"（研报检索 {len(chunks)} 条，摘录：{first}…）"
        if tool_error:
            return argument + "（研报检索失败，本轮论点不含检索证据）"
        return argument + "（研报检索无结果）"

    # ---------- 最终轮：辩论收敛报告 ----------

    def synthesize(self, state: CouncilState, history: list[DebateRound]) -> DebateReport:
        """聚合多空历史产出收敛报告（LLM 主路径 + 模板降级）。由节点在最终轮调用。"""
        query: str = state["query"]
        pro = [h.bull_argument for h in history if h.bull_argument]
        con = [h.bear_argument for h in history if h.bear_argument]
        total_rounds: int = len(history)

        if config_module.CONFIG.mock_mode:
            return self._template_synthesis(query, total_rounds, pro, con, marker="【stub】")
        try:
            system = _SYNTHESIS_SYSTEM_PROMPT.format(query=query, rounds=total_rounds)
            history_text = "\n".join(
                f"第 {i + 1} 轮多方：{h.bull_argument[:200]}"
                + (f"（证据：{'；'.join(h.bull_points[:3])}）" if h.bull_points else "")
                + f"\n第 {i + 1} 轮空方：{h.bear_argument[:200]}"
                + (f"（反驳：{'；'.join(h.bear_points[:3])}）" if h.bear_points else "")
                for i, h in enumerate(history)
            )
            out: DebateSynthesisLlmOutput = structured_call(
                DebateSynthesisLlmOutput, system=system, human=f"{history_text}\n请输出收敛结论 JSON。"
            )
        except Exception as exc:  # noqa: BLE001 —— fail-open：LLM 失败降级模板
            logger.warning("研究员收敛 LLM 失败（降级模板）：%s", exc)
            return self._template_synthesis(
                query, total_rounds, pro, con,
                marker=f"【降级模板｜LLM 失败: {type(exc).__name__}: {exc}】",
            )
        return DebateReport(
            agent=self.name,
            status=AgentStatus.OK,
            note="真实数据 + DeepSeek 生成",
            total_rounds=total_rounds,
            pro_arguments=pro,
            con_arguments=con,
            conclusion=out.conclusion,
            open_questions=out.open_questions,
            sources=[],
        )

    def _template_synthesis(
        self, query: str, total_rounds: int, pro: list[str], con: list[str], marker: str
    ) -> DebateReport:
        """确定性收敛模板（mock / LLM 降级）。"""
        kwargs: dict[str, Any] = {}
        if marker != "【stub】":  # note 为 str 字段：mock 路径不传，用 schema 默认值
            kwargs["note"] = "【降级模板】LLM 调用失败"
        return DebateReport(
            agent=self.name,
            status=AgentStatus.OK,
            **kwargs,
            total_rounds=total_rounds,
            pro_arguments=pro,
            con_arguments=con,
            conclusion=(
                f"{marker}经过 {total_rounds} 轮多空辩论，对「{query}」收敛结论："
                f"多空分歧仍存，建议结合风控结论谨慎决策。"
            ),
            open_questions=[f"{marker}{query} 的关键假设待真实数据验证。"],
            sources=[],
        )

    def fallback(self, error: Exception) -> DebateRound:
        """失败降级：产出合法的失败态轮次记录，循环仍可继续/终止（fail-open）。"""
        return DebateRound(
            round_no=self.round_no,
            bull_argument="",
            bear_argument="",
            error=f"{type(error).__name__}: {error}",
        )