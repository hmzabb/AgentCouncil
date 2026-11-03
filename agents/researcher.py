"""研究员多空辩论：每轮产出一组多方/空方论点；最终轮产出收敛报告。

由 researcher_debate 节点按轮次驱动（conditional_edges + max_rounds 防死循环）。
产出路径（按模式）：
- MOCK_MODE=true：确定性占位论点模板（冒烟测试用）；
- MOCK_MODE=false：DeepSeek LLM 基于研报检索证据 + 前轮辩论历史生成论点；
  LLM 失败时 fail-open 降级为模板论点 + 错误留痕。
收敛报告（synthesize）同理：LLM 汇总多空历史，失败降级模板。
"""

from __future__ import annotations

import logging
from typing import Any

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.base import BaseAgent
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import (
    AgentStatus,
    DebateLlmOutput,
    DebateReport,
    DebateRound,
    DebateSynthesisLlmOutput,
)
from graph.state import CouncilState
from tools.mock_tools import call_tool

logger = logging.getLogger(__name__)

# 确定性占位论点模板（mock / LLM 降级用），按轮次取模循环
_BULL_TEMPLATES: tuple[str, ...] = (
    "{marker}多方论点 A：{q} 基本面支撑仍在，回调不改中长期逻辑。",
    "{marker}多方论点 B：{q} 行业景气度上行，估值消化后仍有空间。",
    "{marker}多方论点 C：{q} 资金面边际改善，配置价值逐步显现。",
)
_BEAR_TEMPLATES: tuple[str, ...] = (
    "{marker}空方论点 A：{q} 短期涨幅已透支部分预期，注意回调风险。",
    "{marker}空方论点 B：{q} 宏观与流动性存在不确定性，安全边际不足。",
    "{marker}空方论点 C：{q} 情绪过热，交易拥挤度偏高。",
)

_ROUND_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：多空辩论研究员，负责第 {round_no} 轮辩论。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- bull_argument: 字符串，多方论点（引用检索证据或数据中的具体事实）；"
    + "\n- bear_argument: 字符串，空方论点（引用检索证据或数据中的具体事实）。"
    + "\n要求：中文；论点基于给定证据，不编造数据；观点要有交锋（针对上一轮可反驳）；"
    + "不要输出【stub】字样。"
)

_SYNTHESIS_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：多空辩论主持人，负责对「{query}」的 {rounds} 轮辩论做收敛总结。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- conclusion: 字符串，多空辩论收敛结论（100~200 字，指出哪方论据更有力）；"
    + "\n- open_questions: 字符串数组，仍待数据验证的未决问题。"
    + "\n要求：中文；基于给定辩论历史，不编造；不要输出【stub】字样。"
)


class ResearcherAgent(BaseAgent[DebateRound]):
    """研究员：单轮多空辩论的产出者（LLM 主路径 + 模板降级）。"""

    def __init__(self, round_no: int) -> None:
        """初始化本轮研究员。

        参数：
            round_no: 轮次序号（≥1，由节点维护计数）。
        """
        self.name = "researcher_debate"
        self.round_no = round_no

    def execute(self, state: CouncilState) -> DebateRound:
        query: str = state["query"]
        # 工具：研报检索（真实模式走 Milvus；mock 走占位数据）
        rag = call_tool("research_rag", query=f"{query} 多空辩论", top_k=2)
        chunks: list[dict[str, Any]] = rag.data["chunks"] if rag.ok and rag.data else []
        tool_error: str | None = (
            None if (rag.ok and rag.data) else f"research_rag: {rag.error or '无检索结果'}"
        )

        if config_module.CONFIG.mock_mode:
            return self._template_round(query, chunks, tool_error, marker="【stub】")
        try:
            out: DebateLlmOutput = self._llm_round(query, chunks, state)
        except Exception as exc:  # noqa: BLE001 —— fail-open：LLM 失败降级模板
            logger.warning("研究员第 %d 轮 LLM 失败（降级模板）：%s", self.round_no, exc)
            tool_error = f"{tool_error}；LLM 调用失败: {type(exc).__name__}: {exc}" if tool_error \
                else f"LLM 调用失败: {type(exc).__name__}: {exc}"
            return self._template_round(query, chunks, tool_error, marker="【降级模板】")
        return DebateRound(
            round_no=self.round_no,
            bull_argument=out.bull_argument,
            bear_argument=out.bear_argument,
            error=tool_error,
        )

    # ---------- LLM 主路径 ----------

    def _llm_round(
        self, query: str, chunks: list[dict[str, Any]], state: CouncilState
    ) -> DebateLlmOutput:
        """构造本轮辩论提示词（检索证据 + 前轮交锋历史）并调用 LLM。"""
        evidence = "\n".join(
            f"【研报证据｜{((c.get('metadata') or {}).get('source') or c.get('doc_name'))}｜"
            f"相似度 {c.get('score')}】{str(c['chunk_text'])[:300]}"
            for c in chunks
        ) or "（本轮无检索证据）"
        history = state.get("debate_history", [])
        history_text = ""
        if history:
            last = history[-1]
            history_text = (
                f"上一轮多方论点：{last.bull_argument[:200]}\n"
                f"上一轮空方论点：{last.bear_argument[:200]}"
            )
        system = _ROUND_SYSTEM_PROMPT.format(round_no=self.round_no)
        human = (
            f"辩论主题：{query}\n"
            f"{history_text}\n"
            f"本轮检索证据：\n{evidence}\n"
            f"请输出第 {self.round_no} 轮多空论点 JSON。"
        )
        return structured_call(DebateLlmOutput, system=system, human=human)

    # ---------- 降级模板路径 ----------

    def _template_round(
        self, query: str, chunks: list[dict[str, Any]], tool_error: str | None, marker: str
    ) -> DebateRound:
        """模板论点（mock / LLM 降级共用，marker 区分【stub】/【降级模板】）。"""
        idx: int = (self.round_no - 1) % len(_BULL_TEMPLATES)
        if chunks:
            first = str(chunks[0]["chunk_text"])[:60]
            evidence = f"（研报检索 {len(chunks)} 条，摘录：{first}…）"
        elif tool_error:
            evidence = "（研报检索失败，本轮论点不含检索证据）"
        else:
            evidence = "（研报检索无结果）"
        bull: str = _BULL_TEMPLATES[idx].format(marker=marker, q=query) + evidence
        bear: str = _BEAR_TEMPLATES[idx].format(marker=marker, q=query)
        return DebateRound(
            round_no=self.round_no,
            bull_argument=bull,
            bear_argument=bear,
            error=tool_error,
        )

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
                f"第 {i + 1} 轮多方：{h.bull_argument[:200]}\n第 {i + 1} 轮空方：{h.bear_argument[:200]}"
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
