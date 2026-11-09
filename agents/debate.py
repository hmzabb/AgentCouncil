"""独立多空 Agent：多方（Bull）与空方（Bear）分别独立调用 LLM，形成真实交锋。

与「单个 Agent 一次产出两方论点」的本质区别：
1. 两次独立 LLM 调用：双方不共享同一生成上下文，避免自我妥协式「伪辩论」；
2. 空方生成前已拿到同轮多方论点，必须给出针对性反驳（rebuttals）；
3. 双方各自 fail-open：一方失败只留痕，另一方结论仍然有效；
4. 证据与历史格式化集中在本模块，编排层（researcher.py）复用，避免双份实现。

设计说明（为何不继承 BaseAgent）：
这两个 Agent 的输入是「辩论上下文（query + 证据 + 历史 + 对手论点）」而非整条
CouncilState，输出是单方论点而非报告类 schema；因此提供显式签名的 run()，
内部同样遵循 fail-open 约定（异常转为带 error 留痕的合法输出，绝不向上抛）。
"""

from __future__ import annotations

import logging
from typing import Any

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import BearArgumentLlmOutput, BullArgumentLlmOutput, DebateSideOutput

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

_BULL_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：多方研究员（Bull），负责「{query}」第 {round_no} 轮辩论的做多论证。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- argument: 字符串，多方核心论点（120~250 字，须引用给定证据或数据中的具体事实）；"
    + "\n- evidence: 字符串数组，支撑该论点的证据要点（每条须可回溯到给定材料）。"
    + "\n要求：中文；不得编造数据；数据缺失处须明确说明；"
    + "若上一轮空方提出了有效质疑，须正面回应而非回避；不要输出【stub】字样。"
)

_BEAR_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：空方研究员（Bear），负责「{query}」第 {round_no} 轮辩论的看空论证，"
    + "并针对同轮多方论点逐条反驳。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- argument: 字符串，空方核心论点（120~250 字，须引用给定证据或数据中的具体事实）；"
    + "\n- rebuttals: 字符串数组，对同轮多方论点的具体反驳（每条须指向多方某个论点）。"
    + "\n要求：中文；不得编造数据；反驳须基于给定材料而非空泛表态；不要输出【stub】字样。"
)


def format_evidence(chunks: list[dict[str, Any]], limit: int = 300) -> str:
    """把 RAG 检索结果格式化为辩论证据文本（无结果时返回占位说明）。"""
    if not chunks:
        return "（本轮无检索证据）"
    return "\n".join(
        f"【研报证据｜{((c.get('metadata') or {}).get('source') or c.get('doc_name'))}｜"
        f"相似度 {c.get('score')}】{str(c['chunk_text'])[:limit]}"
        for c in chunks
    )


def format_history(history: list[Any], last_only: bool = True) -> str:
    """把辩论历史格式化为交锋上下文文本（默认只取上一轮，控制提示词长度）。"""
    if not history:
        return ""
    rounds: list[Any] = history[-1:] if last_only else history
    parts: list[str] = []
    for r in rounds:
        round_no: Any = getattr(r, "round_no", "?")
        parts.append(f"第 {round_no} 轮多方论点：{str(getattr(r, 'bull_argument', ''))[:200]}")
        parts.append(f"第 {round_no} 轮空方论点：{str(getattr(r, 'bear_argument', ''))[:200]}")
    return "\n".join(parts)


class BullAgent:
    """多方研究员：独立 LLM 调用产出做多论证（fail-open 降级模板）。"""

    name = "bull_researcher"

    def __init__(self, round_no: int) -> None:
        """初始化多方 Agent。

        参数：
            round_no: 轮次序号（≥1，由编排节点维护计数）。
        """
        self.round_no = round_no

    def run(self, query: str, evidence_text: str, history_text: str = "") -> DebateSideOutput:
        """产出本轮多方论点（异常一律降级为带 error 留痕的合法输出）。"""
        try:
            return self._execute(query, evidence_text, history_text)
        except Exception as exc:  # noqa: BLE001 —— fail-open：单方失败不影响空方与链路
            logger.warning("多方 Agent 第 %d 轮失败（fail-open 降级）：%s", self.round_no, exc)
            return DebateSideOutput(
                agent=self.name,
                argument=_BULL_TEMPLATES[(self.round_no - 1) % len(_BULL_TEMPLATES)].format(
                    marker="【降级模板】", q=query
                ),
                points=[],
                error=f"LLM 调用失败: {type(exc).__name__}: {exc}",
            )

    def _execute(self, query: str, evidence_text: str, history_text: str) -> DebateSideOutput:
        if config_module.CONFIG.mock_mode:
            return DebateSideOutput(
                agent=self.name,
                argument=_BULL_TEMPLATES[(self.round_no - 1) % len(_BULL_TEMPLATES)].format(
                    marker="【stub】", q=query
                ),
                points=[],
            )
        system: str = _BULL_SYSTEM_PROMPT.format(query=query, round_no=self.round_no)
        human: str = (
            f"标的/关键词：{query}\n\n"
            f"【检索证据】\n{evidence_text}\n\n"
            f"【上一轮交锋】\n{history_text or '（首轮，无历史）'}"
        )
        out: BullArgumentLlmOutput = structured_call(
            BullArgumentLlmOutput, system=system, human=human
        )
        return DebateSideOutput(
            agent=self.name, argument=out.argument, points=list(out.evidence)
        )


class BearAgent:
    """空方研究员：独立 LLM 调用产出看空论证，并逐条反驳同轮多方论点。"""

    name = "bear_researcher"

    def __init__(self, round_no: int) -> None:
        """初始化空方 Agent。

        参数：
            round_no: 轮次序号（≥1，由编排节点维护计数）。
        """
        self.round_no = round_no

    def run(
        self,
        query: str,
        evidence_text: str,
        history_text: str = "",
        bull_argument: str = "",
    ) -> DebateSideOutput:
        """产出本轮空方论点（异常一律降级为带 error 留痕的合法输出）。"""
        try:
            return self._execute(query, evidence_text, history_text, bull_argument)
        except Exception as exc:  # noqa: BLE001 —— fail-open：单方失败不影响链路
            logger.warning("空方 Agent 第 %d 轮失败（fail-open 降级）：%s", self.round_no, exc)
            return DebateSideOutput(
                agent=self.name,
                argument=_BEAR_TEMPLATES[(self.round_no - 1) % len(_BEAR_TEMPLATES)].format(
                    marker="【降级模板】", q=query
                ),
                points=[],
                error=f"LLM 调用失败: {type(exc).__name__}: {exc}",
            )

    def _execute(
        self, query: str, evidence_text: str, history_text: str, bull_argument: str
    ) -> DebateSideOutput:
        if config_module.CONFIG.mock_mode:
            return DebateSideOutput(
                agent=self.name,
                argument=_BEAR_TEMPLATES[(self.round_no - 1) % len(_BEAR_TEMPLATES)].format(
                    marker="【stub】", q=query
                ),
                points=[],
            )
        system: str = _BEAR_SYSTEM_PROMPT.format(query=query, round_no=self.round_no)
        human: str = (
            f"标的/关键词：{query}\n\n"
            f"【检索证据】\n{evidence_text}\n\n"
            f"【同轮多方论点（需逐条反驳）】\n{bull_argument or '（多方本轮无产出）'}\n\n"
            f"【上一轮交锋】\n{history_text or '（首轮，无历史）'}"
        )
        out: BearArgumentLlmOutput = structured_call(
            BearArgumentLlmOutput, system=system, human=human
        )
        return DebateSideOutput(
            agent=self.name, argument=out.argument, points=list(out.rebuttals)
        )