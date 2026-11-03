"""风控评审：基于分析师报告 + 行情快照生成风险清单。

产出路径（按模式）：
- MOCK_MODE=true：确定性占位规则（冒烟测试用）；
- MOCK_MODE=false：DeepSeek LLM 基于分析师报告 + 行情快照
  （get_market_data，fail-open：行情失败不阻断风控，仅提示数据缺失）
  生成风险清单；LLM 失败时降级为确定性基础风险清单 + 错误留痕。
代码兜底规则（两种模式都生效，LLM 无法绕过）：
- 存在分析师失败 / 工具失败 / 无报告 → 追加「信息不完整」高风险项；
- 整体风险等级 = 风险清单中的最高等级（不是 LLM 直接给的）；
- blocked 恒为 False（一票否决留给后续明确规则，避免 LLM 误触发）。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.base import BaseAgent
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import (
    AgentStatus,
    AnalystReport,
    RiskItem,
    RiskLlmOutput,
    RiskReviewReport,
)
from graph.state import CouncilState
from tools.mock_tools import call_tool

logger = logging.getLogger(__name__)

_RISK_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：风控评审官。任务：基于分析师报告与行情快照对「{query}」做风险评估。"
    + "\n输出 JSON 对象，字段如下："
    + "\n- risks: 数组，每项含 title(字符串，风险名称)、level(字符串，取值仅限 低/中/高)、"
    + "description(字符串，风险描述)、mitigation(字符串，缓释措施)；"
    + "\n- comment: 字符串，评审意见（50~150 字）。"
    + "\n要求：中文；风险基于给定报告与行情中的事实，不编造；不要输出【stub】字样。"
)


class RiskAgent(BaseAgent[RiskReviewReport]):
    """风控评审：LLM 风险清单（报告 + 行情）+ 代码兜底规则（信息不完整必上高风险）。"""

    name = "risk_review"

    def execute(self, state: CouncilState) -> RiskReviewReport:
        query: str = state["query"]
        reports: list[AnalystReport] = state.get("analyst_reports", [])
        failed: int = sum(1 for r in reports if r.status is AgentStatus.FAILED)
        tool_failures: int = sum(len(r.tool_errors) for r in reports)

        # 行情快照（fail-open：非 ticker 查询 / 行情服务失败不阻断风控）
        market: dict[str, Any] | None = None
        market_error: str | None = None
        mkt = call_tool("get_market_data", symbol=query)
        if mkt.ok and mkt.data:
            market = mkt.data
        else:
            market_error = mkt.error

        if config_module.CONFIG.mock_mode:
            risks, comment, note = self._template_risks(query), None, None
        else:
            try:
                out: RiskLlmOutput = self._llm_risks(query, reports, market, market_error)
                risks, comment = out.risks, out.comment
                note = "真实数据 + DeepSeek 生成"
            except Exception as exc:  # noqa: BLE001 —— fail-open：LLM 失败降级模板
                logger.warning("风控 LLM 失败（降级模板）：%s", exc)
                risks = self._template_risks(query)
                comment = f"【降级模板】LLM 调用失败（{type(exc).__name__}），采用确定性基础风险清单。"
                note = "【降级模板】LLM 调用失败"

        # ---------- 代码兜底规则：信息不完整必上高风险（LLM 无法绕过）----------
        if (failed > 0 or tool_failures > 0 or not reports) and not any(
            r.title == "信息不完整" for r in risks
        ):
            desc_parts: list[str] = []
            if failed > 0:
                desc_parts.append(f"{failed} 位分析师失败")
            if tool_failures > 0:
                desc_parts.append(f"{tool_failures} 次工具调用失败")
            if not reports:
                desc_parts.append("分析师阶段无产出")
            risks.append(
                RiskItem(
                    title="信息不完整",
                    level="高",
                    description=f"{'、'.join(desc_parts)}，输入信息不完整。",
                    mitigation="补充数据源后复审。",
                )
            )
        # 整体风险等级 = 清单最高等级（确定性，不依赖 LLM）
        risk_level: Literal["低", "中", "高"] = (
            "高" if any(r.level == "高" for r in risks)
            else "中" if any(r.level == "中" for r in risks)
            else "低"
        )
        kwargs: dict[str, Any] = {}
        if note is not None:  # note 为 str 字段：mock 路径不传，用 schema 默认值
            kwargs["note"] = note
        return RiskReviewReport(
            agent=self.name,
            status=AgentStatus.OK,
            **kwargs,
            risk_level=risk_level,
            risks=risks,
            blocked=False,
            comment=comment or f"对「{query}」的风险评审完成。",
        )

    # ---------- LLM 主路径 ----------

    def _llm_risks(
        self,
        query: str,
        reports: list[AnalystReport],
        market: dict[str, Any] | None,
        market_error: str | None,
    ) -> RiskLlmOutput:
        """构造风控提示词（分析师报告摘要 + 行情快照）并调用 LLM。"""
        report_text = "\n".join(
            f"【{r.role}分析师｜置信度 {r.confidence}】摘要：{r.summary[:300]}\n"
            f"  论点：{'；'.join(k[:100] for k in r.key_points[:3])}"
            + (f"\n  工具失败留痕：{'；'.join(r.tool_errors[:3])}" if r.tool_errors else "")
            for r in reports
        ) or "（无分析师报告）"
        if market:
            market_text = (
                f"【行情快照】{market.get('name') or market['symbol']} {market['date']} "
                f"收盘 {market['price']}（{market['change_pct']:+}%），MA5={market['ma5']}，"
                f"MA20={market['ma20']}，MA60={market['ma60']}，RSI14={market['rsi']}，"
                f"MACD DIF={market['macd_dif']} / DEA={market['macd_dea']}"
                f" / 柱={market['macd_hist']:+}，成交量 {market['volume']} 手。"
            )
        else:
            market_text = f"（行情数据缺失：{market_error}，风控仅基于分析师报告）"
        system = _RISK_SYSTEM_PROMPT.format(query=query)
        human = (
            f"分析标的：{query}\n{market_text}\n分析师报告：\n{report_text}\n"
            f"请输出风险清单 JSON。"
        )
        return structured_call(RiskLlmOutput, system=system, human=human)

    # ---------- 降级模板路径 ----------

    def _template_risks(self, query: str) -> list[RiskItem]:
        """确定性基础风险清单（mock / LLM 降级共用）。"""
        return [
            RiskItem(
                title="市场风险",
                level="中",
                description=f"【stub】{query} 受大盘与流动性影响，价格波动存在不确定性。",
                mitigation="控制仓位，设置止损纪律。",
            ),
            RiskItem(
                title="估值风险",
                level="低",
                description=f"【stub】{query} 当前估值处于历史中位附近。",
                mitigation="分批建仓，等待安全边际。",
            ),
        ]

    def fallback(self, error: Exception) -> RiskReviewReport:
        """失败降级：风控自身失败按高风险处理（fail-open 但审慎）。"""
        return RiskReviewReport(
            agent=self.name,
            status=AgentStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            risk_level="高",
            risks=[
                RiskItem(
                    title="风控节点失败",
                    level="高",
                    description=f"风控评审未完成：{error}",
                    mitigation="人工复核。",
                )
            ],
            blocked=False,
            comment="",
        )
