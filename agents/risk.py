"""风控评审：基于分析师报告 + 行情快照生成风险清单，并执行代码级一票否决规则。

产出路径（按模式）：
- MOCK_MODE=true：确定性占位规则（冒烟测试用）；
- MOCK_MODE=false：DeepSeek LLM 基于分析师报告 + 行情快照
  （get_market_data，fail-open：行情失败不阻断风控，仅提示数据缺失）
  生成风险清单；LLM 失败时降级为确定性基础风险清单 + 错误留痕。
代码兜底规则（两种模式都生效，LLM 无法绕过）：
- 存在分析师失败 / 工具失败 / 无报告 → 追加「信息不完整」高风险项；
- 整体风险等级 = 风险清单中的最高等级（不是 LLM 直接给的）；
- 一票否决（blocked）由确定性规则判定，LLM 无权决定：
    V1 分析师阶段无任何有效报告（信息完全不可用）→ 否决；
    V2 技术面破位：空头排列（MA5<MA20<MA60）+ MACD 柱为负 + RSI 低于阈值 → 否决；
    V3 高波动：近 20 日年化波动率与近 60 日最大回撤同时超阈值 → 否决
       （波动与回撤双高说明风险已实质兑现，不适合给出积极结论）；
  否决一旦触发，经理层评级上限被压到「卖出」（见 agents/manager.py）。
确定性风险项（代码追加，LLM 无权删改）：
- 极端波动：单日涨跌幅绝对值 ≥ 阈值；
- ATR 放大：ATR14 占收盘价比例 ≥ 阈值（日内波幅异常，止损空间被压缩）；
- 量价背离：价格处于近 60 日区间高位但近 5 日缩量（上攻缺乏成交量确认）。
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
        ok_reports: list[AnalystReport] = [r for r in reports if r.status is AgentStatus.OK]
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
        # ---------- 代码兜底规则：极端波动必上高风险（LLM 无法绕过）----------
        extreme_pct: float = config_module.CONFIG.risk_extreme_move_pct
        if market:
            try:
                change_pct = float(market["change_pct"])
            except (KeyError, TypeError, ValueError):
                change_pct = 0.0
            if abs(change_pct) >= extreme_pct and not any(
                r.title == "极端波动" for r in risks
            ):
                risks.append(
                    RiskItem(
                        title="极端波动",
                        level="高",
                        description=(
                            f"单日涨跌幅 {change_pct:+}%，绝对值已达 {extreme_pct}%，"
                            f"存在流动性与情绪冲击风险。"
                        ),
                        mitigation="避免追涨杀跌，等待波动收敛后再评估。",
                    )
                )
        # ---------- 代码兜底规则：量化指标风险项（LLM 无法绕过）----------
        for item in self._quant_risk_items(market):
            if not any(r.title == item.title for r in risks):
                risks.append(item)
        # 整体风险等级 = 清单最高等级（确定性，不依赖 LLM）
        risk_level: Literal["低", "中", "高"] = (
            "高" if any(r.level == "高" for r in risks)
            else "中" if any(r.level == "中" for r in risks)
            else "低"
        )
        # ---------- 代码级一票否决（LLM 无权触发/绕过）----------
        blocked: bool
        block_reason: str | None
        blocked, block_reason = self._veto(
            reports=reports, ok_count=len(ok_reports), market=market
        )
        if blocked:
            # 否决必须有对应的高风险项，保证报告自洽（清单与结论不矛盾）
            if not any(r.level == "高" for r in risks):
                risks.append(
                    RiskItem(
                        title="一票否决触发",
                        level="高",
                        description=f"风控规则否决：{block_reason}",
                        mitigation="暂停新增仓位，人工复核后再决策。",
                    )
                )
            risk_level = "高"
        kwargs: dict[str, Any] = {}
        if note is not None:  # note 为 str 字段：mock 路径不传，用 schema 默认值
            kwargs["note"] = note
        return RiskReviewReport(
            agent=self.name,
            status=AgentStatus.OK,
            **kwargs,
            risk_level=risk_level,
            risks=risks,
            blocked=blocked,
            block_reason=block_reason,
            comment=comment or f"对「{query}」的风险评审完成。",
        )

    def _veto(
        self,
        reports: list[AnalystReport],
        ok_count: int,
        market: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        """确定性一票否决规则：返回 (是否否决, 触发原因)。

        V1 信息完全不可用：没有任何成功的分析师报告 → 否决
           （不掌握信息时不应给出任何投资结论，宁可不出结论）。
        V2 技术面破位：MA5<MA20<MA60（空头排列）+ MACD 柱为负 + RSI 低于阈值
           → 否决（趋势与动量双双恶化，风险已实质兑现而非潜在）。
        V3 高波动：近 20 日年化波动率 ≥ 阈值 且 近 60 日最大回撤 ≥ 阈值
           → 否决（波动与回撤双高，持仓风险超出可承受范围）。
        规则阈值来自配置，可由 .env 调整；行情缺失时 V2/V3 自动跳过（fail-open）。
        """
        cfg = config_module.CONFIG
        if cfg.risk_veto_on_missing_data and ok_count == 0:
            return True, f"V1 分析师阶段无有效报告（共 {len(reports)} 份，全部失败），信息不可用"
        if not market:
            return False, None
        try:
            ma5 = float(market["ma5"])
            ma20 = float(market["ma20"])
            ma60 = float(market["ma60"])
            rsi = float(market["rsi"])
            macd_hist = float(market["macd_hist"])
        except (KeyError, TypeError, ValueError):
            ma5 = ma20 = ma60 = rsi = macd_hist = None  # type: ignore[assignment]
        if (
            ma5 is not None
            and ma20 is not None
            and ma60 is not None
            and rsi is not None
            and macd_hist is not None
            and ma5 < ma20 < ma60
            and macd_hist < 0
            and rsi < cfg.risk_veto_rsi_floor
        ):
            return (
                True,
                f"V2 技术面破位：MA5({ma5})<MA20({ma20})<MA60({ma60}) 空头排列，"
                f"MACD 柱为负（{macd_hist}），RSI={rsi} 低于阈值 {cfg.risk_veto_rsi_floor}",
            )
        # V3 高波动否决：波动率与回撤同时超标（任一指标缺失则跳过，fail-open）
        try:
            vol_pct = float(market["volatility_20d_pct"])
            drawdown_pct = abs(float(market["max_drawdown_60d_pct"]))
        except (KeyError, TypeError, ValueError):
            return False, None
        if vol_pct >= cfg.risk_veto_annual_vol_pct and drawdown_pct >= cfg.risk_veto_drawdown_pct:
            return (
                True,
                f"V3 高波动：近 20 日年化波动率 {vol_pct}% ≥ {cfg.risk_veto_annual_vol_pct}% "
                f"且近 60 日最大回撤 {drawdown_pct}% ≥ {cfg.risk_veto_drawdown_pct}%，"
                f"风险已实质兑现",
            )
        return False, None

    # ---------- 量化指标确定性规则 ----------

    def _quant_risk_items(self, market: dict[str, Any] | None) -> list[RiskItem]:
        """由行情量化指标生成确定性风险项（阈值全部来自配置，可审计、可调）。

        - ATR 放大（高）：ATR14 占收盘价比例 ≥ 阈值，日内波幅异常、止损空间被压缩；
        - 量价背离（中）：价格处于近 60 日区间高位但近 5 日缩量，上攻缺乏量能确认。
        行情缺失或字段非法时返回空列表（fail-open，不臆造风险）。
        """
        if not market:
            return []
        items: list[RiskItem] = []
        cfg = config_module.CONFIG

        try:
            atr_pct = float(market.get("atr_pct") or 0.0)
        except (TypeError, ValueError):
            atr_pct = 0.0
        if atr_pct >= cfg.risk_atr_spike_pct:
            items.append(
                RiskItem(
                    title="ATR 放大",
                    level="高",
                    description=(
                        f"ATR14 占收盘价 {atr_pct}%（阈值 {cfg.risk_atr_spike_pct}%），"
                        f"日内波幅异常放大，止损空间被压缩。"
                    ),
                    mitigation="下调仓位并放宽止损带，避免被日内噪声扫损。",
                )
            )

        try:
            position = float(market.get("position_60d", 0.5))
            vol_ratio = float(market.get("volume_ratio_5d", 1.0))
        except (TypeError, ValueError):
            position, vol_ratio = 0.5, 1.0
        if position >= cfg.risk_divergence_position and vol_ratio <= cfg.risk_divergence_ratio:
            items.append(
                RiskItem(
                    title="量价背离",
                    level="中",
                    description=(
                        f"价格位于近 60 日区间 {position:.0%} 位置"
                        f"（阈值 {cfg.risk_divergence_position:.0%}），"
                        f"但近 5 日量比 {vol_ratio}（阈值 {cfg.risk_divergence_ratio}）显示缩量，"
                        f"上攻缺乏成交量确认。"
                    ),
                    mitigation="观察放量突破确认后再加仓，缩量滞涨时以减仓为主。",
                )
            )
        return items

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
                f" / 柱={market['macd_hist']:+}，成交量 {market['volume']} 手；"
                f"近 20 日年化波动率 {market.get('volatility_20d_pct', 0)}%，"
                f"近 60 日最大回撤 {market.get('max_drawdown_60d_pct', 0)}%，"
                f"ATR14={market.get('atr14', 0)}（占价 {market.get('atr_pct', 0)}%），"
                f"量比 {market.get('volume_ratio_5d', 1)}，"
                f"60 日区间位置 {market.get('position_60d', 0.5)}。"
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