"""分析师团队：基本面 / 技术面 / 情绪面 三名分析师，图级 Send 并行。

工具自采（自主性）：每名分析师先由 LLM 规划本轮要调用的工具集合
（plan_tools），代码做白名单校验（只允许本角色可用的工具）后执行，
最终实际调用的工具记录在 AnalystReport.tools_used。
- 角色允许集（_ROLE_ALLOWED_TOOLS）宽于默认集（_ROLE_DEFAULT_TOOLS）；
- mock 模式 / LLM 规划失败 → 直接使用默认集（fail-open，不阻断分析）。

各角色默认工具链（已全部接入真实数据源）：
- 基本面：get_financials（AkShare 东财估值 + 同花顺财务摘要）+
  get_news（AkShare 东财个股新闻）；
- 技术面：get_market_data（AkShare 前复权日线 → MA/RSI/MACD）；
- 情绪面：get_news（AkShare 个股新闻）+ research_rag（Milvus + bge
  检索研报，chunk 带来源元数据）。

角色分工（system prompt 强制，见 _ROLE_BOUNDARIES）：
- 每名分析师只产出自己维度的报告，不评论其他维度、不替其他角色决策；
- 最终投资评级由基金经理统一决策，分析师不输出评级。

产出路径（按模式）：
- MOCK_MODE=true：确定性占位模板（冒烟测试用）；
- MOCK_MODE=false：DeepSeek LLM 基于真实工具数据生成报告（note 标注
  「真实数据 + DeepSeek 生成」）；LLM 失败时 fail-open 降级：
  技术面 → 三信号投票规则模板，其余角色 → 占位模板 + 错误留痕，
  绝不在降级时伪造 LLM 产出。
"""

from __future__ import annotations

import logging
from typing import Any

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from agents.base import BaseAgent
from agents.llm import SYSTEM_ROLE_HINT, structured_call
from agents.schemas import (
    AgentStatus,
    AnalystLlmOutput,
    AnalystReport,
    ToolPlanLlmOutput,
)
from graph.state import CouncilState
from tools.mock_tools import call_tool

logger = logging.getLogger(__name__)

# 分析师角色（CLAUDE.md 架构要求：三分析师并行 Send）
ANALYST_ROLES: tuple[str, ...] = ("基本面", "技术面", "情绪面")

# 工具目录（工具自采的候选池：名称 → 能力描述，供 LLM 规划时参考）
TOOL_CATALOG: dict[str, str] = {
    "get_financials": "财务指标：PE/PB/ROE/营收增速/净利增速/报告期（东财 + 同花顺）",
    "get_market_data": "行情与技术指标：收盘价/涨跌幅/MA5/MA20/MA60/RSI14/MACD（前复权日线）",
    "get_news": "个股新闻与公告：标题/时间/摘要（东财资讯）",
    "research_rag": "研报向量检索：相关研报片段与来源元数据（Milvus + bge）",
}

# 角色允许调用的工具（白名单，代码强制校验：LLM 规划越界一律裁剪）
_ROLE_ALLOWED_TOOLS: dict[str, tuple[str, ...]] = {
    "基本面": ("get_financials", "get_news", "research_rag"),
    "技术面": ("get_market_data", "get_news"),
    "情绪面": ("get_news", "research_rag", "get_market_data"),
}

# 角色默认工具（mock 模式 / 规划失败时使用，也是最小必要集）
_ROLE_DEFAULT_TOOLS: dict[str, tuple[str, ...]] = {
    "基本面": ("get_financials", "get_news"),
    "技术面": ("get_market_data",),
    "情绪面": ("get_news", "research_rag"),
}

_TOOL_PLAN_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：{role}分析师的工具规划器。任务：判断分析「{query}」需要调用哪些数据工具。"
    + "\n可选工具（只能从这里选，不得编造工具名）：\n{catalog}"
    + "\n输出 JSON 对象，字段如下："
    + "\n- tools: 字符串数组，按重要性排序的工具名（1~4 个）；"
    + "\n- reason: 字符串，一句话说明为什么需要这些工具。"
    + "\n要求：中文；优先选与本角色维度直接相关的工具；不要输出【stub】字样。"
)

# 各角色职责边界：只产出自己维度、不替别人决策（system prompt 强制）
_ROLE_BOUNDARIES: dict[str, str] = {
    "基本面": "职责边界：只基于财务/经营数据做基本面分析；不评价股价走势与技术指标，"
              "不评论市场情绪；不给出最终投资评级（评级由基金经理统一决策）。",
    "技术面": "职责边界：只基于行情与技术指标做技术面分析；不讨论基本面与新闻情绪；"
              "不给出最终投资评级（评级由基金经理统一决策）。",
    "情绪面": "职责边界：只基于新闻与研报观点做市场情绪分析；不预测价格走势；"
              "不给出最终投资评级（评级由基金经理统一决策）。",
}

# 各角色系统提示词模板（内容型要求，数据在 human 中给出）
_ROLE_SYSTEM_PROMPT: str = (
    SYSTEM_ROLE_HINT
    + "\n你的角色：{role}分析师。任务：基于给定数据对「{query}」输出一份分析报告。"
    + "\n{boundary}"
    + "\n输出 JSON 对象，字段如下："
    + "\n- summary: 字符串，结论摘要（80~200 字）；"
    + "\n- key_points: 字符串数组，3 条左右核心论点，每条须引用给定数据中的具体数字；"
    + "\n- confidence: 0~1 的数字，你对本次判断的置信度。"
    + "\n要求：中文输出；观点必须基于给定数据，不得编造数据；"
    + "数据缺失时明确说明「数据缺失」；不要输出【stub】字样。"
)


class AnalystAgent(BaseAgent[AnalystReport]):
    """单个分析师：工具取数 → LLM 生成报告（失败降级模板），工具失败只留痕不崩溃。"""

    def __init__(self, role: str, confidence: float) -> None:
        """初始化分析师。

        参数：
            role:       分析师角色（基本面 / 技术面 / 情绪面）。
            confidence: 占位置信度（mock 模式 / LLM 降级模板时使用；
                        真实模式由 LLM 输出覆盖）。
        """
        self.name = f"analyst_{role}"
        self.role = role
        self.confidence = confidence

    def plan_tools(self, query: str) -> tuple[list[str], str | None, str | None]:
        """工具自采规划：LLM 决定调用哪些工具，代码白名单校验后返回。

        返回：
            (tools, reason, plan_error)——
            tools:      实际执行的工具列表（已与角色白名单求交，保证非空）；
            reason:     LLM 给出的规划理由（mock / 失败时为 None）；
            plan_error: 规划失败留痕（fail-open，回退默认工具集）。

        安全边界：LLM 只能从角色白名单（_ROLE_ALLOWED_TOOLS）中选，
        越界/编造的工具名一律裁剪；裁剪后为空则回退默认工具集。
        """
        defaults: list[str] = list(_ROLE_DEFAULT_TOOLS.get(self.role, ()))
        if config_module.CONFIG.mock_mode:
            return defaults, None, None
        allowed: set[str] = set(_ROLE_ALLOWED_TOOLS.get(self.role, tuple(defaults)))
        catalog: str = "\n".join(
            f"- {name}: {desc}" for name, desc in TOOL_CATALOG.items() if name in allowed
        )
        try:
            plan: ToolPlanLlmOutput = structured_call(
                ToolPlanLlmOutput,
                system=_TOOL_PLAN_SYSTEM_PROMPT.format(role=self.role, query=query, catalog=catalog),
                human=f"分析目标：{query}\n请输出本轮工具规划 JSON。",
            )
        except Exception as exc:  # noqa: BLE001 —— fail-open：规划失败回退默认工具
            logger.warning("分析师 [%s] 工具规划失败（回退默认工具集）：%s", self.name, exc)
            return defaults, None, f"工具规划失败（已回退默认工具集）: {type(exc).__name__}: {exc}"
        selected: list[str] = [t for t in plan.tools if t in allowed]
        if not selected:
            return defaults, plan.reason, "工具规划结果全部越界（已回退默认工具集）"
        return selected[:4], plan.reason, None

    def collect_context(
        self, query: str, tools: list[str] | None = None
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """按指定工具收集数据（execute 与评测评审复用同一「给定数据」）。

        参数：
            query: 分析目标。
            tools: 本次要调用的工具列表；None 表示使用角色默认工具集。

        返回：
            (context, tool_errors, sources)——
            context:     工具数据的原始 dict（market / rag_chunks / financials / news）；
            tool_errors: 工具失败留痕（fail-open，不中断）；
            sources:     依据来源列表（过程可追溯）。
        评测用途（eval/judge.py）：LLM-as-Judge 评审「论点是否有据」时，
        用本方法（传入报告记录的 tools_used）+ describe_context 重建同一份给定数据。
        """
        selected: list[str] = (
            list(tools) if tools is not None else list(_ROLE_DEFAULT_TOOLS.get(self.role, ()))
        )
        tool_errors: list[str] = []
        sources: list[str] = []
        context: dict[str, Any] = {}
        news_limit: int = 3 if self.role == "情绪面" else 2
        for name in selected:
            if name == "get_financials":
                context["financials"] = self._collect_financials(query, tool_errors, sources)
            elif name == "get_market_data":
                market = call_tool("get_market_data", symbol=query)
                if market.ok and market.data:
                    context["market"] = market.data
                    sources.append(self._market_source(market.data))
                else:
                    tool_errors.append(f"get_market_data: {market.error}")
            elif name == "get_news":
                context["news"] = self._collect_news(query, news_limit, tool_errors, sources)
            elif name == "research_rag":
                rag = call_tool("research_rag", query=query, top_k=3)
                if rag.ok and rag.data:
                    chunks: list[dict[str, Any]] = rag.data["chunks"]
                    context["rag_chunks"] = chunks
                    sources.append(f"研报检索 {len(chunks)} 条（Milvus）")
                    for c in chunks:
                        meta = c.get("metadata") or {}
                        label = str(meta.get("source") or c.get("doc_name"))
                        sources.append(f"　└ {label}（相似度 {c.get('score')}）")
                else:
                    tool_errors.append(f"research_rag: {rag.error}")
        return context, tool_errors, sources

    def execute(self, state: CouncilState) -> AnalystReport:
        query: str = state["query"]

        # ---------- 1) 工具自采 + 取数（mock 拿占位数据，真实拿真实数据）----------
        tools, plan_reason, plan_error = self.plan_tools(query)
        context, tool_errors, sources = self.collect_context(query, tools=tools)
        if plan_error:
            tool_errors.append(plan_error)
        if plan_reason:
            sources.append(f"工具自采：{', '.join(tools)}（规划理由：{plan_reason}）")

        # ---------- 2) 产出：mock → 模板；真实 → LLM 主路径 + 模板降级 ----------
        if config_module.CONFIG.mock_mode:
            report = self._template_report(query, context, tool_errors, sources, degraded=False)
            return report.model_copy(update={"tools_used": tools})
        try:
            out: AnalystLlmOutput = self._llm_report(query, context)
        except Exception as exc:  # noqa: BLE001 —— fail-open：LLM 失败降级模板
            logger.warning("分析师 [%s] LLM 调用失败（降级模板）：%s", self.name, exc)
            tool_errors.append(f"LLM 调用失败（降级模板）: {type(exc).__name__}: {exc}")
            report = self._template_report(query, context, tool_errors, sources, degraded=True)
            return report.model_copy(update={"tools_used": tools})
        self.confidence = out.confidence  # 真实模式：置信度由 LLM 输出
        return AnalystReport(
            agent=self.name,
            status=AgentStatus.OK,
            note="真实数据 + DeepSeek 生成",
            role=self.role,
            summary=out.summary,
            key_points=out.key_points,
            confidence=out.confidence,
            sources=sources,
            tools_used=tools,
            tool_errors=tool_errors,
        )

    # ---------- LLM 主路径 ----------

    def _llm_report(self, query: str, context: dict[str, Any]) -> AnalystLlmOutput:
        """构造提示词并调用 DeepSeek 生成结构化报告。"""
        system = _ROLE_SYSTEM_PROMPT.format(
            role=self.role, query=query, boundary=_ROLE_BOUNDARIES.get(self.role, "")
        )
        human = (
            f"查询目标：{query}\n"
            f"给定数据：\n{self.describe_context(context)}"
            f"\n请输出你的分析报告 JSON。"
        )
        return structured_call(AnalystLlmOutput, system=system, human=human)

    def describe_context(self, context: dict[str, Any]) -> str:
        """把工具数据压缩为提示词上下文（截断防止过长）。

        评测复用：eval/judge.py 用同一方法重建「给定数据」文本供评审对比。
        """
        lines: list[str] = []
        market = context.get("market")
        if market:
            lines.append(
                "【行情与技术指标（AkShare 前复权）】"
                f"{market.get('name') or market['symbol']}，最新交易日 {market['date']}："
                f"收盘 {market['price']}（涨跌 {market['change_pct']:+}%），"
                f"MA5={market['ma5']}，MA20={market['ma20']}，MA60={market['ma60']}，"
                f"RSI14={market['rsi']}，MACD DIF={market['macd_dif']} / DEA={market['macd_dea']}"
                f" / 柱={market['macd_hist']:+}，成交量 {market['volume']} 手；"
                f"近 20 日年化波动率 {market.get('volatility_20d_pct', 0)}%，"
                f"近 60 日最大回撤 {market.get('max_drawdown_60d_pct', 0)}%，"
                f"ATR14 占价 {market.get('atr_pct', 0)}%，"
                f"量比 {market.get('volume_ratio_5d', 1)}，"
                f"60 日区间位置 {market.get('position_60d', 0.5)}。"
            )
        for c in context.get("rag_chunks", []):
            meta = c.get("metadata") or {}
            text = str(c["chunk_text"])[:300]
            lines.append(
                f"【研报片段｜{meta.get('source') or c.get('doc_name')}｜"
                f"ticker={meta.get('ticker')} date={meta.get('date')}｜相似度 {c.get('score')}】{text}"
            )
        fin = context.get("financials")
        if fin:
            lines.append(
                "【财务指标】"
                f"PE={fin['pe']}，PB={fin['pb']}，ROE={fin['roe']}，"
                f"营收增速 {fin['revenue_growth']}%，净利增速 {fin['net_profit_growth']}%，"
                f"报告期 {fin['report_period']}。"
            )
        for n in context.get("news", []):
            lines.append(f"【新闻｜{n['source']}｜{n['published_at']}】{n['title']}：{n['snippet'][:120]}")
        if not lines:
            return "（无可用工具数据：相关工具失败或未接入，请明确说明数据缺失并降低置信度）"
        return "\n".join(lines)

    # ---------- 降级模板路径（mock / LLM 失败） ----------

    def _template_report(
        self,
        query: str,
        context: dict[str, Any],
        tool_errors: list[str],
        sources: list[str],
        degraded: bool,
    ) -> AnalystReport:
        """确定性模板报告。技术面用三信号投票规则模板（数据驱动），其余角色占位文本。"""
        prefix = "【降级模板】" if degraded else "【stub】"
        if self.role == "技术面" and context.get("market"):
            tech = self._build_technical_summary(context["market"])
            self.confidence = float(tech["confidence"])
            return AnalystReport(
                agent=self.name,
                status=AgentStatus.OK,
                note=("真实数据 + 规则模板（LLM 降级，见 tool_errors）" if degraded
                      else "【stub】占位行情 + 规则模板"),
                role=self.role,
                summary=tech["summary"],
                key_points=tech["key_points"],
                confidence=self.confidence,
                sources=sources,
                tool_errors=tool_errors,
            )
        kwargs: dict[str, Any] = {}
        if degraded:  # note 为 str 字段：不降级时不传，用 schema 默认值
            kwargs["note"] = "【降级模板】LLM 调用失败，见 tool_errors"
        return AnalystReport(
            agent=self.name,
            status=AgentStatus.OK,
            **kwargs,
            role=self.role,
            summary=(
                f"{prefix}从「{self.role}」维度对「{query}」的占位分析："
                f"整体判断为中性偏积极，需结合辩论与风控环节收敛。"
            ),
            key_points=[
                f"{prefix}{self.role} 维度机会点 1：{query} 中长期逻辑未见破坏。",
                f"{prefix}{self.role} 维度机会点 2：可小仓位跟踪验证。",
                f"{prefix}{self.role} 维度关注点：需真实数据支撑后复核。",
            ],
            confidence=self.confidence,
            sources=sources,
            tool_errors=tool_errors,
        )

    def _build_technical_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        """技术面规则模板：三信号投票（均线趋势 / RSI / MACD），置信度=偏多信号占比。"""
        price = float(data["price"])
        ma5, ma20 = float(data["ma5"]), float(data["ma20"])
        rsi = float(data["rsi"])
        dif, dea, hist = (
            float(data["macd_dif"]),
            float(data["macd_dea"]),
            float(data["macd_hist"]),
        )
        symbol = str(data["symbol"])
        name = str(data.get("name") or symbol)

        # 信号 1：均线趋势
        if ma5 > ma20 and price > ma20:
            trend = ("多头排列：MA5 高于 MA20 且价格站上 MA20，短期趋势偏强", 1)
        elif ma5 < ma20 and price < ma20:
            trend = ("空头排列：MA5 低于 MA20 且价格跌破 MA20，短期趋势偏弱", -1)
        else:
            trend = ("均线缠绕：MA5 与 MA20 相互穿插，趋势方向不明", 0)
        # 信号 2：RSI 超买超卖
        if rsi >= 70:
            rsi_judge = ("RSI14 处于超买区（≥70），短线回调风险积聚", -1)
        elif rsi <= 30:
            rsi_judge = ("RSI14 处于超卖区（≤30），存在超跌修复动能", 1)
        else:
            rsi_judge = ("RSI14 处于中性区间（30~70），动量不极端", 0)
        # 信号 3：MACD 多空
        if hist > 0 and dif > dea:
            macd_judge = ("MACD 红柱且 DIF 高于 DEA，多头动能占优", 1)
        elif hist < 0 and dif < dea:
            macd_judge = ("MACD 绿柱且 DIF 低于 DEA，空头动能占优", -1)
        else:
            macd_judge = ("MACD 柱状与 DIF/DEA 关系背离，动能信号混杂", 0)

        signals = [trend, rsi_judge, macd_judge]
        bullish = sum(1 for _, v in signals if v > 0)
        bearish = sum(1 for _, v in signals if v < 0)
        confidence = round(bullish / len(signals), 2)
        if bullish > bearish:
            overall = "偏多"
        elif bearish > bullish:
            overall = "偏空"
        else:
            overall = "中性"

        summary = (
            f"{name}（{symbol}）最新收盘 {price:.2f}（{float(data['change_pct']):+.2f}%），"
            f"MA5={ma5:.2f} / MA20={ma20:.2f} / MA60={float(data['ma60']):.2f}；"
            f"RSI14={rsi:.1f}；MACD DIF={dif:.4f} / DEA={dea:.4f} / 柱={hist:+.4f}；"
            f"近 20 日年化波动率 {data.get('volatility_20d_pct', 0)}%，"
            f"近 60 日最大回撤 {data.get('max_drawdown_60d_pct', 0)}%，"
            f"量比 {data.get('volume_ratio_5d', 1)}，"
            f"60 日区间位置 {data.get('position_60d', 0.5)}。"
            f"三信号投票 {bullish} 多 / {bearish} 空，综合技术面判断：{overall}。"
        )
        key_points = [
            f"均线：{trend[0]}。",
            f"动量：{rsi_judge[0]}。",
            f"MACD：{macd_judge[0]}。",
        ]
        vol_pct = data.get("volatility_20d_pct")
        dd_pct = data.get("max_drawdown_60d_pct")
        if vol_pct is not None and dd_pct is not None:
            key_points.append(
                f"风险量化：近 20 日年化波动率 {vol_pct}%，近 60 日最大回撤 {dd_pct}%。"
            )
        source = (
            f"行情参考（{data.get('note')}）：{name} {data['date']} 收盘 {price}，"
            f"MA5={ma5}，RSI={rsi}，MACD柱={hist}"
        )
        return {
            "summary": summary,
            "key_points": key_points,
            "confidence": confidence,
            "source": source,
        }

    # ---------- 工具调用辅助 ----------

    def _market_source(self, data: dict[str, Any]) -> str:
        """行情工具数据的来源描述（真实数据直接引用）。"""
        name = str(data.get("name") or data["symbol"])
        return (
            f"行情参考（{data.get('note')}）：{name} {data['date']} 收盘 {data['price']}，"
            f"MA5={data['ma5']}，RSI={data['rsi']}，MACD柱={data['macd_hist']}"
        )

    def _collect_news(
        self, query: str, max_results: int, tool_errors: list[str], sources: list[str]
    ) -> list[dict[str, Any]]:
        """调用 get_news；失败 fail-open 留痕。返回新闻条目（成功时）。"""
        news = call_tool("get_news", query=query, max_results=max_results)
        if news.ok and news.data:
            items: list[dict[str, Any]] = news.data.get("items", [])
            sources.extend(f"新闻：{it['title']}（{it['source']}）" for it in items)
            return items
        tool_errors.append(f"get_news: {news.error}")
        return []

    def _collect_financials(
        self, query: str, tool_errors: list[str], sources: list[str]
    ) -> dict[str, Any] | None:
        """调用 get_financials；失败 fail-open 留痕。返回财务指标（成功时）。"""
        fin = call_tool("get_financials", symbol=query)
        if fin.ok and fin.data:
            m: dict[str, Any] = fin.data["metrics"]
            sources.append(
                f"财务指标（{m['symbol']}，报告期 {m['report_period']}，"
                f"PE={m['pe']}，PB={m['pb']}，ROE={m['roe']}）"
            )
            return m
        tool_errors.append(f"get_financials: {fin.error}")
        return None

    def fallback(self, error: Exception) -> AnalystReport:
        """失败降级：产出合法的失败态报告，供下游统计与留痕（fail-open）。"""
        return AnalystReport(
            agent=self.name,
            status=AgentStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            role=self.role,
            summary="",
            key_points=[],
            confidence=0.0,
            sources=[],
        )