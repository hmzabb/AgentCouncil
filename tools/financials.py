"""财务指标工具真实实现（AkShare）：估值（PE/PB）+ 盈利与成长（ROE/增速）。

数据源（2026-11-02 实测，AkShare 1.20.22）：
1. 估值：ak.stock_value_em(symbol)（东方财富估值，按交易日行，取最新一行）：
   PE(TTM) / PE(静) / 市净率；
2. 盈利与成长：ak.stock_financial_abstract_ths(symbol, indicator="按报告期")
   （同花顺财务摘要，按报告期行，取最新报告期）：净资产收益率、
   营业总收入同比增长率、净利润同比增长率、报告期。

实测坑位记录（2026-11-02）：
- ak.stock_a_indicator_lg 在本版本不存在（乐咕接口已移除）；
- ak.stock_financial_analysis_indicator_em（东财财务指标）请求返回空
  data（result.data 为 None），不可用；
- 同花顺摘要的数值是带单位字符串（"24.47%"、"644.67亿"），百分比字段
  须去掉 % 再转 float；个别股票某些字段可能为 NaN（数据缺失），
  缺失即 fail-open 报错并指明字段，不伪造数值。

fail-open 约定：任何异常（网络失败 / 接口变更 / 字段缺失）由
BaseTool.invoke 包装为 ToolResult(ok=False)，下游留痕，不伪造财务数据。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

import config as config_module
from tools.base import BaseTool
from tools.market_data import normalize_symbol

# ------------------------------------------------------------------ schema


class GetFinancialsArgs(BaseModel):
    """财务指标入参。"""

    symbol: str = Field(min_length=1, description="证券代码")


class FinancialMetrics(BaseModel):
    """核心财务指标（mock 与真实实现共用本 schema，保证字段对齐）。"""

    symbol: str
    pe: float = Field(description="市盈率（TTM，缺失时取静态）")
    pb: float = Field(description="市净率")
    roe: float = Field(description="净资产收益率（%，最新报告期）")
    revenue_growth: float = Field(description="营业总收入同比增长率（%）")
    net_profit_growth: float = Field(description="净利润同比增长率（%）")
    report_period: str = Field(description="最新报告期 YYYY-MM-DD")
    note: str = Field(default="【stub】假数据", description="数据来源说明")


class GetFinancialsResult(BaseModel):
    """财务指标结果。"""

    metrics: FinancialMetrics


# ------------------------------------------------------------------ 解析辅助


def _parse_pct(value: Any) -> float | None:
    """把 '24.47%' / 24.47 / NaN 解析为 float；缺失返回 None（不伪造 0）。"""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text: str = str(value).strip()
    if text in {"", "--", "None", "nan"}:
        return None
    return float(text.replace("%", ""))


def _pick(row: Any, *candidates: str) -> Any:
    """按候选列名读取（接口列名随版本变化时兜底）；无候选列返回 None。"""
    for name in candidates:
        if name in row.index:
            return row[name]
    return None


# ------------------------------------------------------------------ 工具实现


class GetFinancialsTool(BaseTool):
    """财务指标工具（真实实现：AkShare 东财估值 + 同花顺财务摘要）。"""

    name = "get_financials"
    Args = GetFinancialsArgs
    Result = GetFinancialsResult

    def _execute(self, args: GetFinancialsArgs) -> GetFinancialsResult:
        if config_module.CONFIG.demo_tool_failure:
            # 演示 fail-open：模拟上游财务数据服务超时（与 mock 版行为一致）
            raise TimeoutError("【演示】模拟财务数据服务超时（fail-open 测试）")
        import akshare as ak  # 延迟导入：仅真实模式使用

        code: str = normalize_symbol(args.symbol)

        # ---------- 1) 估值（东财，最新交易日） ----------
        value_df: Any = ak.stock_value_em(symbol=code)
        if value_df is None or value_df.empty:
            raise RuntimeError(f"AkShare 未返回 {code} 的估值数据（代码错误？）")
        last: Any = value_df.iloc[-1]
        # PE 优先 TTM，缺失（亏损股 TTM 为 NaN）时退回静态
        pe: float | None = _parse_pct(_pick(last, "PE(TTM)")) or _parse_pct(_pick(last, "PE(静)"))
        pb: float | None = _parse_pct(_pick(last, "市净率"))

        # ---------- 2) 盈利与成长（同花顺摘要，最新报告期） ----------
        ths: Any = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if ths is None or ths.empty:
            raise RuntimeError(f"AkShare 未返回 {code} 的财务摘要（代码错误？）")
        row: Any = ths.iloc[-1]
        roe: float | None = _parse_pct(_pick(row, "净资产收益率"))
        revenue_growth: float | None = _parse_pct(_pick(row, "营业总收入同比增长率"))
        net_profit_growth: float | None = _parse_pct(_pick(row, "净利润同比增长率"))
        report_period: str = str(_pick(row, "报告期") or "")[:10]

        missing: list[str] = [
            name for name, value in {
                "PE": pe, "PB": pb, "ROE": roe,
                "营业总收入同比增长率": revenue_growth,
                "净利润同比增长率": net_profit_growth,
                "报告期": report_period or None,
            }.items() if value is None
        ]
        if missing:
            raise RuntimeError(
                f"{code} 以下财务字段缺失（接口返回空值，不伪造数据）: {'、'.join(missing)}"
            )

        return GetFinancialsResult(
            metrics=FinancialMetrics(
                symbol=code,
                pe=round(pe, 2),  # type: ignore[arg-type] —— 上方 missing 检查已排除 None
                pb=round(pb, 2),  # type: ignore[arg-type]
                roe=round(roe, 2),  # type: ignore[arg-type]
                revenue_growth=round(revenue_growth, 2),  # type: ignore[arg-type]
                net_profit_growth=round(net_profit_growth, 2),  # type: ignore[arg-type]
                report_period=report_period,  # type: ignore[arg-type]
                note="真实数据（AkShare：东财估值 + 同花顺财务摘要）",
            )
        )
