"""行情工具真实实现（AkShare）：日线历史 → 现价 + MA / RSI / MACD 指标。

实测坑位记录（2026-11-02，AkShare 东方财富接口）：
1. 返回 DataFrame 列名随版本中英文混用（中文「日期/开盘/收盘」或英文
   Date/Open/Close），本模块用候选名双映射读取，不依赖单一列名常量；
2. 入参只接受 6 位纯数字代码（600519），不接受 .SH/.SZ 后缀，须归一化；
3. 历史行情必须用 adjust="qfq"（前复权），否则分红除权会造成均线/指标
   在除权日跳变失真；
4. 成交量单位为「手」（1 手 = 100 股），非股数。

fail-open 约定：任何异常（网络失败 / 停牌 / 代码非法）由 BaseTool.invoke
包装为 ToolResult(ok=False)，下游分析师降级留痕，不伪造行情数据。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import config as config_module
from pydantic import BaseModel, Field

from tools.base import BaseTool

# ------------------------------------------------------------------ schema


class GetMarketDataArgs(BaseModel):
    """行情查询入参。"""

    symbol: str = Field(min_length=1, description="证券代码，如 600519.SH / 600519")


class MarketDataResult(BaseModel):
    """行情快照 + 技术指标（mock 与真实实现共用本 schema，保证字段对齐）。"""

    symbol: str
    name: str
    date: str = Field(description="最新交易日 YYYY-MM-DD")
    price: float = Field(description="最新收盘价（前复权口径）")
    change_pct: float
    volume: int = Field(description="成交量（手）")
    currency: str
    ma5: float
    ma20: float
    ma60: float
    rsi: float = Field(ge=0.0, le=100.0, description="RSI14")
    macd_dif: float
    macd_dea: float
    macd_hist: float = Field(description="MACD 柱（2*(DIF-DEA)）")
    note: str = Field(default="", description="数据来源说明（mock 时为【stub】标注）")


# ------------------------------------------------------------------ 数据获取与指标计算


def normalize_symbol(symbol: str) -> str:
    """把 600519.SH / SH600519 / 600519 归一化为 6 位纯数字代码。

    AkShare 接口只接受纯数字；非 6 位数字（如美股代码）直接报错（fail-open）。
    """
    digits: str = re.sub(r"\D", "", symbol)
    if len(digits) != 6:
        raise ValueError(f"仅支持 6 位 A 股代码（AkShare 接口限制），收到: {symbol!r}")
    return digits


def fetch_history(symbol: str, days: int) -> "Any":  # 返回 pd.DataFrame，延迟导入不写死类型
    """通过 AkShare 拉取前复权日线历史（近 days 个自然日窗口）。"""
    import akshare as ak  # 延迟导入：akshare 依赖较多，仅真实模式使用

    code: str = normalize_symbol(symbol)
    end: str = datetime.now().strftime("%Y%m%d")
    # 自然日 × 1.6 ≈ 交易日数（扣除周末与节假日），保证窗口内足够 60 日均线
    start: str = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")
    df: Any = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
    )
    if df is None or df.empty:
        raise RuntimeError(f"AkShare 未返回 {code} 的行情数据（代码错误 / 停牌 / 无交易？）")
    return df


def _pick_column(df: Any, *candidates: str) -> Any:
    """按候选名（中文/英文兼容）读取列：AkShare 列名随版本变化，双映射兜底。"""
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise KeyError(f"列不存在（候选: {candidates}），实际列: {list(df.columns)}")


def compute_indicators(df: Any) -> dict[str, float | str]:
    """由日线历史计算 MA5/20/60、RSI14、MACD(12,26,9)，返回最新值。

    均线/MACD 使用前复权收盘价；RSI 为 Wilder 平滑。
    """
    close = _pick_column(df, "收盘", "close").astype(float)
    if len(close) < 60:
        raise RuntimeError(f"历史数据不足 60 个交易日（实际 {len(close)}），无法计算 MA60")

    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])

    # RSI14（Wilder）：平均涨幅 / 平均跌幅
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-12)  # 除零保护
    rsi = float(100 - 100 / (1 + rs.iloc[-1]))

    # MACD(12,26,9)：DIF = EMA12 - EMA26；DEA = DIF 的 EMA9；柱 = 2*(DIF-DEA)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    date_val = str(_pick_column(df, "日期", "date").iloc[-1])[:10]
    change_pct = float(_pick_column(df, "涨跌幅", "change_pct").iloc[-1])
    volume = int(_pick_column(df, "成交量", "volume").iloc[-1])
    return {
        "date": date_val,
        "price": float(close.iloc[-1]),
        "change_pct": change_pct,
        "volume": volume,
        "ma5": round(ma5, 4),
        "ma20": round(ma20, 4),
        "ma60": round(ma60, 4),
        "rsi": round(rsi, 2),
        "macd_dif": round(float(dif.iloc[-1]), 4),
        "macd_dea": round(float(dea.iloc[-1]), 4),
        "macd_hist": round(float(hist.iloc[-1]), 4),
    }


def fetch_name(symbol: str) -> str:
    """尽力而为获取股票简称（失败不影响主流程，返回空串）。"""
    try:
        import akshare as ak

        code = normalize_symbol(symbol)
        info = ak.stock_individual_info_em(symbol=code)
        row = info[info["item"] == "股票简称"]
        if not row.empty:
            return str(row["value"].iloc[0])
    except Exception:  # noqa: BLE001 —— 简称只是展示信息，失败可忽略
        pass
    return ""


# ------------------------------------------------------------------ 工具实现


class GetMarketDataTool(BaseTool):
    """行情查询工具（真实实现：AkShare 前复权日线 → 现价 + 技术指标）。"""

    name = "get_market_data"
    Args = GetMarketDataArgs
    Result = MarketDataResult

    def _execute(self, args: GetMarketDataArgs) -> MarketDataResult:
        if config_module.CONFIG.demo_tool_failure:
            # 演示 fail-open：模拟上游行情服务超时（与 mock 版行为一致）
            raise TimeoutError("【演示】模拟行情服务超时（fail-open 测试）")
        code: str = normalize_symbol(args.symbol)
        df: Any = fetch_history(code, config_module.CONFIG.market_history_days)
        ind: dict[str, float | str] = compute_indicators(df)
        return MarketDataResult(
            symbol=code,
            name=fetch_name(code),
            currency="CNY",
            note="真实数据（AkShare，前复权）",
            **ind,  # type: ignore[arg-type]
        )
