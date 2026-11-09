"""Mock 工具集：stub 模式返回假数据，但 schema 与真实工具完全一致。

工具清单（对应 CLAUDE.md 架构约定）：
- get_market_data   mock 假行情；MOCK_MODE=false 时路由到 tools.market_data（AkShare 真实实现）
- research_rag      mock 假检索；MOCK_MODE=false 时路由到 tools.rag（Milvus + bge 真实实现）
- get_financials    mock 假财务；MOCK_MODE=false 时路由到 tools.financials（AkShare 真实实现）
- get_news          mock 假新闻；MOCK_MODE=false 时路由到 tools.news（AkShare 真实实现）
- calculator        四则运算计算器（真实实现，AST 白名单保证安全）
- get_date          当前日期（真实实现）

schema 来源：全部复用真实实现模块的 schema（market_data / rag /
financials / news），保证 mock/真实两种模式字段对齐。
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

import config as config_module  # 动态读取配置（支持测试期 reload_config 切换场景）
from tools.base import BaseTool, ToolResult
from tools.cache import TTLCache, make_cache_key
from tools.financials import (  # 复用真实 schema
    FinancialMetrics,
    GetFinancialsArgs,
    GetFinancialsResult,
)
from tools.market_data import GetMarketDataArgs, MarketDataResult  # 复用真实 schema
from tools.news import GetNewsArgs, GetNewsResult, NewsItem  # 复用真实 schema
from tools.rag import RagChunk, ResearchRagArgs, ResearchRagResult  # 复用真实 schema

# ---------------------------------------------------------------- 行情（mock）

class GetMarketDataTool(BaseTool):
    """行情查询工具（mock：返回固定假行情；schema 与真实 AkShare 实现一致）。"""

    name = "get_market_data"
    Args = GetMarketDataArgs
    Result = MarketDataResult

    def _execute(self, args: GetMarketDataArgs) -> MarketDataResult:
        if config_module.CONFIG.demo_tool_failure:
            # 演示 fail-open：模拟上游行情服务超时
            raise TimeoutError("【演示】模拟行情服务超时（fail-open 测试）")
        return MarketDataResult(
            symbol=args.symbol,
            name="【stub】示例股票",
            date=datetime.now().strftime("%Y-%m-%d"),
            price=100.5,
            change_pct=1.23,
            volume=12_345_678,
            currency="CNY",
            ma5=101.2,
            ma20=99.8,
            ma60=98.5,
            rsi=58.3,
            macd_dif=1.21,
            macd_dea=1.05,
            macd_hist=0.32,
            volatility_20d_pct=25.4,
            max_drawdown_60d_pct=-7.8,
            atr14=2.15,
            atr_pct=2.14,
            volume_ratio_5d=1.06,
            position_60d=0.55,
            note="【stub】假数据",
        )


# ---------------------------------------------------------------- RAG 检索（mock）

class ResearchRagTool(BaseTool):
    """研报检索工具（mock：确定性假检索结果；schema 与真实 Milvus 实现一致）。"""

    name = "research_rag"
    Args = ResearchRagArgs
    Result = ResearchRagResult

    def _execute(self, args: ResearchRagArgs) -> ResearchRagResult:
        chunks: list[RagChunk] = [
            RagChunk(
                chunk_text=f"【stub】与「{args.query}」相关的第 {i + 1} 段研报占位文本（Milvus 检索占位）。",
                doc_name=f"【stub】研报_{i + 1}.pdf",
                score=round(0.9 - 0.1 * i, 2),
                metadata=None,  # mock 无真实元数据
            )
            for i in range(args.top_k)
        ]
        return ResearchRagResult(chunks=chunks)


# ---------------------------------------------------------------- 财务指标（mock）

class GetFinancialsTool(BaseTool):
    """财务指标工具（mock：返回固定假财务数据，仅 MOCK_MODE=true 使用）。

    真实模式（MOCK_MODE=false）由 call_tool 路由到 tools.financials
    （AkShare 真实实现）；本类保留真实模式守卫，防止被绕过路由直接调用
    时把假财务数据伪装成真实数据（CLAUDE.md 约束 4）。
    """

    name = "get_financials"
    Args = GetFinancialsArgs
    Result = GetFinancialsResult

    def _execute(self, args: GetFinancialsArgs) -> GetFinancialsResult:
        if not config_module.CONFIG.mock_mode:
            raise NotImplementedError(
                "get_financials 真实模式应由 tools.financials（AkShare）处理，拒绝返回假数据"
            )
        return GetFinancialsResult(
            metrics=FinancialMetrics(
                symbol=args.symbol,
                pe=25.4,
                pb=3.1,
                roe=15.6,
                revenue_growth=12.3,
                net_profit_growth=18.9,
                report_period="2026Q3【stub】",
            )
        )


# ---------------------------------------------------------------- 新闻（mock）

class GetNewsTool(BaseTool):
    """新闻检索工具（mock：按模板生成确定性假新闻，仅 MOCK_MODE=true 使用）。

    真实模式（MOCK_MODE=false）由 call_tool 路由到 tools.news
    （AkShare 真实实现）；本类保留真实模式守卫，防止被绕过路由直接调用
    时伪造新闻数据。
    """

    name = "get_news"
    Args = GetNewsArgs
    Result = GetNewsResult

    def _execute(self, args: GetNewsArgs) -> GetNewsResult:
        if not config_module.CONFIG.mock_mode:
            raise NotImplementedError(
                "get_news 真实模式应由 tools.news（AkShare）处理，拒绝返回假数据"
            )
        sources: list[str] = ["财新网", "界面新闻", "晚点 LatePost", "第一财经", "Reuters"]
        items: list[NewsItem] = [
            NewsItem(
                title=f"【stub】{args.query}：第 {i + 1} 篇模拟报道",
                source=sources[i % len(sources)],
                url=f"https://example.com/stub/{i + 1}",  # 占位链接，非真实来源
                published_at="2026-11-01",  # 假日期，仅演示
                snippet=f"【stub】关于「{args.query}」的第 {i + 1} 条假新闻摘要，仅用于链路联调。",
            )
            for i in range(args.max_results)
        ]
        return GetNewsResult(items=items)


# ---------------------------------------------------------------- 计算器（真实实现）

# AST 白名单：仅允许常量与四则运算节点，杜绝任意代码执行
_ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)


class CalculatorArgs(BaseModel):
    """计算器入参。"""

    expression: str = Field(min_length=1, max_length=200, description="四则运算表达式，如 (1+2)*3/4")


class CalculatorResult(BaseModel):
    """计算结果。"""

    expression: str
    result: float


class CalculatorTool(BaseTool):
    """四则运算计算器（真实实现，非 stub；AST 白名单保证安全）。"""

    name = "calculator"
    Args = CalculatorArgs
    Result = CalculatorResult

    def _execute(self, args: CalculatorArgs) -> CalculatorResult:
        tree = ast.parse(args.expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_AST_NODES):
                raise ValueError(f"不支持的表达式节点: {type(node).__name__}")
        # 白名单校验通过后 eval 安全：无内置函数、无属性访问、无变量
        value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
        if not isinstance(value, (int, float)):
            raise ValueError("表达式结果必须是数字")
        return CalculatorResult(expression=args.expression, result=float(value))


# ---------------------------------------------------------------- 日期（真实实现）

class GetDateArgs(BaseModel):
    """无入参。"""

    model_config = ConfigDict(extra="forbid")


class GetDateResult(BaseModel):
    """当前日期。"""

    date: str
    weekday: str


class GetDateTool(BaseTool):
    """当前日期查询（真实实现，非 stub）。"""

    name = "get_date"
    Args = GetDateArgs
    Result = GetDateResult

    def _execute(self, args: GetDateArgs) -> GetDateResult:
        now = datetime.now()
        weekdays: list[str] = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return GetDateResult(date=now.strftime("%Y-%m-%d"), weekday=weekdays[now.weekday()])


# ---------------------------------------------------------------- 注册表与路由

MOCK_TOOLS: dict[str, BaseTool] = {
    GetMarketDataTool.name: GetMarketDataTool(),
    ResearchRagTool.name: ResearchRagTool(),
    GetFinancialsTool.name: GetFinancialsTool(),
    GetNewsTool.name: GetNewsTool(),
    CalculatorTool.name: CalculatorTool(),
    GetDateTool.name: GetDateTool(),
}

# 真实模式下从 mock 路由到真实实现的工厂（延迟导入：不使用时零开销）
_REAL_TOOL_FACTORIES: dict[str, Callable[[], BaseTool]] = {}


def _register_real(name: str, factory: Callable[[], BaseTool]) -> None:
    """注册真实工具工厂（模块导入时执行）。"""
    _REAL_TOOL_FACTORIES[name] = factory


def _make_real_get_market_data() -> BaseTool:
    from tools.market_data import GetMarketDataTool

    return GetMarketDataTool()


def _make_real_research_rag() -> BaseTool:
    from tools.rag import ResearchRagTool

    return ResearchRagTool()


def _make_real_get_financials() -> BaseTool:
    from tools.financials import GetFinancialsTool

    return GetFinancialsTool()


def _make_real_get_news() -> BaseTool:
    from tools.news import GetNewsTool

    return GetNewsTool()


_register_real("get_market_data", _make_real_get_market_data)
_register_real("research_rag", _make_real_research_rag)
_register_real("get_financials", _make_real_get_financials)
_register_real("get_news", _make_real_get_news)

# 各工具独立的 TTL 缓存（TTL 每次调用时按 config 动态刷新，支持 reload_config）
_TOOL_CACHES: dict[str, TTLCache] = {}


def _cache_for(name: str) -> TTLCache:
    """获取（或创建）工具缓存，并按当前配置刷新 TTL。

    RAG 检索结果相对稳定（研报不常变），TTL 更长；行情/财务/新闻更新更快。
    """
    cache: TTLCache | None = _TOOL_CACHES.get(name)
    if cache is None:
        cache = TTLCache(ttl_s=1)
        _TOOL_CACHES[name] = cache
    cache.ttl_s = (
        config_module.CONFIG.rag_cache_ttl_s
        if name == "research_rag"
        else config_module.CONFIG.tool_cache_ttl_s
    )
    return cache


def call_tool(name: str, **kwargs: object) -> ToolResult:
    """统一工具调用入口（mock / 真实双模式路由）。

    - MOCK_MODE=false 且工具已有真实实现 → 调用真实实现；真实实现初始化
      失败（依赖未装 / 服务未启动）同样 fail-open 返回 ok=False，绝不回退
      到假数据；
    - 真实模式下成功结果按 TTL 缓存（同一标的短时间重复查询不再打外部接口）；
      失败结果不缓存，避免把瞬时故障固化为缓存；
    - 未知工具名同样 fail-open：返回 ok=False 的结果，而非抛 KeyError。
    """
    if not config_module.CONFIG.mock_mode and name in _REAL_TOOL_FACTORIES:
        cache: TTLCache = _cache_for(name)
        key: str = make_cache_key(name, dict(kwargs))
        hit, cached = cache.get(key)
        if hit:
            return cached.model_copy(deep=True)  # 防止下游修改污染缓存
        try:
            result: ToolResult = _REAL_TOOL_FACTORIES[name]().invoke(**kwargs)
        except Exception as exc:  # noqa: BLE001 —— fail-open：真实工具不可用时留痕
            return ToolResult(
                tool=name,
                ok=False,
                error=f"真实工具初始化失败: {type(exc).__name__}: {exc}",
            )
        if result.ok:
            cache.set(key, result)
        return result
    tool: BaseTool | None = MOCK_TOOLS.get(name)
    if tool is None:
        return ToolResult(tool=name, ok=False, error=f"未注册的工具: {name}")
    return tool.invoke(**kwargs)


def tool_cache_stats() -> dict[str, dict[str, int]]:
    """返回各工具缓存命中统计（供 /health 等可观测性端点使用）。"""
    return {name: cache.stats() for name, cache in _TOOL_CACHES.items()}