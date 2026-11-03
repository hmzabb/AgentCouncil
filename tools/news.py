"""新闻检索工具真实实现（AkShare）：东财个股新闻 stock_news_em。

数据源（2026-11-02 实测，AkShare 1.20.22）：ak.stock_news_em(symbol)
返回个股近期新闻，列：关键词 / 新闻标题 / 新闻内容 / 发布时间 /
文章来源 / 新闻链接。本工具把 query 归一化为 6 位代码后取前
max_results 条。

范围说明：当前仅支持按个股代码查询新闻（基本面/情绪面分析师按标的
取新闻）；行业关键词检索需后续接入其他新闻源，收到非代码 query 时
fail-open 报错，不伪造新闻。

fail-open 约定：代码非法 / 接口异常 / 无新闻 → ToolResult(ok=False)
留痕，不伪造新闻数据。
"""

from __future__ import annotations

from typing import Any

import config as config_module
from pydantic import BaseModel, Field

from tools.base import BaseTool
from tools.market_data import normalize_symbol

# ------------------------------------------------------------------ schema


class GetNewsArgs(BaseModel):
    """新闻检索入参。"""

    query: str = Field(min_length=1, description="证券代码（当前仅支持个股代码查询）")
    max_results: int = Field(default=5, ge=1, le=20, description="最大返回条数")


class NewsItem(BaseModel):
    """单条新闻。"""

    title: str
    source: str
    url: str
    published_at: str
    snippet: str


class GetNewsResult(BaseModel):
    """新闻检索结果。"""

    items: list[NewsItem]


# ------------------------------------------------------------------ 工具实现


class GetNewsTool(BaseTool):
    """新闻检索工具（真实实现：AkShare 东财个股新闻）。"""

    name = "get_news"
    Args = GetNewsArgs
    Result = GetNewsResult

    def _execute(self, args: GetNewsArgs) -> GetNewsResult:
        if config_module.CONFIG.demo_tool_failure:
            # 演示 fail-open：模拟新闻服务超时（与 mock 版行为一致）
            raise TimeoutError("【演示】模拟新闻服务超时（fail-open 测试）")
        import akshare as ak  # 延迟导入：仅真实模式使用

        # 当前仅支持个股代码：非代码关键词（如「AI 算力」）如实报错
        code: str = normalize_symbol(args.query)
        df: Any = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            raise RuntimeError(f"AkShare 未返回 {code} 的新闻（代码错误 / 暂无新闻？）")

        items: list[NewsItem] = []
        for _, row in df.head(args.max_results).iterrows():
            title: str = str(row.get("新闻标题") or "").strip() or "（无标题）"
            content: str = str(row.get("新闻内容") or "").strip()
            items.append(
                NewsItem(
                    title=title,
                    source=str(row.get("文章来源") or "未知来源"),
                    url=str(row.get("新闻链接") or ""),
                    published_at=str(row.get("发布时间") or "")[:19],
                    snippet=(content or title)[:150],
                )
            )
        return GetNewsResult(items=items)
