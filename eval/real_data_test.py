"""真实数据链路测试：Milvus RAG + AkShare 行情/财务/新闻 + 三角色分析师。

与 smoke_test（固定 MOCK_MODE=true 的 mock 链路）互补：本文件固定
MOCK_MODE=false，所有断言基于真实外部数据（docker Milvus 本地库 /
AkShare 东方财富、同花顺接口），外部服务未就绪时明确失败（不伪造通过）。

前置条件：
    docker start milvus-standalone
    python data/ingest.py        # 下载样本研报并建库（首次含 bge 权重下载）

运行（项目根目录）：
    python -m eval.real_data_test
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

# 保证无论以 `-m` 还是直接执行脚本方式运行，都能从项目根目录导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 固定真实模式：须在导入 config 前设置（config 导入时加载 .env，dotenv 不覆盖已有变量）
os.environ["MOCK_MODE"] = "false"

import config as config_module  # noqa: E402
from agents.analyst import AnalystAgent  # noqa: E402
from agents.schemas import AgentStatus  # noqa: E402
from graph.state import CouncilState  # noqa: E402
from main import run_council  # noqa: E402
from tools.mock_tools import call_tool  # noqa: E402


def scenario_1_rag_milvus() -> None:
    """场景 1：research_rag 真实检索 Milvus（chunk 带元数据、无【stub】假数据）。"""
    rag = call_tool("research_rag", query="白酒 行业 竞争格局 盈利", top_k=3)
    assert rag.ok, (
        f"Milvus 检索失败（请确认 docker start milvus-standalone 且已运行 data/ingest.py）: {rag.error}"
    )
    chunks: list[dict[str, Any]] = rag.data["chunks"]
    assert chunks, "检索结果为空"
    for c in chunks:
        assert "【stub】" not in c["chunk_text"], f"检索文本不得来自 mock: {c['doc_name']}"
        assert c["metadata"] and c["metadata"].get("source"), f"chunk 缺少元数据: {c['doc_name']}"
    print(
        f"[1/6] Milvus RAG PASS：检索到 {len(chunks)} 条真实研报 chunk，"
        f"top1 相似度 {chunks[0]['score']}，来源 {chunks[0]['metadata'].get('source')}"
    )


def scenario_2_market_akshare() -> None:
    """场景 2：get_market_data 真实 AkShare 行情；指标与独立重算交叉验证。"""
    m = call_tool("get_market_data", symbol="600519.SH")
    assert m.ok, f"行情获取失败（网络/接口问题）: {m.error}"
    d: dict[str, Any] = m.data
    assert 0.0 <= d["rsi"] <= 100.0, f"RSI 越界: {d['rsi']}"
    assert d["price"] > 0 and d["ma5"] > 0, f"价格/均线异常: {d}"
    # 独立重算：重新拉取历史并重算指标，交叉验证工具指标计算无误
    from tools.market_data import compute_indicators, fetch_history

    ind: dict[str, Any] = compute_indicators(
        fetch_history("600519", config_module.CONFIG.market_history_days)
    )
    tolerance: dict[str, float] = {"rsi": 0.011}  # rsi 保留 2 位小数，其余 4 位
    for key in ("ma5", "ma20", "ma60", "rsi", "macd_dif", "macd_dea", "macd_hist"):
        assert abs(d[key] - ind[key]) <= tolerance.get(key, 1e-3), (
            f"{key} 与独立重算不一致: {d[key]} vs {ind[key]}"
        )
    print(
        f"[2/6] AkShare 行情 PASS：{d['name']}（{d['symbol']}）{d['date']} 收盘 {d['price']}，"
        f"MA5={d['ma5']}，RSI={d['rsi']}，MACD柱={d['macd_hist']}（独立重算一致）"
    )


def scenario_3_tech_analyst() -> None:
    """场景 3：技术分析师单 Agent 真实跑通：ticker → AkShare → 结构化报告。

    报告路径随环境自动切换：配置了 DeepSeek key 走 LLM 生成；
    否则降级为三信号投票规则模板（同样基于真实行情数据，并留痕 LLM 失败）。
    """
    state: CouncilState = {"query": "600519.SH"}
    report = AnalystAgent(role="技术面", confidence=0.73).run(state)
    assert report.status is AgentStatus.OK, f"技术分析师失败: {report.error}"
    assert report.key_points and all(
        "【stub】" not in kp for kp in report.key_points
    ), f"技术面论点不得来自假数据: {report.key_points}"
    assert 0.0 < report.confidence <= 1.0, f"置信度异常: {report.confidence}"
    if "DeepSeek" in (report.note or ""):
        assert not report.tool_errors, f"LLM 路径下工具调用应成功: {report.tool_errors}"
        mode = "LLM（DeepSeek）"
    else:
        # LLM 不可用时的降级路径：规则模板（三信号投票），失败须如实留痕
        assert any("LLM" in e for e in report.tool_errors), (
            f"降级路径应留痕 LLM 失败: {report.tool_errors}"
        )
        assert any("MA5" in kp for kp in report.key_points), "降级模板论点应包含均线指标"
        mode = "规则模板降级"
    print(f"[3/6] 技术分析师单 Agent PASS（{mode}）：置信度 {report.confidence}")
    print(f"  摘要：{report.summary}")
    for kp in report.key_points:
        print(f"    - {kp}")
    print(f"  来源：{report.sources}")


def scenario_4_full_graph() -> None:
    """场景 4：全链路真实模式跑通（三角色分析师全部引用真实数据来源）。"""
    final_state: dict[str, Any] = run_council("600519.SH")
    decision = final_state.get("final_decision")
    assert decision is not None and decision.status is AgentStatus.OK
    reports: list[Any] = final_state["analyst_reports"]
    roles = {r.role for r in reports}
    assert roles == {"基本面", "技术面", "情绪面"}, f"分析师角色不全: {roles}"
    tech = next(r for r in reports if r.role == "技术面")
    assert all("【stub】" not in kp for kp in tech.key_points), "技术面论点不得用假数据"
    fund = next(r for r in reports if r.role == "基本面")
    assert any("财务指标" in s for s in fund.sources), (
        f"基本面未引用财务指标来源: {fund.sources}"
    )
    senti = next(r for r in reports if r.role == "情绪面")
    assert any("新闻" in s for s in senti.sources), f"情绪面未引用新闻来源: {senti.sources}"
    assert any("Milvus" in s for s in senti.sources), f"情绪面未引用 Milvus 检索来源: {senti.sources}"
    # 全部数据工具已接入：不应再有「尚未接入」类留痕
    assert not any("尚未接入" in e for e in decision.pipeline_errors), (
        f"所有数据工具已接入，不应再留痕未接入: {decision.pipeline_errors}"
    )
    print(
        f"[4/6] 全链路真实模式 PASS：评级={decision.rating}，置信度={decision.confidence:.2f}，"
        f"三名分析师引用真实行情/财务/新闻/RAG 来源，无未接入工具留痕"
    )


def scenario_5_financials_news() -> None:
    """场景 5：get_financials / get_news 真实数据 + 独立重取交叉验证。"""
    fin = call_tool("get_financials", symbol="600519")
    assert fin.ok, f"财务指标获取失败: {fin.error}"
    m: dict[str, Any] = fin.data["metrics"]
    assert m["pe"] > 0 and m["pb"] > 0, f"PE/PB 异常: {m}"
    assert -100.0 < m["roe"] < 100.0, f"ROE 异常: {m['roe']}"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m["report_period"]), f"报告期异常: {m['report_period']}"
    # 独立重取交叉验证：同花顺摘要最新报告期与工具输出一致
    import akshare as ak

    ths = ak.stock_financial_abstract_ths(symbol="600519", indicator="按报告期")
    latest = ths.iloc[-1]
    assert m["report_period"] == str(latest["报告期"])[:10], (
        f"报告期与独立重取不一致: {m['report_period']} vs {latest['报告期']}"
    )
    assert abs(m["roe"] - float(str(latest["净资产收益率"]).replace("%", ""))) < 1e-3, (
        f"ROE 与独立重取不一致: {m['roe']} vs {latest['净资产收益率']}"
    )
    news = call_tool("get_news", query="600519", max_results=5)
    assert news.ok, f"新闻获取失败: {news.error}"
    items: list[dict[str, Any]] = news.data["items"]
    assert items, "新闻结果为空"
    for it in items:
        assert it["title"] and "【stub】" not in it["title"], f"新闻标题异常: {it}"
        assert it["url"].startswith("http"), f"新闻链接异常: {it['url']}"
    print(
        f"[5/6] 财务 + 新闻 PASS：{m['symbol']} 报告期 {m['report_period']}，"
        f"PE={m['pe']}，PB={m['pb']}，ROE={m['roe']}%，"
        f"营收增速 {m['revenue_growth']}%，净利增速 {m['net_profit_growth']}%（独立重取一致）；"
        f"新闻 {len(items)} 条（最新：{items[0]['title'][:40]}…）"
    )


def scenario_6_fail_open_real_tools() -> None:
    """场景 6：真实工具 fail-open：非法入参 / 服务不可用 → ok=False，不抛异常不伪造。"""
    # 1) 非法代码：三个数据工具均返回 ok=False（normalize_symbol 拒绝非 6 位代码）
    for name, kwargs in [
        ("get_market_data", {"symbol": "NOT-A-CODE"}),
        ("get_financials", {"symbol": "ABC"}),
        ("get_news", {"query": "AI 算力"}),
    ]:
        result = call_tool(name, **kwargs)
        assert not result.ok, f"{name} 对非法入参应返回 ok=False"
        assert result.error, f"{name} 失败应留痕 error"
    # 2) Milvus 集合不存在：research_rag 返回 ok=False（不伪造检索结果）
    os.environ["MILVUS_COLLECTION"] = "no_such_collection_xyz"
    config_module.reload_config()
    try:
        rag = call_tool("research_rag", query="白酒 竞争格局", top_k=2)
    finally:
        os.environ.pop("MILVUS_COLLECTION", None)
        config_module.reload_config()
    assert not rag.ok and "不存在" in (rag.error or ""), f"RAG 集合缺失应留痕: {rag.error}"
    # 3) DEMO_TOOL_FAILURE：真实模式下行情工具同样 fail-open
    os.environ["DEMO_TOOL_FAILURE"] = "true"
    config_module.reload_config()
    try:
        m = call_tool("get_market_data", symbol="600519")
    finally:
        os.environ.pop("DEMO_TOOL_FAILURE", None)
        config_module.reload_config()
    assert not m.ok and "超时" in (m.error or ""), f"演示故障应留痕超时: {m.error}"
    print(
        "[6/6] 真实工具 fail-open PASS：非法代码 / Milvus 集合缺失 / 演示超时 "
        "均返回 ok=False 且留痕，链路不崩溃"
    )


def main() -> int:
    """依次执行六个场景；任一失败即非零退出。"""
    print("=== AgentCouncil 真实数据链路测试（Milvus + AkShare，全部基于真实运行） ===")
    scenario_1_rag_milvus()
    scenario_2_market_akshare()
    scenario_3_tech_analyst()
    scenario_4_full_graph()
    scenario_5_financials_news()
    scenario_6_fail_open_real_tools()
    print("=== 全部场景 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
