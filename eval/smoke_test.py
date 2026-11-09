"""冒烟测试：验证 stub 链路真实跑通（所有断言基于真实运行结果，禁止模拟）。

场景：
  1. 正常链路：graph.invoke 走完 Send 并行 + 辩论循环全部节点，
     产出合法 FinalDecision（评级 + 置信度）；
  2. schema 与真实工具：Pydantic v2 约束真实生效；calculator / get_date 真实运行；
  3. fail-open：注入工具故障（DEMO_TOOL_FAILURE=true），链路仍完整走通并留痕，
     且失败传导至风控与评级；
  4. 自主性与风控否决：独立多空 Agent 各自产出、分析师工具自采白名单生效、
     风控 V1/V2 一票否决规则真实触发；
  5. 量化风控：波动率/最大回撤/ATR/量比/区间位置由构造序列真实计算，
     风控 V3 高波动否决、ATR 放大、量价背离规则真实触发（含对照组防误杀）。

运行方式（在项目根目录）：
    python -m eval.smoke_test
    python eval/smoke_test.py    # 同样支持
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# 保证无论以 `-m` 还是直接执行脚本方式运行，都能从项目根目录导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 冒烟测试固定走 mock 链路（确定性、不依赖外部服务）；
# 须在导入 config 前设置（config 导入时加载 .env，dotenv 不覆盖已有变量）。
# 真实数据链路（AkShare/Milvus）由 eval/real_data_test.py 覆盖。
os.environ["MOCK_MODE"] = "true"

import config as config_module  # noqa: E402 —— 动态读取配置（场景 3 会重载）
from agents.analyst import AnalystAgent  # noqa: E402 —— 工具自采白名单
from agents.debate import BearAgent, BullAgent  # noqa: E402 —— 独立多空 Agent
from agents.researcher import ResearcherAgent  # noqa: E402 —— 辩论编排
from agents.risk import RiskAgent  # noqa: E402 —— 一票否决规则
from agents.schemas import AgentStatus, AnalystReport, FinalDecision  # noqa: E402
from main import run_council  # noqa: E402
from tools.mock_tools import call_tool  # noqa: E402


def scenario_1_normal_run() -> None:
    """场景 1：正常链路必须走通（Send 并行 + 辩论循环 + 评级产出）。"""
    final_state: dict[str, Any] = run_council("600519.SH")
    decision = final_state.get("final_decision")
    assert isinstance(decision, FinalDecision), f"final_decision 类型错误: {type(decision)}"
    assert decision.agent == "manager_summarize"
    assert decision.status is AgentStatus.OK, f"决策状态异常: {decision.status}"
    assert decision.rating in {"买入", "增持", "中性", "减持", "卖出"}
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.summary and decision.reasoning, "摘要与理由不能为空"
    # Send 并行：三名分析师（基本面/技术面/情绪面）全部到位且成功
    reports = final_state.get("analyst_reports", [])
    roles = {r.role for r in reports}
    assert roles == {"基本面", "技术面", "情绪面"}, f"分析师角色不全: {roles}"
    assert all(r.status is AgentStatus.OK for r in reports)
    # 辩论循环：轮数必须等于配置上限（conditional_edges + max_rounds）
    history = final_state.get("debate_history", [])
    max_rounds = config_module.CONFIG.debate_max_rounds
    assert len(history) == max_rounds, f"辩论轮数 {len(history)} != {max_rounds}"
    assert final_state["debate_report"].total_rounds == max_rounds
    # messages 过程留痕（可追溯）
    assert final_state.get("messages"), "messages 过程留痕缺失"
    print(
        f"[1/5] 正常链路 PASS：评级={decision.rating}，置信度={decision.confidence}（真实计算），"
        f"分析师 {len(reports)} 位，辩论 {len(history)} 轮"
    )


def scenario_2_schema_and_real_tools() -> None:
    """场景 2：Pydantic v2 约束真实生效；真实实现的工具真实可用。"""
    # 越界值（confidence > 1.0）必须被拒绝
    try:
        FinalDecision(
            agent="x", status="ok", rating="中性", confidence=1.5,
            summary="s", reasoning="r", key_points=[],
        )
    except ValidationError:
        pass  # 符合预期
    else:
        raise AssertionError("confidence=1.5 未被 schema 拒绝，Pydantic 约束未生效")
    # 未声明字段（extra="forbid"）必须被拒绝
    try:
        FinalDecision(
            agent="x", status="ok", rating="中性", confidence=0.5,
            summary="s", reasoning="r", key_points=[], unknown_field=1,
        )
    except ValidationError:
        pass  # 符合预期
    else:
        raise AssertionError("未声明字段未被 schema 拒绝，extra=forbid 未生效")
    # 非法评级必须被 Literal 拒绝
    try:
        FinalDecision(
            agent="x", status="ok", rating="梭哈", confidence=0.5,
            summary="s", reasoning="r", key_points=[],
        )
    except ValidationError:
        pass  # 符合预期
    else:
        raise AssertionError("非法评级未被 Literal 拒绝")
    # calculator 为真实实现：真实计算 (1+2)*3 = 9.0
    calc = call_tool("calculator", expression="(1+2)*3")
    assert calc.ok and calc.data and calc.data["result"] == 9.0, f"calculator 结果异常: {calc}"
    # calculator 对非法表达式 fail-open（返回错误结果而非抛异常）
    bad = call_tool("calculator", expression="__import__('os').getcwd()")
    assert not bad.ok, "非法表达式应被 AST 白名单拒绝"
    # get_date 为真实实现：返回今天日期
    today = call_tool("get_date")
    assert today.ok and re.fullmatch(r"\d{4}-\d{2}-\d{2}", today.data["date"]), f"get_date 异常: {today}"
    print("[2/5] schema 与真实工具 PASS：约束真实拒绝；calculator/get_date 真实运行")


def scenario_3_fail_open() -> None:
    """场景 3：注入工具故障，链路必须仍完整走通（fail-open），并传导至评级。"""
    os.environ["DEMO_TOOL_FAILURE"] = "true"
    config_module.reload_config()
    try:
        final_state: dict[str, Any] = run_council("600519.SH")
    finally:
        os.environ.pop("DEMO_TOOL_FAILURE", None)
        config_module.reload_config()  # 恢复默认配置
    decision = final_state.get("final_decision")
    assert isinstance(decision, FinalDecision)
    assert decision.status is AgentStatus.OK, "工具失败不应导致整体链路失败"
    assert final_state.get("errors"), "工具失败应被留痕到 state.errors"
    assert decision.pipeline_errors, "工具失败应汇总到 final_decision.pipeline_errors"
    # 工具失败不拖垮分析师：分析师降级为「来源缺失 + 留痕」，而非整体失败
    reports = final_state["analyst_reports"]
    assert all(r.status is AgentStatus.OK for r in reports), "工具失败不应导致分析师整体失败"
    assert any(r.tool_errors for r in reports), "工具失败应被分析师留痕到 tool_errors"
    # 失败留痕传导：风控追加「信息不完整」→ 风险等级 高 → 评级降为 减持
    assert decision.rating == "减持", f"评级应降为减持，实际 {decision.rating}"
    assert any("信息不完整" in hint for hint in decision.risk_hints), "风险提示应包含信息不完整"
    # 辩论循环在故障下仍完整走完
    assert len(final_state.get("debate_history", [])) == config_module.CONFIG.debate_max_rounds
    print(
        f"[3/5] fail-open PASS：工具故意失败后链路仍走通，"
        f"留痕 {len(final_state['errors'])} 条错误，评级降为 {decision.rating}"
    )


def scenario_4_autonomy_and_veto() -> None:
    """场景 4：自主性（独立多空 Agent / 工具自采）与风控一票否决规则。"""
    # ---------- 4.1 独立多空 Agent：两个独立调用，身份与产出互不相同 ----------
    bull = BullAgent(round_no=1).run("600519.SH", "（本轮无检索证据）")
    bear = BearAgent(round_no=1).run("600519.SH", "（本轮无检索证据）", bull_argument=bull.argument)
    assert bull.agent == "bull_researcher" and bear.agent == "bear_researcher"
    assert bull.argument and bear.argument, "多空双方论点不能为空"
    assert bull.argument != bear.argument, "多空论点不应相同（否则是伪辩论）"
    # 编排层：一轮辩论的产物同时记录双方论点与要点字段
    round_result = ResearcherAgent(round_no=1).run({"query": "600519.SH"})
    assert round_result.round_no == 1
    assert round_result.bull_argument and round_result.bear_argument
    assert isinstance(round_result.bull_points, list) and isinstance(round_result.bear_points, list)

    # ---------- 4.2 工具自采：默认工具集来自角色白名单，可裁剪到单一工具 ----------
    tech = AnalystAgent(role="技术面", confidence=0.73)
    tools, _reason, plan_error = tech.plan_tools("600519.SH")
    assert tools == ["get_market_data"], f"技术面默认工具集异常: {tools}"
    assert plan_error is None, f"mock 模式不应有规划错误: {plan_error}"
    ctx, _errs, _srcs = tech.collect_context("600519.SH", tools=["get_market_data"])
    assert "market" in ctx, "指定工具后应只取行情数据"
    assert "news" not in ctx, "未指定的工具不应被调用"

    # ---------- 4.3 风控 V1：无有效分析师报告 → 一票否决 ----------
    risk_v1 = RiskAgent().run({"query": "600519.SH", "analyst_reports": []})
    assert risk_v1.blocked is True, "V1 规则应触发一票否决"
    assert risk_v1.block_reason and "V1" in risk_v1.block_reason
    assert risk_v1.risk_level == "高", "否决必须伴随高风险等级"

    # ---------- 4.4 风控 V2：空头排列 + MACD 柱为负 + RSI 破位 → 一票否决 ----------
    ok_report = AnalystReport(
        agent="analyst", status=AgentStatus.OK, role="技术面",
        summary="测试用报告", key_points=["测试论点"], confidence=0.7,
    )
    bearish_market = {
        "symbol": "600519.SH", "date": "2026-01-01", "price": 100.0, "change_pct": -3.0,
        "ma5": 95.0, "ma20": 105.0, "ma60": 115.0, "rsi": 28.0,
        "macd_dif": -2.0, "macd_dea": -1.0, "macd_hist": -1.0, "volume": 1000,
    }
    blocked_v2, reason_v2 = RiskAgent()._veto(
        reports=[ok_report], ok_count=1, market=bearish_market
    )
    assert blocked_v2 is True and reason_v2 and "V2" in reason_v2, f"V2 规则未触发: {reason_v2}"
    # 对照组：仅 RSI 低但均线多头排列 → 不应否决（避免误杀）
    mild_market = dict(bearish_market, ma5=115.0, ma20=110.0, ma60=100.0, macd_hist=0.5)
    blocked_mild, _ = RiskAgent()._veto(reports=[ok_report], ok_count=1, market=mild_market)
    assert blocked_mild is False, "多头排列时不应触发技术面否决"

    print(
        "[4/5] 自主性与风控否决 PASS：独立多空 Agent 生效；工具自采白名单生效；"
        f"V1 触发（{risk_v1.block_reason[:24]}…）、V2 触发且对照组未误杀"
    )


def scenario_5_quant_risk_rules() -> None:
    """场景 5：量化风险指标与 V3 / ATR / 量价背离规则（确定性、零网络依赖）。

    指标本身用构造的日线序列真实计算，不依赖 AkShare（真实数据链路由
    eval/real_data_test.py 覆盖）。
    """
    # ---------- 5.1 指标计算：构造 80 日序列，逐项校验数值语义 ----------
    from tools.market_data import compute_risk_metrics

    n = 80
    close = [100.0 + i for i in range(n)]  # 单调上涨：回撤应为 0
    high = [v + 1.0 for v in close]
    low = [v - 1.0 for v in close]
    volume = [1000.0] * n
    metrics = compute_risk_metrics(close, high, low, volume)
    assert metrics["max_drawdown_60d_pct"] == 0.0, "单调上涨序列最大回撤应为 0"
    assert metrics["position_60d"] > 0.95, "单调上涨序列应位于区间高位"
    assert metrics["volume_ratio_5d"] == 1.0, "等量序列量比应为 1"
    assert metrics["atr14"] > 0, "ATR 应为正"
    # 极端波动序列：冲高后腰斩并在高低位来回震荡 → 回撤与大波动必须被量化出来
    spike = [100.0] * 30 + [200.0] * 10 + [100.0, 200.0] * 20
    crash = compute_risk_metrics(spike, [v + 2 for v in spike], [v - 2 for v in spike], volume)
    assert crash["max_drawdown_60d_pct"] <= -50.0, f"腰斩序列回撤应 ≤ -50%，实际 {crash}"
    assert crash["volatility_20d_pct"] > 0.0
    # 输入缺失时 fail-open：返回中性默认值而非报错
    empty = compute_risk_metrics([])
    assert empty["position_60d"] == 0.5 and empty["volume_ratio_5d"] == 1.0

    # ---------- 5.2 风控 V3：高波动 + 大回撤 → 一票否决 ----------
    ok_report = AnalystReport(
        agent="analyst", status=AgentStatus.OK, role="技术面",
        summary="测试用报告", key_points=["测试论点"], confidence=0.7,
    )
    wild_market = {
        "symbol": "600519.SH", "date": "2026-01-01", "price": 100.0, "change_pct": 1.0,
        "ma5": 105.0, "ma20": 100.0, "ma60": 95.0, "rsi": 55.0,
        "macd_dif": 1.0, "macd_dea": 0.5, "macd_hist": 1.0, "volume": 1000,
        "volatility_20d_pct": 85.0, "max_drawdown_60d_pct": -40.0,
    }
    blocked_v3, reason_v3 = RiskAgent()._veto(
        reports=[ok_report], ok_count=1, market=wild_market
    )
    assert blocked_v3 is True and reason_v3 and "V3" in reason_v3, f"V3 规则未触发: {reason_v3}"
    # 对照组：只有波动率高、回撤小 → 不应否决（避免误杀）
    mild_vol = dict(wild_market, max_drawdown_60d_pct=-5.0)
    blocked_mild, _ = RiskAgent()._veto(reports=[ok_report], ok_count=1, market=mild_vol)
    assert blocked_mild is False, "回撤未超阈值时不应触发 V3 否决"

    # ---------- 5.3 ATR 放大与量价背离：走真实 execute 路径，用配置阈值触发 ----------
    # mock 行情为中性值（ATR 占价 2.14%、区间位置 0.55、量比 1.06），
    # 默认阈值下不应产生这两条风险项（避免误报）。
    state = {"query": "600519.SH", "analyst_reports": [ok_report]}
    baseline = RiskAgent().run(state)
    assert not any(r.title == "ATR 放大" for r in baseline.risks), "中性行情不应误报 ATR 放大"
    assert not any(r.title == "量价背离" for r in baseline.risks), "中性行情不应误报量价背离"

    os.environ["RISK_ATR_SPIKE_PCT"] = "1.0"  # 阈值低于 mock 的 2.14%
    config_module.reload_config()
    try:
        risk_atr = RiskAgent().run(state)
    finally:
        os.environ.pop("RISK_ATR_SPIKE_PCT", None)
        config_module.reload_config()
    assert any(r.title == "ATR 放大" for r in risk_atr.risks), "ATR 放大风险项未追加"
    assert risk_atr.risk_level == "高", "ATR 放大应把整体风险等级抬到高"

    os.environ["RISK_DIVERGENCE_POSITION"] = "0.4"  # mock 区间位置 0.55 达标
    os.environ["RISK_DIVERGENCE_RATIO"] = "1.1"     # mock 量比 1.06 视为缩量
    config_module.reload_config()
    try:
        risk_div = RiskAgent().run(state)
    finally:
        os.environ.pop("RISK_DIVERGENCE_POSITION", None)
        os.environ.pop("RISK_DIVERGENCE_RATIO", None)
        config_module.reload_config()
    assert any(r.title == "量价背离" for r in risk_div.risks), "量价背离风险项未追加"
    # 规则对照：区间高位但放量（量比 1.4 > 默认阈值 0.8）→ 不应判定背离
    assert (
        RiskAgent()._quant_risk_items(
            {"atr_pct": 2.14, "position_60d": 0.92, "volume_ratio_5d": 1.4}
        )
        == []
    ), "区间高位但放量时不应判定量价背离"

    print(
        "[5/5] 量化风控 PASS：指标真实计算（回撤/区间位置/量比/ATR）；"
        f"V3 触发（{reason_v3[:24]}…）且对照组未误杀；ATR 放大与量价背离规则生效"
    )


def main() -> int:
    """依次执行五个场景；任一失败即非零退出。"""
    print("=== AgentCouncil stub 冒烟测试（全部基于真实运行） ===")
    scenario_1_normal_run()
    scenario_2_schema_and_real_tools()
    scenario_3_fail_open()
    scenario_4_autonomy_and_veto()
    scenario_5_quant_risk_rules()
    print("=== 全部场景 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())