"""冒烟测试：验证 stub 链路真实跑通（所有断言基于真实运行结果，禁止模拟）。

场景：
  1. 正常链路：graph.invoke 走完 Send 并行 + 辩论循环全部节点，
     产出合法 FinalDecision（评级 + 置信度）；
  2. schema 与真实工具：Pydantic v2 约束真实生效；calculator / get_date 真实运行；
  3. fail-open：注入工具故障（DEMO_TOOL_FAILURE=true），链路仍完整走通并留痕，
     且失败传导至风控与评级。

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
from agents.schemas import AgentStatus, FinalDecision  # noqa: E402
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
        f"[1/3] 正常链路 PASS：评级={decision.rating}，置信度={decision.confidence}（真实计算），"
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
    print("[2/3] schema 与真实工具 PASS：约束真实拒绝；calculator/get_date 真实运行")


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
        f"[3/3] fail-open PASS：工具故意失败后链路仍走通，"
        f"留痕 {len(final_state['errors'])} 条错误，评级降为 {decision.rating}"
    )


def main() -> int:
    """依次执行三个场景；任一失败即非零退出。"""
    print("=== AgentCouncil stub 冒烟测试（全部基于真实运行） ===")
    scenario_1_normal_run()
    scenario_2_schema_and_real_tools()
    scenario_3_fail_open()
    print("=== 全部场景 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
