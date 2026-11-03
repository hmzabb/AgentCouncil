"""主链路 StateGraph 构建器（含 Send 并行、辩论循环、checkpointer）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    route_after_debate,
    run_analyst,
    run_analyst_team,
    run_analyst_team_serial,
    run_manager_summarize,
    run_researcher_debate,
    run_risk_review,
)
from graph.state import CouncilState

if TYPE_CHECKING:  # pragma: no cover —— 仅类型标注使用，避免运行时依赖具体导出路径
    from langgraph.graph.state import CompiledStateGraph


def build_council_graph(serial_analysts: bool = False) -> "CompiledStateGraph":
    """构建 AgentCouncil 主链路图。

    链路：
        START → analyst_team（分发器，Send 图级并行）
              → analyst ×3（基本面/技术面/情绪面）—— 全部完成后汇聚
              → researcher_debate（conditional_edges 循环 N 轮，max_rounds 防死循环）
              → risk_review → manager_summarize → END

    参数：
        serial_analysts: True 时分析师团队改为单节点 for 循环串行执行
                         （run_analyst_team_serial），仅用于「并行 vs 串行」
                         耗时对比评测（eval/parallelism_test.py）；
                         默认 False 走 Send 图级并行（主链路）。

    - 所有节点均经过 fail_open_node 包装：任何单节点失败都不会中断整条图；
    - checkpointer 使用内存版（MemorySaver），线程级断点续跑；持久化后续换 SqliteSaver。
    """
    builder = StateGraph(CouncilState)
    if serial_analysts:
        builder.add_node("analyst_team", run_analyst_team_serial)   # 串行版（评测对比用）
    else:
        builder.add_node("analyst_team", run_analyst_team)          # 分发器：返回 Command(goto=[Send...])
        builder.add_node("analyst", run_analyst)                    # 并行 worker（每分支独立检查点）
    builder.add_node("researcher_debate", run_researcher_debate)
    builder.add_node("risk_review", run_risk_review)
    builder.add_node("manager_summarize", run_manager_summarize)

    builder.add_edge(START, "analyst_team")
    if serial_analysts:
        # 串行版：分析师节点直接进入辩论
        builder.add_edge("analyst_team", "researcher_debate")
    else:
        # 所有 Send 分支完成后自动汇聚进入辩论（LangGraph fan-in 语义）
        builder.add_edge("analyst", "researcher_debate")
    # 辩论循环：未达 max_rounds 回到 researcher_debate，否则进入风控
    builder.add_conditional_edges(
        "researcher_debate",
        route_after_debate,
        {"researcher_debate": "researcher_debate", "risk_review": "risk_review"},
    )
    builder.add_edge("risk_review", "manager_summarize")
    builder.add_edge("manager_summarize", END)

    return builder.compile(checkpointer=MemorySaver())
