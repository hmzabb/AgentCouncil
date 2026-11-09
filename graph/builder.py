"""主链路 StateGraph 构建器（含 Send 并行、辩论循环、checkpointer）。"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
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

# 进程级图单例缓存：checkpointer（MemorySaver）必须跨请求共享，
# 否则每个请求新建空 MemorySaver，thread_id 断点续跑无法成立。
# key = serial_analysts（串行版仅评测对比用，与主链路分别缓存）。
_GRAPH_CACHE: dict[bool, CompiledStateGraph] = {}
_GRAPH_LOCK: threading.Lock = threading.Lock()


def build_council_graph(serial_analysts: bool = False) -> CompiledStateGraph:
    """构建（或复用）AgentCouncil 主链路图。

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

    单例语义：同一 serial_analysts 取值在本进程内只编译一次，共享同一个
    MemorySaver —— 同一 thread_id 的多次 invoke 才能读到上一次的 checkpoint
    并从中断点续跑（进程重启后内存状态仍会丢失，持久化需换 SqliteSaver）。

    - 所有节点均经过 fail_open_node 包装：任何单节点失败都不会中断整条图；
    - 节点内动态读取 config，图复用不影响 reload_config 的测试场景切换。
    """
    cached: CompiledStateGraph | None = _GRAPH_CACHE.get(serial_analysts)
    if cached is not None:
        return cached
    with _GRAPH_LOCK:
        cached = _GRAPH_CACHE.get(serial_analysts)
        if cached is not None:  # 双检：并发首编译只保留一个实例
            return cached
        cached = _compile_graph(serial_analysts)
        _GRAPH_CACHE[serial_analysts] = cached
        return cached


def _compile_graph(serial_analysts: bool) -> CompiledStateGraph:
    """实际编译图（仅供 build_council_graph 单例调用）。"""
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

    return builder.compile(checkpointer=_build_checkpointer())


def _build_checkpointer() -> MemorySaver:
    """构建内存 checkpointer，并显式登记本项目自定义类型的反序列化白名单。

    LangGraph 1.x 的 msgpack 反序列化默认只放行已知类型；未登记的自定义
    Pydantic 模型在 checkpoint 回读时会告警（「will be blocked in a future
    version」）。这里显式声明本项目状态模型，避免未来版本直接报错中断
    断点续跑，同时不放开任意类型（安全边界不扩大）。
    """
    allowed: list[tuple[str, str]] = [
        ("agents.schemas", name)
        for name in (
            "AgentStatus",
            "AnalystReport",
            "DebateSideOutput",
            "DebateRound",
            "DebateReport",
            "RiskItem",
            "RiskReviewReport",
            "FinalDecision",
        )
    ]
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))