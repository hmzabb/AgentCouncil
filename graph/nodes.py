"""LangGraph 节点实现：图中每个节点对应一个执行阶段。

职责边界：
- 节点 = 编排与状态更新（读取 state → 驱动 Agent → 写回阶段产物）；
- Agent = 具体产出逻辑（agents/ 包内实现）；
- 所有节点经 fail_open_node 包装：节点级异常降级为合法产物 + 错误留痕，
  保证任何单点失败都不会中断整条图（CLAUDE.md 要求）。

图结构（见 graph/builder.py）：
    START → analyst_team（分发器，返回 Command(goto=[Send...])，图级并行）
          → analyst ×3（基本面/技术面/情绪面，并行 worker）
          → researcher_debate（多空辩论，conditional_edges 循环 N 轮，max_rounds 防死循环）
          → risk_review → manager_summarize → END
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langgraph.types import Command, Send

import config as config_module  # 动态读取配置（冒烟测试会通过 reload_config 重载）
from agents.analyst import ANALYST_ROLES, AnalystAgent
from agents.manager import ManagerAgent
from agents.researcher import ResearcherAgent
from agents.risk import RiskAgent
from agents.schemas import (
    AgentStatus,
    AnalystReport,
    DebateRound,
    FinalDecision,
    RiskItem,
    RiskReviewReport,
)
from graph.state import CouncilState

logger = logging.getLogger(__name__)

# 各角色分析师的确定性占位置信度（mock / LLM 降级模板用）。
# 真实模式下置信度由 LLM 输出覆盖（见 agents/analyst.py）
ROLE_CONFIDENCE: dict[str, float] = {"基本面": 0.68, "技术面": 0.73, "情绪面": 0.78}

# 节点返回类型：普通节点返回部分状态更新 dict；分发器节点返回 Command(goto=[Send...])
# （langgraph 1.2 不再接受裸 list[Send]，须用 Command.goto 承载图级并行分发）
NodeReturn = dict[str, Any] | Command  # type: ignore[misc] —— Command 无需参数化
NodeFunc = Callable[[CouncilState], NodeReturn]


def fail_open_node(fallback: Callable[[Exception, CouncilState], NodeReturn]):
    """节点级 fail-open 装饰器。

    LangGraph 中节点抛异常会终止整条 graph；本装饰器把节点异常转换为
    「降级状态更新 + errors 留痕」，保证单点失败不拖垮全链路。

    参数：
        fallback: 接收 (exception, state)，返回该节点的降级状态更新
                  （dict 或 Send 列表；dict 场景由装饰器统一追加 errors 项，
                  Send 列表场景请自行在 payload 中携带 errors）。
    """

    def decorator(fn: NodeFunc) -> NodeFunc:
        @functools.wraps(fn)  # 保留函数签名/注解，供 LangGraph 识别返回类型
        def wrapper(state: CouncilState) -> NodeReturn:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001 —— fail-open：节点失败不崩溃
                logger.warning("节点 [%s] 整体失败（fail-open 降级）：%s", fn.__name__, exc)
                update = fallback(exc, state)
                if isinstance(update, dict):
                    update.setdefault(
                        "errors",
                        [f"节点 {fn.__name__} 失败: {type(exc).__name__}: {exc}"],
                    )
                return update

        return wrapper

    return decorator


@fail_open_node(
    lambda exc, state: Command(
        # 分发器失败也 fail-open：错误作为 Command.update 写回 state，
        # goto 仍照常分发全部角色，保证链路不中断
        update={"errors": [f"节点 analyst_team 失败: {type(exc).__name__}: {exc}"]},
        goto=_build_analyst_sends(state.get("query", "")),
    )
)
def run_analyst_team(state: CouncilState) -> Command:
    """analyst_team 节点（分发器）：用 Send API 将分析师团队图级并行 fan-out。

    CLAUDE.md 架构要求：基本面/技术面/情绪面 三分析师并行 Send。
    与节点内线程池并行相比，图级并行的优势：各分支独立检查点、
    可单独重试、天然支持断点续跑。

    注意：Send 分支的输入 state 只含 payload 字段，因此必须把分支内
    需要的全部上下文（query、assigned_role）显式放入每个 Send 的 payload。
    """
    return Command(goto=_build_analyst_sends(state["query"]))


def _build_analyst_sends(query: str) -> list[Send]:
    """构造分析师团队的 Send 分发列表（每个角色一个并行分支）。"""
    return [
        Send("analyst", {"assigned_role": role, "query": query})
        for role in ANALYST_ROLES
    ]


@fail_open_node(
    lambda exc, state: {
        "analyst_reports": [
            AnalystReport(
                agent="analyst",
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                role=state.get("assigned_role", "未知"),
                summary="",
                key_points=[],
                confidence=0.0,
            )
        ]
    }
)
def run_analyst(state: CouncilState) -> dict[str, Any]:
    """analyst 节点（并行 worker）：单个分析师产出结构化报告。

    assigned_role 由 dispatcher 的 Send payload 注入（仅本分支可见）。
    """
    role: str = state.get("assigned_role", "未知")
    agent = AnalystAgent(role=role, confidence=ROLE_CONFIDENCE.get(role, 0.70))
    report: AnalystReport = agent.run(state)
    update: dict[str, Any] = {"analyst_reports": [report]}
    errors: list[str] = ([report.error] if report.error else []) + list(report.tool_errors)
    if errors:
        update["errors"] = errors
    update["messages"] = [
        AIMessage(content=f"【trace】{report.agent}（{report.role}）完成，置信度 {report.confidence}")
    ]
    return update


@fail_open_node(
    lambda exc, state: {
        # 串行节点失败时产出全失败态报告（fail-open），供评测对比场景使用
        "analyst_reports": [
            AnalystReport(
                agent="analyst",
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                role=role,
                summary="",
                key_points=[],
                confidence=0.0,
            )
            for role in ANALYST_ROLES
        ]
    }
)
def run_analyst_team_serial(state: CouncilState) -> dict[str, Any]:
    """串行版分析师团队节点（仅用于「并行 vs 串行」耗时对比评测）。

    与 run_analyst_team（Send 图级并行）产出完全一致：三名分析师依次
    执行后一次性聚合 analyst_reports；由 eval/parallelism_test.py 通过
    build_council_graph(serial_analysts=True) 挂载，主链路不使用。
    """
    reports: list[AnalystReport] = []
    errors: list[str] = []
    for role in ANALYST_ROLES:
        branch: CouncilState = {**state, "assigned_role": role}
        agent = AnalystAgent(role=role, confidence=ROLE_CONFIDENCE.get(role, 0.70))
        report: AnalystReport = agent.run(branch)
        reports.append(report)
        if report.error:
            errors.append(report.error)
        errors.extend(report.tool_errors)
    update: dict[str, Any] = {"analyst_reports": reports}
    if errors:
        update["errors"] = errors
    update["messages"] = [
        AIMessage(content=f"【trace】分析师团队串行完成，共 {len(reports)} 位")
    ]
    return update


@fail_open_node(
    lambda exc, state: {
        # 节点失败也推进轮次计数：保证循环必然终止（防死循环）
        "debate_round": state.get("debate_round", 0) + 1,
        "debate_history": [
            DebateRound(
                round_no=state.get("debate_round", 0) + 1,
                bull_argument="",
                bear_argument="",
                error=f"{type(exc).__name__}: {exc}",
            )
        ],
    }
)
def run_researcher_debate(state: CouncilState) -> dict[str, Any]:
    """researcher_debate 节点：执行一轮多空辩论；达到最大轮次时收敛产出 debate_report。

    循环由 conditional_edges（route_after_debate）+ debate_round 计数实现，
    max_rounds 保证无论节点成败循环都能终止。
    """
    query: str = state["query"]
    round_no: int = state.get("debate_round", 0) + 1
    round_result: DebateRound = ResearcherAgent(round_no=round_no).run(state)
    update: dict[str, Any] = {"debate_round": round_no, "debate_history": [round_result]}
    if round_result.error:
        update["errors"] = [round_result.error]
    update["messages"] = [AIMessage(content=f"【trace】辩论第 {round_no} 轮完成")]
    # 达到最大轮次：聚合历史产出收敛报告（真实模式 LLM 汇总，失败降级模板）
    if round_no >= config_module.CONFIG.debate_max_rounds:
        history: list[DebateRound] = list(state.get("debate_history", [])) + [round_result]
        update["debate_report"] = ResearcherAgent(round_no=round_no).synthesize(state, history)
    return update


def route_after_debate(state: CouncilState) -> str:
    """辩论循环的条件路由：未达最大轮次继续辩论，否则进入风控（防死循环）。

    注意：本函数是 conditional_edges 的路由函数（非节点），必须快速且不抛异常；
    计数缺失时按 0 处理，进入辩论循环而非卡死。
    """
    if state.get("debate_round", 0) < config_module.CONFIG.debate_max_rounds:
        return "researcher_debate"
    return "risk_review"


@fail_open_node(
    lambda exc, state: {
        "risk_report": RiskReviewReport(
            agent="risk_review",
            status=AgentStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            risk_level="高",
            risks=[
                RiskItem(
                    title="风控节点失败",
                    level="高",
                    description=f"风控评审未完成：{exc}",
                    mitigation="人工复核。",
                )
            ],
            blocked=False,
            comment="",
        )
    }
)
def run_risk_review(state: CouncilState) -> dict[str, Any]:
    """risk_review 节点：风控评审。"""
    report: RiskReviewReport = RiskAgent().run(state)
    update: dict[str, Any] = {"risk_report": report}
    if report.error:
        update["errors"] = [report.error]
    update["messages"] = [
        AIMessage(content=f"【trace】风控评审完成，整体风险等级：{report.risk_level}")
    ]
    return update


@fail_open_node(
    lambda exc, state: {
        "final_decision": FinalDecision(
            agent="manager_summarize",
            status=AgentStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            rating="中性",
            confidence=0.0,
            summary="【stub】manager_summarize 节点整体失败，无法形成投资分析。",
            reasoning=f"经理汇总失败：{exc}",
            key_points=[],
        )
    }
)
def run_manager_summarize(state: CouncilState) -> dict[str, Any]:
    """manager_summarize 节点：基金经理汇总全链路产出 final_decision。

    fail-open 保证：无论上游任何阶段失败，本节点必然产出 final_decision。
    """
    report: FinalDecision = ManagerAgent().run(state)
    update: dict[str, Any] = {"final_decision": report}
    if report.error:
        update["errors"] = [report.error]
    update["messages"] = [
        AIMessage(content=f"【trace】基金经理汇总完成，评级：{report.rating}，置信度 {report.confidence}")
    ]
    return update
