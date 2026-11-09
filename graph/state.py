"""Council 链路共享状态（LangGraph StateGraph 的 state schema）。

对应 CLAUDE.md 架构要求：State 含 messages、各角色报告、
debate_history（多空）、risk_report、final_decision。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agents.schemas import (
    AnalystReport,
    DebateReport,
    DebateRound,
    FinalDecision,
    RiskReviewReport,
)


class CouncilState(TypedDict, total=False):
    """AgentCouncil 主链路状态。

    说明：
    - total=False：除 query 外的字段均为「阶段产物」，由对应节点逐步填充；
    - messages 使用 add_messages 合并（过程可追溯 + LLM 上下文预留）；
    - analyst_reports / debate_history / errors 使用 operator.add 追加式合并，
      配合 Send 并行：各分支只回传本分支新增内容，LangGraph 在汇聚点自动合并；
    - assigned_role 仅作为 Send payload 的分支内工作分配字段；
    - checkpointer（内存版）已接入：state 值需可序列化，Pydantic 模型由
      langgraph 的序列化器处理；后续持久化可换 SqliteSaver。
    """

    query: str                                             # 输入：股票代码或行业关键词
    assigned_role: str                                     # Send 并行：分发给单个分析师的角色（分支内可见）
    deadline_ts: float                                     # 全链路预算：monotonic 截止时间戳（<=0 表示不限制）
    messages: Annotated[list[BaseMessage], add_messages]   # 过程消息（可追溯 + LLM 上下文预留）
    analyst_reports: Annotated[list[AnalystReport], operator.add]   # 分析师报告（Send 并行结果聚合）
    debate_round: int                                      # 辩论轮次计数（conditional_edges 防死循环）
    debate_history: Annotated[list[DebateRound], operator.add]     # 多空辩论历史
    debate_report: DebateReport                            # 辩论收敛报告
    risk_report: RiskReviewReport                          # 风控报告
    final_decision: FinalDecision                          # 基金经理最终决策
    errors: Annotated[list[str], operator.add]             # 各节点错误追加留痕