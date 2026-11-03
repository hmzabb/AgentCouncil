"""Agent 结构化输出 schema（Pydantic v2）。

约定：
- 每个 Agent 的输出都由此处的 schema 严格约束（extra 字段一律拒绝）；
- fail-open：Agent 执行失败时也产出合法的失败态输出（status="failed"），
  保证下游节点永远能拿到结构完整的数据；
- 后续接入真实模型时，只需在 Agent 内填充相同结构，schema 保持不变。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentStatus(str, Enum):
    """Agent 执行状态。"""

    OK = "ok"
    FAILED = "failed"


class BaseAgentOutput(BaseModel):
    """所有 Agent 报告的公共字段。"""

    model_config = ConfigDict(extra="forbid")  # 拒绝未声明字段，保证结构稳定

    agent: str = Field(description="产出该输出的 Agent 名称")
    status: AgentStatus = Field(description="执行状态：ok / failed（fail-open）")
    error: str | None = Field(default=None, description="失败原因；status=failed 时非空")
    note: str = Field(default="【stub】占位输出，未接入真实模型", description="实现阶段说明")


class AnalystReport(BaseAgentOutput):
    """单个分析师的报告（analyst 节点产出，Send 图级并行聚合）。"""

    role: str = Field(description="分析师角色：基本面 / 技术面 / 情绪面")
    summary: str = Field(description="结论摘要")
    key_points: list[str] = Field(description="核心论点")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 [0,1]")
    sources: list[str] = Field(default_factory=list, description="依据来源（stub 为占位来源）")
    tool_errors: list[str] = Field(default_factory=list, description="本报告调用工具的失败留痕（fail-open）")

    @model_validator(mode="after")
    def _ok_requires_points(self) -> "AnalystReport":
        """业务约束：status=ok 时必须有核心论点；失败态（fail-open）不受限。"""
        if self.status is AgentStatus.OK and not self.key_points:
            raise ValueError("status=ok 时 key_points 不能为空")
        return self


class DebateRound(BaseModel):
    """单轮多空辩论记录（researcher_debate 循环节点产出，进入 debate_history）。"""

    model_config = ConfigDict(extra="forbid")

    round_no: int = Field(ge=1, description="轮次序号")
    bull_argument: str = Field(description="多方论点")
    bear_argument: str = Field(description="空方论点")
    error: str | None = Field(default=None, description="本轮工具/执行失败留痕（fail-open）")


class DebateReport(BaseAgentOutput):
    """研究员辩论收敛报告（辩论循环达到最大轮次时产出）。"""

    total_rounds: int = Field(ge=0, description="实际完成的辩论轮数")
    pro_arguments: list[str] = Field(description="多方核心论点汇总")
    con_arguments: list[str] = Field(description="空方核心论点汇总")
    conclusion: str = Field(description="辩论收敛结论")
    open_questions: list[str] = Field(default_factory=list, description="未决问题")
    sources: list[str] = Field(default_factory=list, description="依据来源")


class RiskItem(BaseModel):
    """单条风险。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="风险名称")
    level: Literal["低", "中", "高"] = Field(description="风险等级")
    description: str = Field(description="风险描述")
    mitigation: str = Field(description="缓释措施")


class RiskReviewReport(BaseAgentOutput):
    """风控评审报告（risk_review 节点产出）。"""

    risk_level: Literal["低", "中", "高"] = Field(description="整体风险等级")
    risks: list[RiskItem] = Field(description="风险清单")
    blocked: bool = Field(description="是否一票否决（stub 恒为 False）")
    comment: str = Field(description="评审意见")


class FinalDecision(BaseAgentOutput):
    """基金经理最终决策（manager_summarize 节点产出，即 final_decision）。

    面向用户的最终输出：带置信度的投资分析摘要 + 风险提示，过程可解释、可追溯。
    """

    rating: Literal["买入", "增持", "中性", "减持", "卖出"] = Field(description="投资评级")
    confidence: float = Field(ge=0.0, le=1.0, description="综合置信度 [0,1]")
    summary: str = Field(description="投资分析摘要（中文）")
    reasoning: str = Field(description="决策理由（过程可解释）")
    key_points: list[str] = Field(description="采纳的核心论点")
    risk_hints: list[str] = Field(default_factory=list, description="风险提示")
    sources: list[str] = Field(default_factory=list, description="引用来源汇总")
    pipeline_errors: list[str] = Field(default_factory=list, description="全链路 fail-open 错误留痕")


# ---------------------------------------------------------------------------
# LLM 结构化输出 schema（DeepSeek 接入）
#
# 与报告 schema 分离的原因：工具调用痕迹（tool_errors / sources /
# pipeline_errors）与执行状态（status / error）必须由代码真实填充，
# 不能让模型产出；LLM 只负责「内容型」字段，代码合并组装成完整报告。
# ---------------------------------------------------------------------------


class AnalystLlmOutput(BaseModel):
    """分析师 LLM 输出：内容型字段（工具痕迹由代码合并）。"""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="结论摘要（中文，基于给定数据）")
    key_points: list[str] = Field(min_length=1, max_length=6, description="核心论点（3 条左右，须引用具体数据）")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 [0,1]")


class DebateLlmOutput(BaseModel):
    """研究员单轮辩论 LLM 输出：多方/空方论点。"""

    model_config = ConfigDict(extra="forbid")

    bull_argument: str = Field(description="多方论点（引用数据或检索证据）")
    bear_argument: str = Field(description="空方论点（引用数据或检索证据）")


class DebateSynthesisLlmOutput(BaseModel):
    """研究员辩论收敛 LLM 输出。"""

    model_config = ConfigDict(extra="forbid")

    conclusion: str = Field(description="多空辩论收敛结论（中文）")
    open_questions: list[str] = Field(default_factory=list, description="未决问题")


class RiskLlmOutput(BaseModel):
    """风控 LLM 输出：风险清单与评审意见（整体等级由代码取清单最高等级）。"""

    model_config = ConfigDict(extra="forbid")

    risks: list[RiskItem] = Field(min_length=1, description="风险清单（title/level/description/mitigation）")
    comment: str = Field(description="评审意见（中文）")


class FinalDecisionLlmOutput(BaseModel):
    """基金经理 LLM 输出：内容型决策字段（sources/pipeline_errors 由代码合并）。"""

    model_config = ConfigDict(extra="forbid")

    rating: Literal["买入", "增持", "中性", "减持", "卖出"] = Field(description="投资评级")
    confidence: float = Field(ge=0.0, le=1.0, description="综合置信度 [0,1]")
    summary: str = Field(description="投资分析摘要（中文）")
    reasoning: str = Field(description="决策理由（过程可解释）")
    key_points: list[str] = Field(min_length=1, description="采纳的核心论点")
    risk_hints: list[str] = Field(default_factory=list, description="风险提示")
