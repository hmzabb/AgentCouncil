"""DeepSeek LLM 接入测试：真实调用 API，验证各 Agent 的结构化输出。

与 smoke_test（mock）/ real_data_test（数据层）互补：本文件验证 LLM 层，
固定 MOCK_MODE=false，全部断言基于真实 API 返回（禁止模拟）。

前置：.env 已配置 DEEPSEEK_API_KEY（未配置则测试失败并给出明确提示）。

场景：
  1. 分析师（技术面）：报告由 DeepSeek 生成，引用真实行情数据；
  2. 研究员：单轮多空辩论 + 最终轮收敛报告；
  3. 风控：基于分析师报告的 LLM 风险清单（整体等级由代码取最高）；
  4. 基金经理：全链路汇总 → 评级/置信度/摘要（风控一票权生效）。

运行（项目根目录）：
    python -m eval.llm_test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MOCK_MODE"] = "false"  # 须在导入 config 前设置

import config as config_module  # noqa: E402
from agents.analyst import AnalystAgent  # noqa: E402
from agents.manager import ManagerAgent  # noqa: E402
from agents.researcher import ResearcherAgent  # noqa: E402
from agents.risk import RiskAgent  # noqa: E402
from agents.schemas import AgentStatus  # noqa: E402
from graph.state import CouncilState  # noqa: E402

QUERY: str = "600519.SH"


def scenario_1_analyst() -> Any:
    """场景 1：技术分析师 LLM 报告（引用真实行情数据）。"""
    report = AnalystAgent(role="技术面", confidence=0.73).run({"query": QUERY})
    assert report.status is AgentStatus.OK, f"分析师失败: {report.error}"
    assert "DeepSeek" in (report.note or ""), f"note 应标注 DeepSeek 生成: {report.note}"
    assert not report.tool_errors, f"工具与 LLM 调用应成功: {report.tool_errors}"
    assert all("【stub】" not in kp for kp in report.key_points), "论点不得含占位文本"
    assert 0.0 < report.confidence <= 1.0
    print(f"[1/4] 分析师 LLM PASS：置信度 {report.confidence}")
    print(f"  摘要：{report.summary}")
    for kp in report.key_points:
        print(f"    - {kp}")
    return report


def scenario_2_researcher(report: Any) -> Any:
    """场景 2：研究员单轮 LLM 辩论 + 最终轮 LLM 收敛。"""
    state: CouncilState = {"query": QUERY}
    round1 = ResearcherAgent(round_no=1).run(state)
    assert round1.bull_argument and round1.bear_argument, f"辩论论点为空: {round1}"
    assert all("【stub】" not in s for s in (round1.bull_argument, round1.bear_argument))
    synthesis = ResearcherAgent(round_no=2).synthesize(state, [round1])
    assert synthesis.conclusion and "DeepSeek" in (synthesis.note or ""), (
        f"收敛报告异常: {synthesis.note}"
    )
    print(f"[2/4] 研究员 LLM PASS：{synthesis.total_rounds} 轮收敛")
    print(f"  多方：{round1.bull_argument[:120]}…")
    print(f"  空方：{round1.bear_argument[:120]}…")
    print(f"  收敛：{synthesis.conclusion[:150]}…")
    return synthesis


def scenario_3_risk(report: Any, synthesis: Any) -> Any:
    """场景 3：风控 LLM 风险清单（代码兜底规则仍生效）。"""
    state: CouncilState = {"query": QUERY, "analyst_reports": [report]}
    risk = RiskAgent().run(state)
    assert risk.status is AgentStatus.OK, f"风控失败: {risk.error}"
    assert "DeepSeek" in (risk.note or ""), f"note 应标注 DeepSeek: {risk.note}"
    assert risk.risks, "风险清单不能为空"
    assert risk.risk_level in {"低", "中", "高"}
    print(f"[3/4] 风控 LLM PASS：整体风险 {risk.risk_level}，{len(risk.risks)} 条风险")
    for item in risk.risks:
        print(f"    - {item.level}风险：{item.title}——{item.mitigation}")
    print(f"  评审意见：{risk.comment}")
    return risk


def scenario_4_manager(report: Any, synthesis: Any, risk: Any) -> None:
    """场景 4：基金经理 LLM 决策（风控一票权生效）。"""
    state: CouncilState = {
        "query": QUERY,
        "analyst_reports": [report],
        "debate_report": synthesis,
        "risk_report": risk,
        "errors": [],
    }
    decision = ManagerAgent().run(state)
    assert decision.status is AgentStatus.OK, f"经理失败: {decision.error}"
    assert "DeepSeek" in (decision.note or ""), f"note 应标注 DeepSeek: {decision.note}"
    assert decision.rating in {"买入", "增持", "中性", "减持", "卖出"}
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.summary and decision.reasoning
    print(f"[4/4] 基金经理 LLM PASS：评级 {decision.rating}，置信度 {decision.confidence}")
    print(f"  摘要：{decision.summary}")
    print(f"  理由：{decision.reasoning[:200]}…")


def main() -> int:
    """依次执行四个场景；任一失败即非零退出。"""
    print("=== AgentCouncil DeepSeek LLM 接入测试（全部基于真实 API 返回） ===")
    if not config_module.CONFIG.deepseek_api_key:
        print("错误：未配置 DEEPSEEK_API_KEY（请复制 .env.example 为 .env 并填写 key）",
              file=sys.stderr)
        return 1
    report = scenario_1_analyst()
    synthesis = scenario_2_researcher(report)
    risk = scenario_3_risk(report, synthesis)
    scenario_4_manager(report, synthesis, risk)
    print("=== 全部场景 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
