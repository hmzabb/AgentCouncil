"""AgentCouncil 命令行入口（多 Agent 协作：四数据工具 + DeepSeek 全链路）。

用法示例：
    python main.py --query "600519.SH" --show-analysts   # 股票代码 + 三分析师报告
    python main.py --query "AI 算力"                      # 行业关键词

输出约定：运行过程信息写入 stderr，final_decision 以 JSON 写入 stdout，
便于管道消费（如 `python main.py | jq .rating`）。

注意：MOCK_MODE=false（默认）时行情/财务/新闻走 AkShare、研报检索走
Milvus（需 docker milvus-standalone 已启动且已运行 data/ingest.py 建库）；
任何工具/LLM 失败均 fail-open 留痕到 pipeline_errors，不伪造数据。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import config as config_module  # 动态读取配置（打印当前模式）

from graph.builder import build_council_graph

DEFAULT_QUERY: str = "600519.SH"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="AgentCouncil：多智能体金融分析系统（三分析师并行 + 辩论 + 风控 + 经理决策）"
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="股票代码或行业关键词（默认：%s）" % DEFAULT_QUERY,
    )
    parser.add_argument(
        "--show-analysts",
        action="store_true",
        help="在 stderr 打印三名分析师（基本面/技术面/情绪面）的报告",
    )
    parser.add_argument(
        "--show-debate",
        action="store_true",
        help="在 stderr 逐轮打印多空辩论（验证 conditional_edges 循环与轮次上限）",
    )
    return parser.parse_args(argv)


def run_council(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """运行完整链路并返回最终状态（含 final_decision）。

    参数：
        query:     股票代码或行业关键词（非空字符串）。
        thread_id: checkpointer 线程 ID；None 时自动生成新线程。
                   复用同一 thread_id 可断点续跑（内存版 MemorySaver，
                   进程重启后失效；持久化后续换 SqliteSaver）。

    返回：
        graph.invoke 后的最终 state dict；final_decision 为 FinalDecision 实例。

    异常：
        ValueError: query 为空时抛出（唯一会主动抛出的错误，属于入参问题）。
    """
    query = query.strip()
    if not query:
        raise ValueError("查询不能为空")
    graph = build_council_graph()
    final_state: dict[str, Any] = graph.invoke(
        {"query": query},
        config={"configurable": {"thread_id": thread_id or f"stub-{uuid.uuid4().hex[:8]}"}},
    )
    return final_state


def _print_analyst_reports(final_state: dict[str, Any]) -> None:
    """在 stderr 打印三名分析师的报告（--show-analysts 演示用）。"""
    reports = final_state.get("analyst_reports", [])
    for r in reports:
        print(f"--- {r.role}分析师（置信度 {r.confidence}，状态 {r.status.value}）---", file=sys.stderr)
        print(f"  摘要：{r.summary}", file=sys.stderr)
        for kp in r.key_points:
            print(f"    - {kp}", file=sys.stderr)
        for src in r.sources:
            print(f"  来源：{src}", file=sys.stderr)
        if r.tool_errors:
            print(f"  工具留痕：{'；'.join(r.tool_errors)}", file=sys.stderr)
        if r.error:
            print(f"  失败原因：{r.error}", file=sys.stderr)


def _print_debate(final_state: dict[str, Any]) -> None:
    """在 stderr 逐轮打印多空辩论与收敛结论（--show-debate 演示用）。

    轮次上限由 conditional_edges + max_rounds 保证：打印的轮数必然等于
    debate_max_rounds（节点失败也会推进计数），不存在死循环。
    """
    history = final_state.get("debate_history", [])
    max_rounds = config_module.CONFIG.debate_max_rounds
    print(f"=== 多空辩论（{len(history)}/{max_rounds} 轮，conditional_edges 防死循环） ===", file=sys.stderr)
    for h in history:
        print(f"--- 第 {h.round_no} 轮 ---", file=sys.stderr)
        print(f"  多方：{h.bull_argument or '（空）'}", file=sys.stderr)
        print(f"  空方：{h.bear_argument or '（空）'}", file=sys.stderr)
        if h.error:
            print(f"  留痕：{h.error}", file=sys.stderr)
    debate = final_state.get("debate_report")
    if debate is not None:
        print(f"--- 收敛结论（{debate.total_rounds} 轮） ---", file=sys.stderr)
        print(f"  {debate.conclusion}", file=sys.stderr)
        for q in debate.open_questions:
            print(f"  未决问题：{q}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：运行链路并输出 final_decision JSON。返回进程退出码。"""
    args = parse_args(argv)
    print(f"=== AgentCouncil：查询「{args.query}」（MOCK_MODE={'true' if config_module.CONFIG.mock_mode else 'false'}）===", file=sys.stderr)
    final_state = run_council(args.query)
    if args.show_analysts:
        _print_analyst_reports(final_state)
    if args.show_debate:
        _print_debate(final_state)
    decision = final_state.get("final_decision")
    if decision is None:
        # 理论上不会发生：所有节点均被 fail-open 兜底，manager 必产出决策
        print("错误：链路未产出 final_decision", file=sys.stderr)
        return 1
    # mode="json"：枚举等类型转为 JSON 原生值（否则 str 枚举会序列化成 "AgentStatus.OK"）
    payload: Any = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else decision
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
