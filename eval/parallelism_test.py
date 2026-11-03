"""并行 vs 串行端到端耗时对比：LangGraph Send 图级并行 vs 单节点串行循环。

方法（保证对比公平，全部基于真实运行计时）：
- 同一 query、同一环境（真实模式：AkShare 行情/财务/新闻 + Milvus RAG +
  DeepSeek LLM）、同一批 AnalystAgent 与节点实现；
- 并行图：主链路默认结构，analyst_team 分发器返回
  Command(goto=[Send("analyst", ...)×3])，三名分析师图级并行
  （各分支独立执行与检查点，LangGraph fan-in 汇聚）；
- 串行图：仅把分析师团队替换为单节点 for 循环
  （run_analyst_team_serial，build_council_graph(serial_analysts=True)），
  其余节点（辩论/风控/经理）与并行图完全一致；
- 每次运行独立 thread_id（避免 checkpointer 状态干扰计时）；
- 计时粒度：整图端到端 wall clock + 按超步（superstep）分阶段
  wall clock（graph.stream(stream_mode="updates") 真实计时，非估算）。

前置条件：
    docker start milvus-standalone
    python data/ingest.py        # 已建库
    .env 已配置 DEEPSEEK_API_KEY

运行（项目根目录）：
    python -m eval.parallelism_test
    python -m eval.parallelism_test --query 000001.SZ

注意：LLM 调用与外部数据接口耗时存在波动，加速比仅供参考（可重跑
取均值；串行图的 LLM 请求无法被服务端复用，波动属正常现象）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# 保证无论以 `-m` 还是直接执行脚本方式运行，都能从项目根目录导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 固定真实模式：须在导入 config 前设置（config 导入时加载 .env，dotenv 不覆盖已有变量）
os.environ["MOCK_MODE"] = "false"

import config as config_module  # noqa: E402
from agents.schemas import AgentStatus  # noqa: E402
from graph.builder import build_council_graph  # noqa: E402

DEFAULT_QUERY: str = "600519.SH"


def run_timed(query: str, serial: bool) -> tuple[float, list[tuple[str, float]], dict[str, Any]]:
    """运行一次完整链路并计时。

    返回：
        (端到端总耗时秒, [(超步名, 该超步耗时秒), ...], 最终 state)。

    超步名：该超步内完成的节点名（+ 连接），如 "analyst_team"、
    "analyst"（并行图的三分支同属一个超步）、"researcher_debate"。
    """
    graph = build_council_graph(serial_analysts=serial)
    # 独立 thread_id：避免 MemorySaver 复用上次运行的状态
    thread_id: str = f"bench-{'serial' if serial else 'parallel'}-{uuid.uuid4().hex[:8]}"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    stages: list[tuple[str, float]] = []
    t0: float = time.perf_counter()
    last: float = t0
    for chunk in graph.stream({"query": query}, config=config, stream_mode="updates"):
        now: float = time.perf_counter()
        # 一个 chunk = 一个超步：并行分支在同一超步内完成
        stages.append(("+".join(sorted(chunk.keys())), now - last))
        last = now
    total: float = time.perf_counter() - t0
    # 完整最终 state 从 checkpointer 读取（updates 模式的 chunk 只是节点增量）
    final_state: dict[str, Any] = graph.get_state(config).values
    return total, stages, final_state


def verify(final_state: dict[str, Any], label: str) -> None:
    """校验两种结构产出等价：三份分析师报告 + 最终决策齐备。"""
    reports = final_state.get("analyst_reports", [])
    roles = {r.role for r in reports}
    assert roles == {"基本面", "技术面", "情绪面"}, f"{label}：分析师角色不全: {roles}"
    decision = final_state.get("final_decision")
    assert decision is not None and decision.status is AgentStatus.OK, (
        f"{label}：未产出合法最终决策: {decision}"
    )
    print(f"  [产出校验 PASS] {label}：{len(reports)} 份分析师报告，评级 {decision.rating}")


def print_stages(label: str, stages: list[tuple[str, float]], total: float) -> None:
    """打印分超步耗时与端到端总耗时。"""
    print(f"=== {label} ===")
    for name, seconds in stages:
        print(f"  超步 [{name:<32}] {seconds:7.2f}s")
    print(f"  >>> 端到端总耗时：{total:.2f}s")


def main(argv: list[str] | None = None) -> int:
    """串行/并行各跑一次，输出耗时对比与加速比。"""
    parser = argparse.ArgumentParser(description="AgentCouncil 并行 vs 串行端到端耗时对比")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="股票代码（默认 %s）" % DEFAULT_QUERY)
    args = parser.parse_args(argv)

    print(
        f"=== AgentCouncil 并行 vs 串行耗时对比（query={args.query}，"
        f"MOCK_MODE=false，真实数据 + DeepSeek） ==="
    )
    print("（先跑串行图，再跑并行图；每次独立 thread_id）\n")

    serial_total, serial_stages, serial_state = run_timed(args.query, serial=True)
    verify(serial_state, "串行图")
    print_stages("串行图（analyst_team 单节点 for 循环）", serial_stages, serial_total)

    print()
    parallel_total, parallel_stages, parallel_state = run_timed(args.query, serial=False)
    verify(parallel_state, "并行图")
    print_stages("并行图（analyst_team Send 图级并行）", parallel_stages, parallel_total)

    print()
    speedup: float = serial_total / parallel_total
    print(f"=== 对比结论 ===")
    print(f"  串行端到端：{serial_total:.2f}s")
    print(f"  并行端到端：{parallel_total:.2f}s")
    print(f"  加速比：{speedup:.2f}x（并行节省 {serial_total - parallel_total:.2f}s）")
    print(
        "  注：分析师阶段是主要可并行部分（3 个工具取数 + 3 次 LLM 调用）；\n"
        "      辩论/风控/经理为串行依赖节点，两种结构耗时一致。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
