"""FastAPI + SSE 服务测试：事件流完整性与异常自动降级。

固定 MOCK_MODE=true（确定性、不依赖外部服务）：验证 SSE 事件序列
（各角色报告逐步推送、辩论轮数等于上限不死循环、最终决策必然到达），
以及工具故障时流的降级行为（error 事件留痕 + 正常产出 final_decision）。

运行（项目根目录）：
    python -m eval.api_test
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# 保证无论以 `-m` 还是直接执行脚本方式运行，都能从项目根目录导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 固定 mock 链路（确定性）：须在导入 config 前设置
os.environ["MOCK_MODE"] = "true"

import config as config_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.server import app  # noqa: E402

client = TestClient(app)


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 SSE 响应体为 [(event 类型, data dict), ...] 列表。"""
    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            data = json.loads(line[len("data: "):])
            events.append((current_event or "", data))
            current_event = None
    return events


def _collect_stream(query: str) -> list[tuple[str, dict[str, Any]]]:
    """发起一次 SSE 请求并收集全部事件（TestClient 流式读取）。"""
    chunks: list[str] = []
    with client.stream("GET", f"/stream", params={"query": query}) as resp:
        assert resp.status_code == 200, f"SSE 请求失败: {resp.status_code}"
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            chunks.append(line)
    return _parse_sse("\n".join(chunks))


def scenario_1_health_and_analyze() -> None:
    """场景 1：/health 存活检查 + POST /analyze 非流式最终决策。"""
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok", f"health 异常: {h.json()}"
    r = client.post("/analyze", json={"query": "600519.SH"})
    assert r.status_code == 200, f"analyze 失败: {r.status_code} {r.text}"
    decision: dict[str, Any] = r.json()
    assert decision["rating"] in {"买入", "增持", "中性", "减持", "卖出"}, f"评级非法: {decision['rating']}"
    assert 0.0 <= decision["confidence"] <= 1.0
    assert decision["summary"] and decision["reasoning"]
    print(f"[1/3] health + analyze PASS：评级 {decision['rating']}，置信度 {decision['confidence']}")


def scenario_2_sse_event_sequence() -> None:
    """场景 2：SSE 事件序列完整——三角色报告 + 逐轮辩论 + 风控 + 最终决策。"""
    events = _collect_stream("600519.SH")
    types: list[str] = [e for e, _ in events]
    # 起始与结束事件
    assert types[0] == "run_started", f"首个事件应为 run_started: {types[:3]}"
    assert types[-1] == "done", f"末尾事件应为 done: {types[-3:]}"
    # 三分析师报告逐步推送（Send 并行产出，角色各一份）
    analyst_roles = {d["role"] for e, d in events if e == "analyst_report"}
    assert analyst_roles == {"基本面", "技术面", "情绪面"}, f"分析师事件不全: {analyst_roles}"
    # 辩论逐轮推送且轮数严格等于上限（conditional_edges + max_rounds 防死循环）
    rounds = [d for e, d in events if e == "debate_round"]
    max_rounds = config_module.CONFIG.debate_max_rounds
    assert len(rounds) == max_rounds, f"辩论事件 {len(rounds)} != 上限 {max_rounds}（存在死循环或提前终止）"
    assert [d["round_no"] for d in rounds] == list(range(1, max_rounds + 1)), "辩论轮次序号异常"
    # 风控与最终决策事件齐备
    risk_events = [d for e, d in events if e == "risk_report"]
    assert risk_events and risk_events[0]["risk_level"] in {"低", "中", "高"}
    final_events = [d for e, d in events if e == "final_decision"]
    assert final_events and final_events[0]["rating"] in {"买入", "增持", "中性", "减持", "卖出"}
    # done 事件携带最终决策与耗时
    done = events[-1][1]
    assert done["final_decision"]["rating"] == final_events[0]["rating"]
    assert done["elapsed_s"] > 0
    # 事件顺序：分析师 → 辩论 → 风控 → 最终决策（链路拓扑顺序）
    order = [types.index(t) for t in ("analyst_report", "debate_round", "debate_report", "risk_report", "final_decision")]
    assert order == sorted(order), f"事件顺序违反链路拓扑: {order}"
    print(
        f"[2/3] SSE 事件序列 PASS：{len(events)} 条事件——"
        f"三分析师逐份 + 辩论 {len(rounds)} 轮（无死循环）+ 风控 + 最终决策（{final_events[0]['rating']}）"
    )


def scenario_3_sse_degrade_on_failure() -> None:
    """场景 3：工具故障时 SSE 流自动降级——error 留痕 + 最终决策仍产出。"""
    os.environ["DEMO_TOOL_FAILURE"] = "true"
    config_module.reload_config()
    try:
        events = _collect_stream("600519.SH")
    finally:
        os.environ.pop("DEMO_TOOL_FAILURE", None)
        config_module.reload_config()
    types: list[str] = [e for e, _ in events]
    assert "error" in types, "工具故障应以 error 事件留痕"
    assert types[-1] == "done", "故障下降级流也必须以 done 正常收尾"
    done = events[-1][1]
    decision = done["final_decision"]
    assert decision is not None and decision["status"] == "ok", "故障下仍须产出最终决策"
    # 失败传导：风控「信息不完整」→ 评级降为 减持（代码兜底规则）
    assert decision["rating"] == "减持", f"故障传导评级应为减持: {decision['rating']}"
    error_msgs = [d["messages"] for e, d in events if e == "error"]
    print(
        f"[3/3] SSE 降级 PASS：工具故障 → {sum(len(m) for m in error_msgs)} 条 error 事件留痕，"
        f"流正常收尾且最终决策降级为 {decision['rating']}（fail-open 不中断）"
    )


def main() -> int:
    """依次执行三个场景；任一失败即非零退出。"""
    print("=== AgentCouncil FastAPI + SSE 服务测试（mock 链路，全部基于真实运行） ===")
    scenario_1_health_and_analyze()
    scenario_2_sse_event_sequence()
    scenario_3_sse_degrade_on_failure()
    print("=== 全部场景 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
