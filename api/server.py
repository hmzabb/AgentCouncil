"""FastAPI 服务层：SSE 流式推送各角色报告与最终决策（决策闭环）。

设计（CLAUDE.md：FastAPI + SSE 流式）：
- GET /stream?query=600519.SH：SSE 流，按链路进度逐步推送事件——
  run_started → analyst_report（三分析师逐份，Send 并行产出顺序不定）
  → debate_round（逐轮，轮次上限 conditional_edges + max_rounds 保证，
  不会死循环）→ debate_report → risk_report → final_decision → done；
  异常自动降级：任何 Agent/工具失败不中断流（fail-open 以 error 事件
  留痕，最终必产出 final_decision）；
- POST /analyze：非流式便捷接口，直接返回最终决策 JSON；
- GET /health：存活检查（含当前 mock 模式）。

实现要点：
- LangGraph 图是阻塞实现（AkShare / LLM 同步调用），流式生成器把
  graph.stream 放进工作线程，经 asyncio.Queue + call_soon_threadsafe
  桥接为 async 生成器，避免阻塞事件循环；
- 每个连接独立 thread_id（MemorySaver checkpointer 隔离），传同一
  thread_id 可断点续跑；
- 事件数据一律 model_dump(mode="json") 序列化（枚举/中文安全）。

运行（项目根目录）：
    python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
演示：
    curl -N "http://127.0.0.1:8000/stream?query=600519.SH"
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

# 保证 `python -m uvicorn api.server:app` 从项目根目录启动时能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as config_module  # noqa: E402
from agents.analyst import ANALYST_ROLES  # noqa: E402 —— 分析师团队启动事件
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from graph.builder import build_council_graph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# ------------------------------------------------------------------ SSE 工具


def _dump(obj: Any) -> Any:
    """Pydantic 模型 → JSON 安全 dict（mode="json" 处理枚举与中文）。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _sse(event: str, data: dict[str, Any]) -> str:
    """格式化一条 SSE 消息：`event: <type>\\ndata: <json>\\n\\n`。"""
    payload: str = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _events_for_chunk(chunk: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """把一个超步（updates chunk）翻译为 SSE 事件序列。

    - analyst 分支可能同超步聚合为 list（或逐分支分超步），两种形态都处理；
    - 各节点 update 中的 errors（fail-open 留痕）以非致命 error 事件推送。
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for node, update in chunk.items():
        if node == "analyst_team":
            # 分析师团队节点（并行模式为纯分发器，update 通常为 None/仅 errors；
            # 串行模式为本节点携带全部报告）：先宣告团队启动
            events.append(("analyst_team_started", {"roles": list(ANALYST_ROLES)}))
        updates: list[dict[str, Any]] = update if isinstance(update, list) else [update]
        for u in updates:
            if not isinstance(u, dict):
                continue  # 节点无更新内容（如纯分发器超步 update=None）
            if u.get("errors"):
                events.append(("error", {"node": node, "messages": [str(e) for e in u["errors"]]}))
            for report in u.get("analyst_reports", []):
                events.append(("analyst_report", _dump(report)))
            for rnd in u.get("debate_history", []):
                events.append(("debate_round", _dump(rnd)))
            if u.get("debate_report") is not None:
                events.append(("debate_report", _dump(u["debate_report"])))
            if u.get("risk_report") is not None:
                events.append(("risk_report", _dump(u["risk_report"])))
            if u.get("final_decision") is not None:
                events.append(("final_decision", _dump(u["final_decision"])))
    return events


async def _event_stream(query: str, thread_id: str) -> AsyncIterator[str]:
    """SSE 事件生成器：工作线程跑阻塞图，经队列桥接为 async 流。

    异常自动降级：图内 fail-open 保证不抛；即使意外异常也只发
    fatal_error 事件后正常关闭流，不向客户端抛 500。
    """
    graph = build_council_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def _produce() -> None:
        """阻塞生产者：跑完整图并把超步/最终状态放入队列。"""
        t0: float = time.perf_counter()
        try:
            for chunk in graph.stream({"query": query}, config=config, stream_mode="updates"):
                loop.call_soon_threadsafe(queue.put_nowait, {"chunk": chunk})
            final_state: dict[str, Any] = graph.get_state(config).values
            loop.call_soon_threadsafe(
                queue.put_nowait, {"state": final_state, "elapsed": time.perf_counter() - t0}
            )
        except Exception as exc:  # noqa: BLE001 —— SSE 层兜底：流不中断
            loop.call_soon_threadsafe(
                queue.put_nowait, {"fatal": f"{type(exc).__name__}: {exc}"}
            )

    threading.Thread(target=_produce, daemon=True, name=f"council-{thread_id[:8]}").start()
    yield _sse("run_started", {"query": query, "thread_id": thread_id})
    while True:
        item: dict[str, Any] = await queue.get()
        if item.get("fatal"):
            yield _sse("fatal_error", {"message": item["fatal"]})
            return
        if "chunk" in item:
            for event, data in _events_for_chunk(item["chunk"]):
                yield _sse(event, data)
            continue
        if "state" in item:
            decision = item["state"].get("final_decision")
            yield _sse(
                "done",
                {
                    "elapsed_s": round(item["elapsed"], 2),
                    "final_decision": _dump(decision) if decision is not None else None,
                },
            )
            return


# ------------------------------------------------------------------ FastAPI 应用

app = FastAPI(
    title="AgentCouncil",
    description="多智能体金融分析系统：SSE 流式推送各角色报告与最终决策",
    version="0.4.0",
)
# 前端演示用：允许跨域（本地 demo，不上线）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    """POST /analyze 请求体。"""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, description="股票代码或行业关键词")
    thread_id: str | None = Field(default=None, description="断点续跑线程 ID（可选）")


@app.get("/health")
def health() -> dict[str, Any]:
    """存活检查：返回服务状态与当前模式。"""
    return {"status": "ok", "mock_mode": config_module.CONFIG.mock_mode}


@app.get("/stream")
async def stream(query: str = Query(..., min_length=1, description="股票代码或行业关键词"),
                 thread_id: str | None = Query(None, description="断点续跑线程 ID（可选）")) -> StreamingResponse:
    """SSE 流式分析：逐步推送 run_started/analyst_report/debate_round/
    debate_report/risk_report/final_decision/done 事件。"""
    tid: str = thread_id or f"sse-{uuid.uuid4().hex[:12]}"
    return StreamingResponse(
        _event_stream(query.strip(), tid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用反向代理缓冲，保证逐条推送
        },
    )


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    """非流式分析：直接返回最终决策 JSON（同步端点，FastAPI 自动线程池执行）。"""
    query: str = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    graph = build_council_graph()
    final_state: dict[str, Any] = graph.invoke(
        {"query": query},
        config={"configurable": {"thread_id": payload.thread_id or f"api-{uuid.uuid4().hex[:8]}"}},
    )
    decision = final_state.get("final_decision")
    if decision is None:
        # 理论上不会发生：所有节点均被 fail-open 兜底，manager 必产出决策
        raise HTTPException(status_code=500, detail="链路未产出 final_decision")
    return _dump(decision)


@app.get("/")
def index() -> HTMLResponse:
    """前端演示页面：直接浏览 http://127.0.0.1:8000/ 即可使用。"""
    static_dir = Path(__file__).resolve().parent / "static"
    html_path = static_dir / "index.html"
    if not html_path.is_file():
        return HTMLResponse(content="<h2>index.html not found under api/static/</h2>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))