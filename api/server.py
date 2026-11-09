"""FastAPI 服务层：SSE 流式推送各角色报告与最终决策（决策闭环）。

设计（CLAUDE.md：FastAPI + SSE 流式）：
- GET /stream?query=600519.SH：SSE 流，按链路进度逐步推送事件——
  run_started → analyst_report（三分析师逐份，Send 并行产出顺序不定）
  → debate_round（逐轮，轮次上限 conditional_edges + max_rounds 保证，
  不会死循环）→ debate_report → risk_report → final_decision → done；
  异常自动降级：任何 Agent/工具失败不中断流（fail-open 以 error 事件
  留痕，最终必产出 final_decision）；
- POST /analyze：非流式便捷接口，直接返回最终决策 JSON；
- GET /health：存活检查（含当前 mock 模式与工具缓存统计）；
- GET /metrics：Prometheus 文本格式指标（请求量/延迟/进行中请求/缓存命中/
  模式），无需额外依赖，可直接被 Prometheus 抓取。

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
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

# 保证 `python -m uvicorn api.server:app` 从项目根目录启动时能导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config as config_module  # noqa: E402
from agents.analyst import ANALYST_ROLES  # noqa: E402 —— 分析师团队启动事件
from graph.builder import build_council_graph  # noqa: E402
from tools.mock_tools import tool_cache_stats  # noqa: E402 —— /health 缓存统计

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
            # 断点续跑：同一 thread_id 已完成则直接回放既有决策，不再执行图
            snapshot = graph.get_state(config)
            existing = (
                snapshot.values.get("final_decision") if snapshot and snapshot.values else None
            )
            if existing is not None:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "state": snapshot.values,
                        "elapsed": time.perf_counter() - t0,
                        "resumed": True,
                    },
                )
                return
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
                    "resumed": bool(item.get("resumed", False)),
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

# 免鉴权 / 免限流路径：健康检查、监控指标与前端页面
_PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/metrics", "/", "/docs", "/openapi.json"})
# 单 IP 请求时间窗（滑动窗口限流，仅在 RATE_LIMIT_PER_MIN > 0 时生效）
_rate_windows: dict[str, deque[float]] = {}
_rate_lock: threading.Lock = threading.Lock()

# 进程内指标（Prometheus 文本格式自实现，不引入额外依赖）：
# 请求计数按 (method, path, status) 维度，延迟按 path 维度累加 sum/count。
_metric_lock: threading.Lock = threading.Lock()
_request_counts: dict[tuple[str, str, str], int] = {}
_request_seconds: dict[str, float] = {}
_request_seconds_count: dict[str, int] = {}
_inflight: int = 0


def _metric_observe(
    method: str, path: str, status: int, seconds: float, count_inflight: bool = True
) -> None:
    """记录一次 HTTP 请求（线程安全；SSE 长连接以流结束时刻计入）。

    count_inflight=False 用于鉴权/限流拒绝：这类请求未计入进行中请求数，
    不能递减 inflight（否则计数会变成负数）。
    """
    global _inflight
    with _metric_lock:
        key: tuple[str, str, str] = (method, path, str(status))
        _request_counts[key] = _request_counts.get(key, 0) + 1
        _request_seconds[path] = _request_seconds.get(path, 0.0) + seconds
        _request_seconds_count[path] = _request_seconds_count.get(path, 0) + 1
        if count_inflight:
            _inflight -= 1


def _inflight_add(delta: int) -> None:
    """更新进行中请求数（SSE 长连接能体现并发压力）。"""
    global _inflight
    with _metric_lock:
        _inflight += delta


@app.middleware("http")
async def _guard(request: Request, call_next: Any) -> Any:
    """统一安全门：Bearer 鉴权（可选）+ 单 IP 滑动窗口限流（可选）。

    - API_TOKEN 为空时鉴权关闭（本地演示默认）；非空时必须带
      `Authorization: Bearer <token>`；
    - RATE_LIMIT_PER_MIN=0 时限流关闭；>0 时按客户端 IP 每分钟计数，
      超限返回 429；
    - 配置均在每次请求时动态读取，改 .env 后无需重启即可生效。
    """
    path: str = request.url.path
    if path in _PUBLIC_PATHS:
        return await call_next(request)

    token: str = config_module.CONFIG.api_token
    if token:
        auth: str = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:].strip() != token:
            _metric_observe(request.method, path, 401, 0.0, count_inflight=False)
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    limit: int = config_module.CONFIG.rate_limit_per_min
    if limit > 0:
        client: str = request.client.host if request.client else "unknown"
        now: float = time.monotonic()
        with _rate_lock:
            window: deque[float] = _rate_windows.setdefault(client, deque())
            while window and now - window[0] >= 60.0:
                window.popleft()
            if len(window) >= limit:
                _metric_observe(request.method, path, 429, 0.0, count_inflight=False)
                return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
            window.append(now)

    _inflight_add(1)
    t0: float = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _metric_observe(request.method, path, 500, time.perf_counter() - t0)
        raise
    _metric_observe(request.method, path, response.status_code, time.perf_counter() - t0)
    return response


class AnalyzeRequest(BaseModel):
    """POST /analyze 请求体。"""

    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, description="股票代码或行业关键词")
    thread_id: str | None = Field(default=None, description="断点续跑线程 ID（可选）")


@app.get("/health")
def health() -> dict[str, Any]:
    """存活检查：返回服务状态、当前模式与工具缓存命中统计。"""
    return {
        "status": "ok",
        "mock_mode": config_module.CONFIG.mock_mode,
        "auth_enabled": bool(config_module.CONFIG.api_token),
        "rate_limit_per_min": config_module.CONFIG.rate_limit_per_min,
        "pipeline_deadline_s": config_module.CONFIG.pipeline_deadline_s,
        "tool_cache": tool_cache_stats(),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus 文本格式指标（进程内累计，重启归零）。

    覆盖：HTTP 请求量/延迟/进行中请求数、工具缓存命中与容量、当前模式。
    指标全部来自真实请求与真实工具调用计数，不做任何估算。
    """
    lines: list[str] = [
        "# HELP agentcouncil_http_requests_total 累计 HTTP 请求数（按方法/路径/状态码）",
        "# TYPE agentcouncil_http_requests_total counter",
    ]
    with _metric_lock:
        for (method, path, status), count in sorted(_request_counts.items()):
            lines.append(
                f'agentcouncil_http_requests_total{{method="{method}",path="{path}",'
                f'status="{status}"}} {count}'
            )
        lines += [
            "# HELP agentcouncil_http_request_duration_seconds HTTP 请求耗时（秒，累计）",
            "# TYPE agentcouncil_http_request_duration_seconds summary",
        ]
        for path in sorted(_request_seconds):
            lines.append(
                f'agentcouncil_http_request_duration_seconds_sum{{path="{path}"}} '
                f"{_request_seconds[path]:.6f}"
            )
            lines.append(
                f'agentcouncil_http_request_duration_seconds_count{{path="{path}"}} '
                f"{_request_seconds_count.get(path, 0)}"
            )
        lines += [
            "# HELP agentcouncil_http_inflight_requests 当前进行中的请求数",
            "# TYPE agentcouncil_http_inflight_requests gauge",
            f"agentcouncil_http_inflight_requests {_inflight}",
        ]

    lines += [
        "# HELP agentcouncil_tool_cache_hits_total 工具缓存命中次数",
        "# TYPE agentcouncil_tool_cache_hits_total counter",
        "# HELP agentcouncil_tool_cache_misses_total 工具缓存未命中次数",
        "# TYPE agentcouncil_tool_cache_misses_total counter",
        "# HELP agentcouncil_tool_cache_entries 工具缓存当前条目数",
        "# TYPE agentcouncil_tool_cache_entries gauge",
    ]
    for tool, stat in sorted(tool_cache_stats().items()):
        lines.append(f'agentcouncil_tool_cache_hits_total{{tool="{tool}"}} {stat.get("hits", 0)}')
        lines.append(f'agentcouncil_tool_cache_misses_total{{tool="{tool}"}} {stat.get("misses", 0)}')
        lines.append(f'agentcouncil_tool_cache_entries{{tool="{tool}"}} {stat.get("entries", 0)}')

    lines += [
        "# HELP agentcouncil_mock_mode 当前是否 mock 模式（1=mock）",
        "# TYPE agentcouncil_mock_mode gauge",
        f"agentcouncil_mock_mode {1 if config_module.CONFIG.mock_mode else 0}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


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
    """非流式分析：直接返回最终决策 JSON（同步端点，FastAPI 自动线程池执行）。

    断点续跑：同一 thread_id 若已有 final_decision，直接返回既有结果，
    不重复执行整条图（避免重复计费与重复等待）。
    """
    query: str = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    tid: str = payload.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    graph = build_council_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}
    snapshot = graph.get_state(config)
    existing = snapshot.values.get("final_decision") if snapshot and snapshot.values else None
    if existing is not None:
        return _dump(existing)
    final_state: dict[str, Any] = graph.invoke({"query": query}, config=config)
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