"""RAG 检索工具真实实现：Milvus（docker milvus-standalone）+ bge 向量检索。

设计：
- 连接走 pymilvus MilvusClient（uri = http://MILVUS_HOST:MILVUS_PORT），
  集合名 / top_k / 模型均来自 config；
- 查询向量由 data.embedder 单例加载的 bge 本地编码（归一化后与
  Milvus COSINE 度量匹配），检索参数 ef=64 保证召回；
- 自愈：Milvus 容器重启后集合 load 状态不持久（不会自动加载），
  检索前检查 load state，未加载则 load_collection + 轮询等待
  （实测坑，2026-11-02：docker 重启后直接 search 报 503
  "channel not available"）；
- 防卡死（实测坑，2026-11-03）：segment 加载失败时集合会永久停在
  Loading(50%)，此时 load_collection / get_load_state 无 timeout 会无限
  阻塞，导致并行分支永不返回、图级 fan-in 卡死、前端停在 2/3。故所有
  Milvus RPC 显式传 timeout，并对「加载进度停滞」快速失败；
- fail-open：集合不存在 / Milvus 未启动 / 集合加载失败 / 检索为空 → 抛异常，
  由 BaseTool.invoke 包装为 ToolResult(ok=False) 并留痕，绝不伪造检索结果；
- 返回 chunk 附带 metadata（ticker/date/source/page，入库时由
  data/ingest.py 写入），供下游引用来源、过程可追溯。

与 mock 版的关系：schema（ResearchRagArgs/RagChunk/ResearchRagResult）
在本模块定义，mock 版直接复用，保证两种模式字段对齐。
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

import config as config_module
from tools.base import BaseTool

# 单个 Milvus RPC 硬超时（秒）：集合卡在 Loading 时这些 RPC 会无限阻塞
_RPC_TIMEOUT_S: float = 10.0
# 集合重载等待上限（秒）：小库（数百~数千 chunk）数秒内即可加载完成
_LOAD_TIMEOUT_S: int = 120
_LOAD_POLL_INTERVAL_S: float = 2.0
# 加载进度连续停滞次数上限（3 × 2s = 6s 判定卡死），避免死等
_STALL_POLLS: int = 3


def _load_state(client: Any, collection: str) -> tuple[str, int]:
    """读取集合 load 状态：返回 (状态名, 加载进度百分比)。"""
    info: dict[str, Any] = client.get_load_state(
        collection_name=collection, timeout=_RPC_TIMEOUT_S
    )
    return str(info.get("state")).split(".")[-1], int(info.get("progress") or 0)


def _ensure_loaded(client: Any, collection: str) -> None:
    """集合 load 自愈：容器重启后 load 状态不持久，未加载则重载并等待。

    三重防卡死（缺一即可能永久阻塞）：
    1. 所有 RPC 带 timeout —— load_collection 无超时时会一直等下去；
    2. 轮询有总时限 —— 到点抛异常交给 invoke 降级；
    3. 进度停滞判定 —— segment 加载失败会让进度永久停在 50%，
       继续等待毫无意义，快速失败并提示重建集合。
    """
    state, progress = _load_state(client, collection)
    if state == "Loaded":
        return  # 已加载，直接检索
    if state != "Loading":
        # 仅 NotLoad/NotLoaded 才需要主动触发；已在 Loading 时重复触发无意义且会阻塞
        client.load_collection(collection_name=collection, timeout=_RPC_TIMEOUT_S)
    deadline: float = time.time() + _LOAD_TIMEOUT_S
    stalled: int = 0
    last_progress: int = progress
    while time.time() < deadline:
        state, progress = _load_state(client, collection)
        if state == "Loaded":
            return
        if progress == last_progress:
            stalled += 1
            if stalled >= _STALL_POLLS:
                raise RuntimeError(
                    f"集合 {collection} 加载卡死（进度停在 {progress}%，"
                    f"{int(_STALL_POLLS * _LOAD_POLL_INTERVAL_S)}s 无进展）：segment 加载失败，"
                    f"请查 Milvus 容器日志（常见为 files/json_stats 缓存缺失），"
                    f"并用 python data/ingest.py --rebuild --skip-download 重建集合"
                )
        else:
            stalled = 0
            last_progress = progress
        time.sleep(_LOAD_POLL_INTERVAL_S)
    raise RuntimeError(f"集合 {collection} 重载超时（{_LOAD_TIMEOUT_S}s），请稍后重试")

# ------------------------------------------------------------------ schema


class ResearchRagArgs(BaseModel):
    """研报/知识库检索入参。"""

    query: str = Field(min_length=1, description="检索问题")
    top_k: int = Field(default=5, ge=1, le=10, description="返回条数")


class RagChunk(BaseModel):
    """单条检索结果。"""

    chunk_text: str
    doc_name: str = Field(description="来源文档标识（doc_id，含文件名与切分序号）")
    score: float = Field(ge=0.0, le=1.0, description="相似度得分 [0,1]（COSINE）")
    metadata: dict[str, Any] | None = Field(
        default=None, description="入库元数据（ticker/date/source/page），真实模式必有"
    )


class ResearchRagResult(BaseModel):
    """检索结果。"""

    chunks: list[RagChunk]


# ------------------------------------------------------------------ 工具实现


def get_milvus_client() -> Any:
    """创建 Milvus 客户端（连接失败在此抛出，由 invoke 兜底为 ok=False）。"""
    from pymilvus import MilvusClient  # 延迟导入：仅真实模式使用

    uri = f"http://{config_module.CONFIG.milvus_host}:{config_module.CONFIG.milvus_port}"
    return MilvusClient(uri=uri, timeout=_RPC_TIMEOUT_S)


class ResearchRagTool(BaseTool):
    """研报检索工具（真实实现：Milvus 向量检索 + bge 查询编码）。"""

    name = "research_rag"
    Args = ResearchRagArgs
    Result = ResearchRagResult

    def _execute(self, args: ResearchRagArgs) -> ResearchRagResult:
        from data.embedder import get_embedder  # 延迟导入：模型加载重，检索时才需要

        collection = config_module.CONFIG.milvus_collection
        client = get_milvus_client()
        if not client.has_collection(collection, timeout=_RPC_TIMEOUT_S):
            raise RuntimeError(
                f"Milvus 集合 {collection} 不存在：请先运行 data/ingest.py 建库（确保 docker "
                f"milvus-standalone 已启动）"
            )
        _ensure_loaded(client, collection)  # 容器重启后集合不自载，此处自愈
        query_vec: list[float] = get_embedder().encode([args.query])[0]
        hits: list[dict[str, Any]] = client.search(
            collection_name=collection,
            data=[query_vec],
            limit=args.top_k,
            output_fields=["doc_id", "chunk_text", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            timeout=_RPC_TIMEOUT_S,
        )[0]
        chunks: list[RagChunk] = [
            RagChunk(
                chunk_text=str(hit["entity"]["chunk_text"]),
                doc_name=str(hit["entity"].get("doc_id", "")),
                # COSINE 距离 ∈ [0,1]；截断保护，防浮点边界越界
                score=round(max(0.0, min(1.0, float(hit["distance"]))), 4),
                metadata=hit["entity"].get("metadata"),
            )
            for hit in hits
        ]
        if not chunks:
            raise RuntimeError(f"集合 {collection} 检索为空：请确认 data/ingest.py 已写入数据")
        return ResearchRagResult(chunks=chunks)