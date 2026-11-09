"""bge 嵌入模型封装：本地加载 + 全局单例 + 查询/文档编码。

约定：
- 模型名与维度由 config 控制（BGE_MODEL，默认 BAAI/bge-large-zh-v1.5，
  亦可用 BAAI/bge-m3，见 CLAUDE.md 技术栈）；
- 权重首次加载时从 HF_ENDPOINT（默认国内镜像 hf-mirror.com）下载到本地
  HF 缓存目录，之后离线可用；
- 所有向量统一 L2 归一化（bge 官方检索约定），与 Milvus COSINE 度量匹配；
- bge-*-v1.5 / bge-m3 检索时无需额外 query 指令前缀（与 bge v1 不同），
  故这里不拼接指令；
- 模型加载失败会抛异常，由调用方（工具 / ingest）按 fail-open 约定兜底，
  绝不静默降级为假向量。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import config as config_module

logger = logging.getLogger(__name__)

# 全局单例 + 锁：避免图内多个工具/节点重复加载上 GB 的大模型
_embedder: BgeEmbedder | None = None
_lock: threading.Lock = threading.Lock()


class BgeEmbedder:
    """bge 系列嵌入模型的本地封装（sentence-transformers 加载，CPU 推理）。"""

    def __init__(self, model_name: str) -> None:
        # 镜像端点与缓存目录必须在首次下载前设置（HF 默认缓存会落到 C 盘用户目录）
        os.environ.setdefault("HF_ENDPOINT", config_module.CONFIG.hf_endpoint)
        os.environ.setdefault("HF_HOME", config_module.CONFIG.hf_home)
        from sentence_transformers import SentenceTransformer  # 延迟导入：包体量大，仅真实使用引入

        logger.info("加载嵌入模型 %s（首次运行需下载权重，请耐心等待）…", model_name)
        self._model: Any = SentenceTransformer(model_name, device="cpu")
        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("嵌入模型就绪：%s，向量维度 %d", model_name, self.dim)

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """批量编码文本，返回归一化向量列表（每个向量长度 = self.dim）。"""
        vectors: Any = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # bge 检索必须归一化，配合 COSINE 度量
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]


def get_embedder() -> BgeEmbedder:
    """获取全局单例 embedder（线程安全；首次调用触发模型下载/加载）。"""
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = BgeEmbedder(config_module.CONFIG.bge_model)
        return _embedder
