"""进程内 TTL 缓存：工具与 LLM 结果的复用层，降低重复外部请求与调用成本。

约定：
- 线程安全：SSE 工作线程、FastAPI 线程池与主线程会并发访问同一缓存；
- ttl_s <= 0 表示禁用缓存（直接透传，不写入）；
- 只缓存成功结果：失败（ok=False / 异常）不写入，避免把瞬时故障固化成缓存；
- 键由「命名空间 + 入参」确定性序列化生成，保证同参数命中同一键；
- 容量上限采用近似 LRU：超限时优先清理过期项，仍超限则淘汰最早写入项。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class TTLCache:
    """带 TTL 与容量上限的线程安全缓存。"""

    def __init__(self, ttl_s: float, max_entries: int = 512) -> None:
        """初始化缓存。

        参数：
            ttl_s:       条目存活时间（秒）；<=0 表示禁用。
            max_entries: 条目数上限，超限触发淘汰。
        """
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    @property
    def enabled(self) -> bool:
        """缓存是否启用（ttl_s <= 0 时禁用）。"""
        return self.ttl_s > 0

    def get(self, key: str) -> tuple[bool, Any]:
        """读取缓存：返回 (是否命中, 值)。过期条目视为未命中并清除。"""
        if not self.enabled:
            return False, None
        now: float = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return False, None
            expire_at, value = entry
            if now >= expire_at:
                self._store.pop(key, None)
                self.misses += 1
                return False, None
            self.hits += 1
            return True, value

    def set(self, key: str, value: Any) -> None:
        """写入缓存（禁用时直接返回）。"""
        if not self.enabled:
            return
        now: float = time.monotonic()
        with self._lock:
            if len(self._store) >= self.max_entries:
                self._evict(now)
            self._store[key] = (now + self.ttl_s, value)

    def _evict(self, now: float) -> None:
        """淘汰：先清过期项，仍满则丢弃最早写入的条目（近似 LRU）。"""
        expired = [k for k, (expire_at, _) in self._store.items() if now >= expire_at]
        for k in expired:
            self._store.pop(k, None)
        if len(self._store) >= self.max_entries and self._store:
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest, None)

    def clear(self) -> None:
        """清空缓存并重置命中统计。"""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        """返回命中统计（用于可观测性端点）。"""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """由命名空间与入参构造稳定缓存键（键序无关的 JSON 序列化 + sha256）。"""
    canonical: str = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest: str = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}:{digest}"