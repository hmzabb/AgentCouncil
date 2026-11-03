"""工具基础设施：schema 约定 + fail-open 错误处理。

设计原则（CLAUDE.md）：
- 任何工具失败（参数校验、实现异常、执行超时）都不得使链路崩溃；
- 统一包装为 ToolResult，下游按 ok/error 字段降级处理并留痕。

超时看门狗（2026-11-03 线上故障后补）：fail-open 只能兜住「抛异常」，
兜不住「永不返回」——Milvus 集合卡在 Loading 时 load_collection 会无限
阻塞，导致并行分支永不返回、图级 fan-in 永久等待、前端进度条卡死。
故 invoke 把 _execute 放到守护线程执行并 join(tool_timeout_s)：超时即
返回 ToolResult(ok=False)，超时线程被放弃等待（daemon，随进程退出）。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

import config as config_module

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """工具调用统一返回包装：成功与否都返回本结构（fail-open）。"""

    model_config = {"extra": "forbid"}

    tool: str = Field(description="工具名称")
    ok: bool = Field(description="调用是否成功")
    data: dict[str, Any] | None = Field(
        default=None, description="成功时的结构化输出（output_schema.model_dump()）"
    )
    error: str | None = Field(default=None, description="失败原因；ok=False 时非空")


class BaseTool(ABC):
    """工具基类。

    约定：
        Args:   输入 schema（Pydantic v2 模型），invoke 时严格校验；
        Result: 输出 schema（Pydantic v2 模型），_execute 必须返回其实例。
    后续接入真实数据源时，只需替换 _execute 实现，schema 与调用方式不变。
    """

    name: str = ""
    Args: type[BaseModel]
    Result: type[BaseModel]

    @abstractmethod
    def _execute(self, args: BaseModel) -> BaseModel:
        """工具实现（stub 阶段返回假数据）。任何异常由 invoke 兜底。"""

    def invoke(self, **kwargs: Any) -> ToolResult:
        """fail-open 调用入口：入参校验 / 实现异常 / 执行超时一律降级为 ok=False。"""
        # 1) 输入校验（Pydantic v2 严格模式：类型不符/多余字段直接失败）
        try:
            args: BaseModel = self.Args.model_validate(kwargs, strict=True)
        except Exception as exc:  # noqa: BLE001 —— fail-open：校验失败同样不抛出
            return ToolResult(tool=self.name, ok=False, error=f"参数校验失败: {exc}")
        # 2) 执行（守护线程 + 硬超时：阻塞式实现不得拖死上游图）
        timeout_s: int = config_module.CONFIG.tool_timeout_s
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["result"] = self._execute(args)
            except Exception as exc:  # noqa: BLE001 —— 异常带回主线程统一降级
                box["error"] = exc

        worker = threading.Thread(target=_run, daemon=True, name=f"tool-{self.name}")
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            logger.warning("工具 [%s] 执行超时（>%ss，fail-open 降级）：阻塞线程已放弃等待", self.name, timeout_s)
            return ToolResult(
                tool=self.name,
                ok=False,
                error=f"执行超时：超过 {timeout_s}s 未返回（已 fail-open 降级，不阻塞后续链路）",
            )
        if "error" in box:
            exc = box["error"]
            logger.warning("工具 [%s] 执行失败（fail-open 降级）：%s", self.name, exc)
            return ToolResult(tool=self.name, ok=False, error=f"{type(exc).__name__}: {exc}")
        result: BaseModel = box["result"]
        # 3) 输出 schema 校验（实现返回错误类型同样 fail-open）
        if not isinstance(result, self.Result):
            return ToolResult(
                tool=self.name,
                ok=False,
                error=f"输出类型不符: 期望 {self.Result.__name__}，实际 {type(result).__name__}",
            )
        return ToolResult(tool=self.name, ok=True, data=result.model_dump())