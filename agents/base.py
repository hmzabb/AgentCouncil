"""Agent 基类：统一执行骨架与 fail-open 降级约定。

说明：为类型标注引入 graph.state 的依赖（graph 不反向依赖 agents 的实现，
无循环导入）；Agent 只读取 state 中自己需要的字段。
报告类 Agent 输出继承 BaseAgentOutput；辩论轮次等中间产物为普通 BaseModel。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from graph.state import CouncilState

logger = logging.getLogger(__name__)

TOutput = TypeVar("TOutput", bound=BaseModel)


class BaseAgent(ABC, Generic[TOutput]):
    """所有 Agent 的基类。

    约定（CLAUDE.md「工具 fail-open」精神同样适用于 Agent）：
    - execute 实现主逻辑（stub 阶段为占位逻辑，后续替换为真实模型调用）；
    - run 是统一入口，捕获 execute 的任何异常并降级为 fallback 的失败态输出，
      单个 Agent 失败不允许导致整条 graph 崩溃。
    """

    name: str = "base_agent"

    @abstractmethod
    def execute(self, state: CouncilState) -> TOutput:
        """Agent 主逻辑。读取 state 中需要的字段，返回受 schema 约束的输出。"""

    @abstractmethod
    def fallback(self, error: Exception) -> TOutput:
        """失败降级输出：status=failed 的合法结构，供下游继续消费与留痕。"""

    def run(self, state: CouncilState) -> TOutput:
        """统一执行入口（fail-open）：任何异常都转为失败态输出，绝不向上抛。"""
        try:
            return self.execute(state)
        except Exception as exc:  # noqa: BLE001 —— fail-open：单 Agent 失败不崩溃
            logger.warning("Agent [%s] 执行失败（fail-open 降级）：%s", self.name, exc)
            return self.fallback(exc)
