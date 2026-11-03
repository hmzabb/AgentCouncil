"""全局配置：从环境变量读取，带默认值与容错解析。

数据层已全部接入真实数据源（AkShare 行情/财务/新闻 / Milvus 向量库 /
bge 嵌入），mock_mode=False（默认）时工具走真实实现；真实工具失败
（接口异常 / 字段缺失）返回 ToolResult(ok=False) 留痕（fail-open），
绝不把假数据伪装成真实数据。

约定（CLAUDE.md）：
- 配置缺失或非法时不抛异常，回退到默认值（fail-open）。
- .env 由本模块统一加载，保证任何入口导入本模块即生效。
- 各模块须以 `import config as config_module` 动态读取配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 加载 .env（不存在则静默跳过）；须放在读取环境变量之前
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    """读取布尔型环境变量；非法值回退默认值（fail-open）。"""
    raw: str | None = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, minimum: int | None = None) -> int:
    """读取整型环境变量；非法值回退默认值，并受下限约束（fail-open）。"""
    raw: str | None = os.getenv(name)
    if raw is None:
        return default
    try:
        value: int = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(value, minimum)
    return value


@dataclass(frozen=True)
class CouncilConfig:
    """AgentCouncil 运行配置。

    字段：
        mock_mode:            True 时工具使用占位实现（冒烟测试强制 True）；
                              False（默认）时行情/检索走真实数据源。
        debate_max_rounds:    多空辩论最大轮数（≥1，防死循环）。
        demo_tool_failure:    True 时行情工具故意失败，用于演示 fail-open 降级。
        milvus_*:             Milvus 本地库连接（docker milvus-standalone，19530）。
        bge_model:            bge 嵌入模型名（本地加载，CLAUDE.md：bge-large-zh / bge-m3）。
        hf_endpoint:          HuggingFace 下载端点（国内镜像 hf-mirror.com）。
        hf_home:              模型缓存目录（权重约 1.3GB，放数据盘，勿放 C 盘）。
        market_history_days:  行情历史回看窗口（自然日），用于 MA/RSI/MACD 计算。
        tool_timeout_s:       单次工具调用硬超时（秒）；超时即 fail-open 降级，
                              防止阻塞式实现（Milvus RPC / 第三方 HTTP）把整条图卡死。
        deepseek_*:           DeepSeek 接入预留字段（OpenAI 兼容，尚未使用）。
    """

    mock_mode: bool = False
    debate_max_rounds: int = 3
    demo_tool_failure: bool = False

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "research_docs"
    bge_model: str = "BAAI/bge-large-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_home: str = "D:/Python/hf-cache"
    market_history_days: int = 400
    tool_timeout_s: int = 180

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    @classmethod
    def from_env(cls) -> "CouncilConfig":
        """从环境变量构建配置（.env 已由本模块在导入时加载）。"""
        return cls(
            mock_mode=_get_bool("MOCK_MODE", False),
            debate_max_rounds=_get_int("DEBATE_MAX_ROUNDS", 3, minimum=1),
            demo_tool_failure=_get_bool("DEMO_TOOL_FAILURE", False),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=_get_int("MILVUS_PORT", 19530, minimum=1),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "research_docs"),
            bge_model=os.getenv("BGE_MODEL", "BAAI/bge-large-zh-v1.5"),
            hf_endpoint=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
            hf_home=os.getenv("HF_HOME", "D:/Python/hf-cache"),
            market_history_days=_get_int("MARKET_HISTORY_DAYS", 400, minimum=60),
            tool_timeout_s=_get_int("TOOL_TIMEOUT_S", 180, minimum=10),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )


# 模块级全局配置。
# 注意：各模块须以 `import config as config_module` 方式动态读取
# `config_module.CONFIG`，不要 `from config import CONFIG`（后者绑定旧引用，
# 无法感知 reload_config 的重载，冒烟测试依赖这一机制切换 mock 场景）。
CONFIG: CouncilConfig = CouncilConfig.from_env()


def reload_config() -> CouncilConfig:
    """重新读取环境变量并刷新全局配置（主要用于测试/演示场景切换）。"""
    global CONFIG
    CONFIG = CouncilConfig.from_env()
    return CONFIG