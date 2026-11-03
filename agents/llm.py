"""DeepSeek LLM 封装：OpenAI 兼容客户端 + LangChain prompt 模板 + 结构化输出。

约定（CLAUDE.md）：
- 客户端走 LangChain ChatOpenAI（OpenAI 兼容），base_url / api_key / model
  由 config 的 DEEPSEEK_* 控制（.env 配置）；
- 结构化输出：DeepSeek 仅支持 response_format=json_object（不支持
  json_schema），因此手动绑定 json_object 模式 + prompt 内说明字段，
  返回后解析 JSON 再由 Pydantic 严格校验；
- 实测坑（2026-11-02）：DeepSeek 偶发把 response_format 的
  {"type": "json_object"} 抄进输出 JSON，导致 extra=forbid 拒绝。
  处理：剔除 schema 未声明的附加字段后再校验（只放宽模型产物噪声，
  内容字段仍严格校验，不含法即整体失败）；
- fail-open：未配置 key / 网络失败 / 输出不合法 → 抛异常，由各 Agent
  降级为确定性模板 + 错误留痕，绝不伪造 LLM 产出；
- 温度固定 0.2：报告类任务求稳定可复现。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

import config as config_module  # 动态读取配置（测试期 reload_config 切换场景）
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TOutput = TypeVar("TOutput", bound=BaseModel)

_LLM_TEMPERATURE: float = 0.2

# 公共系统提示（所有 Agent 共用）：json_object 模式要求 prompt 中出现 json 字样
SYSTEM_ROLE_HINT: str = (
    "你是 AgentCouncil 金融分析系统中的一名专业金融从业者。"
    "输出必须为合法 JSON 对象：只输出 JSON 本身，不要输出任何多余文字、"
    "markdown 代码块或注释。"
)


def get_chat_model() -> Any:
    """构建 DeepSeek 对话模型（每次新建：动态读取配置，支持 reload_config）。"""
    from langchain_openai import ChatOpenAI  # 延迟导入：真实模式才需要

    if not config_module.CONFIG.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY（.env），LLM 不可用")
    return ChatOpenAI(
        model=config_module.CONFIG.deepseek_model,
        api_key=config_module.CONFIG.deepseek_api_key,
        base_url=config_module.CONFIG.deepseek_base_url,
        temperature=_LLM_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


def _parse_json_content(content: Any) -> dict[str, Any]:
    """从模型消息内容中解析 JSON 对象（容错：剥 markdown 围栏 / 非对象报错）。"""
    text: str = content if isinstance(content, str) else str(content)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    data: Any = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"模型输出不是 JSON 对象: {type(data).__name__}")
    return data


def structured_call(schema_cls: type[TOutput], system: str, human: str) -> TOutput:
    """单次结构化调用：system/human 提示词 → json_object 输出 → Pydantic 严格校验。

    参数：
        schema_cls: 输出 schema（Pydantic v2，extra=forbid）。
        system:     系统提示词（须含角色 + 输出字段说明）。
        human:      本轮上下文（工具数据 / 上游报告等）。

    任何失败（未配置 key / 网络 / JSON 非法 / 字段校验不过）都向上抛异常，
    由 Agent 层 fail-open 降级（见各 Agent 的模板回退路径）。
    """
    model = get_chat_model().bind(response_format={"type": "json_object"})
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])
    chain = prompt | model
    message: AIMessage = chain.invoke({"input": human})
    data: dict[str, Any] = _parse_json_content(message.content)
    # 模型偶发带回 {"type": "json_object"} 等无害附加字段：剔除未声明字段后
    # 再严格校验（内容字段仍受 extra=forbid 与全部约束保护）
    clean: dict[str, Any] = {k: v for k, v in data.items() if k in schema_cls.model_fields}
    return schema_cls.model_validate(clean)
