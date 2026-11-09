"""pytest 统一入口：把各评测脚本的 mock 场景包装为 CI 可收集的用例。

为什么需要这一层：
- eval/*.py 是可独立运行的评测脚本（`python -m eval.smoke_test` 等），
  场景函数按 `scenario_*` 命名；pytest 默认只收集 `test_*`，
  直接跑 pytest 会「0 tests collected」——CI 显示通过但实际什么都没验证（假绿）；
- 本文件只包装 mock 模式场景（确定性、零成本、不依赖 AkShare/Milvus/DeepSeek），
  真实链路测试（real_data_test / llm_test / golden_test）不进入 CI，按需手动运行。

运行：
    pytest                       # 全部 mock 用例
    pytest -m mock               # 仅 mock 标记
    pytest -m "not slow"         # 跳过慢用例
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# mock 链路必须在导入 config 之前设置（config 导入时读取 .env）
os.environ["MOCK_MODE"] = "true"

from eval import ab_test, api_test, smoke_test  # noqa: E402 —— 复用既有场景实现

# ---------------------------------------------------------------- 冒烟链路


@pytest.mark.mock
def test_smoke_normal_run() -> None:
    """Send 并行 + 辩论循环 + 合法最终决策。"""
    smoke_test.scenario_1_normal_run()


@pytest.mark.mock
def test_smoke_schema_and_real_tools() -> None:
    """Pydantic 约束真实拒绝非法值；calculator/get_date 真实运行。"""
    smoke_test.scenario_2_schema_and_real_tools()


@pytest.mark.mock
def test_smoke_fail_open() -> None:
    """工具故障 → 留痕 → 评级降为减持，链路不中断。"""
    smoke_test.scenario_3_fail_open()


@pytest.mark.mock
def test_smoke_autonomy_and_veto() -> None:
    """独立多空 Agent、工具自采白名单、风控 V1/V2 一票否决。"""
    smoke_test.scenario_4_autonomy_and_veto()


@pytest.mark.mock
def test_smoke_quant_risk_rules() -> None:
    """量化指标真实计算 + 风控 V3 高波动否决 + ATR 放大 / 量价背离规则。"""
    smoke_test.scenario_5_quant_risk_rules()


# ---------------------------------------------------------------- 服务层


@pytest.mark.mock
def test_api_health_analyze_metrics() -> None:
    """/health、POST /analyze、/metrics 真实计数。"""
    api_test.scenario_1_health_and_analyze()


@pytest.mark.mock
def test_api_sse_event_sequence() -> None:
    """SSE 事件序列完整且辩论轮数不死循环。"""
    api_test.scenario_2_sse_event_sequence()


@pytest.mark.mock
def test_api_sse_degradation() -> None:
    """工具故障时 SSE 降级：error 留痕 + 仍产出最终决策。"""
    api_test.scenario_3_sse_degrade_on_failure()


# ---------------------------------------------------------------- A/B 统计框架
# 纯函数单元测试：只验证统计实现正确性，不产生任何「评测结论」。


@pytest.mark.mock
def test_mcnemar_exact_values() -> None:
    """McNemar 精确检验的二项双侧 p 值（手算值对照）。"""
    # n=5，全部只有 A 正确：p = 2 * C(5,0)/2^5 = 0.0625
    assert ab_test._binom_two_sided(0, 5) == pytest.approx(0.0625)
    # n=6，全部只有 A 正确：p = 2 * C(6,0)/2^6 = 0.03125 < 0.05
    assert ab_test._binom_two_sided(0, 6) == pytest.approx(0.03125)
    # n=6，只有 1 次偏向 B：p = (C(6,0)+C(6,1)+C(6,5)+C(6,6))/2^6 = 14/64
    assert ab_test._binom_two_sided(1, 6) == pytest.approx(0.21875)
    # 空样本：无信息，p 必须为 1.0（不能误报显著）
    assert ab_test._binom_two_sided(0, 0) == 1.0


@pytest.mark.mock
def test_mcnemar_significance_flag() -> None:
    """显著/不显著分支必须与 p 值一致。"""
    strong = ab_test.mcnemar_exact([(True, False)] * 8)
    assert strong["a_only_correct"] == 8 and strong["b_only_correct"] == 0
    assert strong["p_value"] < 0.05 and strong["significant_at_0.05"] is True
    weak = ab_test.mcnemar_exact([(True, False), (False, True)] * 2)
    assert weak["p_value"] > 0.05 and weak["significant_at_0.05"] is False


@pytest.mark.mock
def test_bootstrap_ci_behavior() -> None:
    """bootstrap 置信区间：恒定差值不跨 0；正负对称差值跨 0。"""
    const = ab_test.bootstrap_ci([2.0] * 10)
    assert const["mean_diff"] == 2.0
    assert const["ci95"] == [2.0, 2.0] and const["crosses_zero"] is False
    mixed = ab_test.bootstrap_ci([-1.0, 1.0] * 5)
    assert mixed["crosses_zero"] is True
    empty = ab_test.bootstrap_ci([])
    assert empty["n"] == 0 and empty["crosses_zero"] is True


@pytest.mark.mock
def test_parse_overrides_rejects_bad_format() -> None:
    """配置覆盖格式非法必须报错（避免静默跑错配置）。"""
    assert ab_test._parse_overrides("a=1, b=2") == {"a": "1", "b": "2"}
    with pytest.raises(ValueError):
        ab_test._parse_overrides("debate_max_rounds")


@pytest.mark.mock
def test_config_float_bounds_and_fail_open() -> None:
    """比例型配置越界必须钳制、非法值回退默认（否则规则会静默失效）。"""
    import config as config_module

    def read(name: str, raw: str | None, **kwargs: object) -> float:
        if raw is not None:
            os.environ[name] = raw
        try:
            return config_module._get_float(name, 0.8, **kwargs)  # type: ignore[arg-type]
        finally:
            os.environ.pop(name, None)

    assert read("X_TEST", "1.5", minimum=0.0, maximum=1.0) == 1.0, "超上限应钳制到 1.0"
    assert read("X_TEST", "-0.2", minimum=0.0, maximum=1.0) == 0.0, "超下限应钳制到 0.0"
    assert read("X_TEST", "abc", minimum=0.0, maximum=1.0) == 0.8, "非法值应回退默认"
    assert read("X_TEST", None, minimum=0.0, maximum=1.0) == 0.8, "未配置应取默认"
    # 量比阈值不设上限：>1 是合法口径（低于略高于均量的水平即视为缩量）
    assert read("X_TEST", "1.1", minimum=0.0) == 1.1