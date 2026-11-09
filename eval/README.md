# eval/ 评测目录

当前提供四套测试 + 一套真实评测流水线（所有断言与数字均基于真实运行结果，
禁止编造）：

**真实评测流水线（决策闭环的量化验收）**：
1. `python -m eval.run_eval --runs 5`——同 ticker 串行运行 N 次真实链路
   （AkShare + Milvus + DeepSeek），每次运行产物完整存档到 `runs/run_NNN.json`
   （三分析师报告 / 逐轮辩论 / 风控 / 最终决策 / 分超步计时），并统计：
   评级分布与一致率、置信度均值/标准差、端到端 P95/均值、分阶段耗时、
   引用来源比例等确定性指标（`runs/metrics.json`）；
2. `python -m eval.judge`——LLM-as-Judge（DeepSeek 评审，prompt 独立于
   生产链路）：报告质量（事实性/相关性/无幻觉 0~5）+ 论点级有据判定
   （幻觉率=无据论点占比，评审基准为「分析师收到的同一份工具数据」）；
   辩论有效性（多方交锋/空方交锋/对立覆盖 0~5，考察是否真覆盖对立观点
   而非一边倒）。结果存 `runs/judge_results.json`；
3. `python -m eval.render_report`——把每次运行渲染为 Markdown + 自包含
   HTML（`reports/run_NNN.md|html`，含各角色报告、辩论摘要、风控、最终
   决策与评审分数），并渲染 `reports/SUMMARY.md|html` 评测汇总。
4. `python -m eval.ab_test --repeats 3 --a debate_max_rounds=1 --b debate_max_rounds=3`——
   A/B 对比框架（配对采样 + 显著性统计）：同一批标的在配置 A / B 下各重复
   N 次（配对键 = 标的 + 重复序号），输出方向匹配率（McNemar 精确检验，
   手写二项双侧 p 值，无 scipy 依赖）、延迟与辩论轮次的配对差值
   bootstrap 95% 置信区间（固定种子可复现）、同配置重复运行的评级一致率。
   默认 mock 链路（零成本，用于验证统计框架与 CI 回归）；`--real` 切换真实
   链路得出质量结论。结果存 `runs/ab_comparison.json`（含原始配对样本）。
5. `python -m eval.golden_test`——黄金集基准：5 只真实标的（券商研报共识，
   `golden_set.json` 逐条标注来源与日期）对比评级精确匹配率、平均距离、
   方向一致率、风险等级准确率与系统性偏差（真实链路，单次/标的）。

- **smoke_test.py**：mock 链路冒烟测试（固定 MOCK_MODE=true，确定性、不依赖外部服务）。
  5 场景：正常链路（Send 并行 + 辩论循环 + 评级）、Pydantic 约束与真实工具、
  fail-open 降级（工具故障 → 留痕 → 评级降为减持）、自主性与风控否决
  （独立多空 Agent 产出互不相同、分析师工具自采白名单裁剪、风控 V1/V2
  一票否决真实触发且多头排列对照组不误杀）、量化风控（构造日线序列真实计算
  波动率/最大回撤/ATR/量比/区间位置，V3 高波动否决真实触发，ATR 放大与
  量价背离规则按配置阈值触发，并含放量对照组防误杀）；
- **real_data_test.py**：真实数据链路测试（固定 MOCK_MODE=false，依赖
  docker milvus-standalone + 已建库 + 网络）。
  6 场景：Milvus RAG 真实检索、AkShare 行情（指标与独立重算交叉验证）、
  技术分析师单 Agent 真实跑通、全链路真实模式（三角色分析师引用真实来源）、
  财务 + 新闻（独立重取交叉验证）、真实工具 fail-open（非法代码 / 集合缺失 /
  演示超时均 ok=False 留痕）；
- **llm_test.py**：DeepSeek LLM 接入测试（固定 MOCK_MODE=false，需配置
  DEEPSEEK_API_KEY）。4 场景：分析师 / 研究员 / 风控 / 经理的真实 API 调用
  与结构化输出验证；
- **parallelism_test.py**：并行 vs 串行端到端耗时对比（真实模式、真实计时）。
  Send 图级并行（主链路默认）vs 单节点 for 循环串行
  （build_council_graph(serial_analysts=True)），输出分超步耗时 + 加速比；
- **api_test.py**：FastAPI + SSE 服务测试（固定 MOCK_MODE=true）。
  3 场景：/health 与 POST /analyze（含 `/metrics` 指标真实计数校验）、
  SSE 事件序列完整性（三分析师逐份 + 辩论轮数严格等于上限防死循环 +
  事件顺序符合链路拓扑）、工具故障时流的自动降级（error 留痕 + 最终
  决策仍产出且评级降为减持）；
- **test_suites.py**：pytest 统一入口（CI 使用）。把上面 mock 场景
  （smoke 5 个 + api 3 个）包装为 `test_*` 用例，另加 A/B 统计框架的纯函数
  单元测试（McNemar 精确 p 值手算对照、bootstrap CI 跨 0 判定、配置覆盖
  格式校验、配置数值上下限钳制与 fail-open 回退），共 13 个用例，
  全部确定性、零外部依赖：`pytest -m mock`。真实链路脚本
  （real/llm/golden）不进 CI。

目录：
- `runs/`：评测运行产物（run_NNN.json / metrics.json / judge_results.json）；
- `reports/`：渲染出的评测报告（run_NNN.md|html / SUMMARY.md|html）。

可选后续（当前六项指标已覆盖决策一致性/报告质量/辩论有效性/幻觉率/延迟/
报告渲染）：RAGAS 检索质量指标（context precision/recall）、黄金集
决策基准对比、置信度校准曲线。