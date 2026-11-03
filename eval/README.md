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

- **smoke_test.py**：mock 链路冒烟测试（固定 MOCK_MODE=true，确定性、不依赖外部服务）。
  3 场景：正常链路（Send 并行 + 辩论循环 + 评级）、Pydantic 约束与真实工具、
  fail-open 降级（工具故障 → 留痕 → 评级降为减持）；
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
  3 场景：/health 与 POST /analyze、SSE 事件序列完整性（三分析师逐份 +
  辩论轮数严格等于上限防死循环 + 事件顺序符合链路拓扑）、工具故障时
  流的自动降级（error 留痕 + 最终决策仍产出且评级降为减持）。

目录：
- `runs/`：评测运行产物（run_NNN.json / metrics.json / judge_results.json）；
- `reports/`：渲染出的评测报告（run_NNN.md|html / SUMMARY.md|html）。

可选后续（当前六项指标已覆盖决策一致性/报告质量/辩论有效性/幻觉率/延迟/
报告渲染）：RAGAS 检索质量指标（context precision/recall）、黄金集
决策基准对比、置信度校准曲线。
