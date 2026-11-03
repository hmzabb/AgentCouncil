# api/ 服务层（预留）

stub 阶段无对外服务。后续阶段规划：

- FastAPI 接口：`POST /council/run`，请求体 `{topic}` → 返回 final_decision JSON
- 复用 `graph.build_council_graph()`，与 CLI 共用同一条链路
