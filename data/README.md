# data/ 数据层

- **raw/**：研报/财报样本 PDF（data/ingest.py 自动下载交易所披露公开研报；
  下载明细见 raw/sources.json）。大文件不入版本库；
- **embedder.py**：bge 嵌入模型封装（本地加载，全局单例，向量归一化）；
- **ingest.py**：建库脚本——PDF → 切分（400 字符 / 重叠 60）→ bge 向量化 →
  写入 Milvus 集合 `research_docs`（doc_id / chunk_text / embedding / metadata，
  HNSW + COSINE 索引，选型理由见脚本 docstring）。

用法：`python data/ingest.py`（需 docker milvus-standalone 已启动）。

实测坑与恢复（2026-11-02）：
- **容器重启后集合不会自动加载**：重启 Milvus 后直接检索会报 503
  （channel not available）。research_rag 已内置自愈（检索前检查 load
  state 并重载），无需手动干预；
- **元数据漂移 / 段文件缺失**：若检索持续失败且报 `invalid local path
  ... meta.json`，说明集合元数据与当前磁盘布局不一致（容器历史状态
  漂移），用 `python data/ingest.py --rebuild --skip-download` 重建集合即可
  （PDF 已下载时跳过下载，约 5 分钟，含 bge 向量化）。

后续：黄金集 / 评测集 / 链路运行产物缓存。
