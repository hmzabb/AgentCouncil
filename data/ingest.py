"""Milvus 研报知识库构建脚本：公开研报/财报 PDF → 切分 → bge 向量化 → 写入 Milvus。

用法（项目根目录，docker milvus-standalone 已启动）：
    python data/ingest.py                  # 下载样本 PDF（data/raw 为空时）+ 建库
    python data/ingest.py --skip-download  # 仅处理 data/raw 下已有 PDF
    python data/ingest.py --rebuild        # 先删除集合再重建（覆盖旧数据）

样本数据来源（真实下载，明细落盘 data/raw/sources.json）：
- 东方财富研报中心的公开卖方研报：AkShare stock_research_report_em
  接口，PDF 直链（pdf.dfcfw.com）；默认抓取 600519（贵州茅台）最新
  研报，与 demo 查询 600519.SH 呼应，元数据含机构 / 日期 / 评级；
- 接口不可用时：手动把公开研报/财报 PDF 放入 data/raw/ 后
  用 --skip-download 建库（来源信息从文件名解析）。

Milvus 集合 schema（字段与 CLAUDE.md 要求一致）：
    doc_id:     VARCHAR 主键（文件名#页码c序号）
    chunk_text: VARCHAR 切分文本
    embedding:  FLOAT_VECTOR（bge 维度，L2 归一化）
    metadata:   JSON（ticker / date / source / page）

索引选型（构建时打印）：
- 选用 HNSW（metric=COSINE，M=16，efConstruction=200，检索 ef=64）：
  * 本项目库规模小（数百~数千 chunk），HNSW 图索引无需训练、召回率高
    （小库 >95%）、查询毫秒级，是单机小库场景的最优解；
  * IVF_PQ 面向百万级向量：倒排聚类 + 乘积量化压缩内存，但量化有精度
    损失、需训练、nprobe 调参复杂，小库上引入其复杂度不划算 —— 不选；
  * FLAT 暴力全扫精度最高但查询线性扫描，作为演示 HNSW 更贴近生产选型。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

# 保证无论以 `python data/ingest.py` 还是 `-m data.ingest` 方式运行，
# 都能从项目根目录导入 config / tools 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as config_module
from tools.rag import get_milvus_client

logger = logging.getLogger(__name__)

RAW_DIR: Path = Path(__file__).resolve().parent / "raw"
SOURCES_JSON: Path = RAW_DIR / "sources.json"

# 切分参数：字符数口径（中文研报单字符约 0.5~1 token）
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 60

# 下载偏好：默认抓取与 demo 查询一致的贵州茅台（600519）研报
_DEFAULT_SYMBOL: str = "600519"

_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


# ------------------------------------------------------------------ PDF 下载（真实数据源）


def fetch_report_rows(symbol: str) -> list[dict[str, str]]:
    """从东方财富研报中心拉取公开研报列表（title/org/date/url，PDF 直链）。

    实测（2026-11-02，AkShare 1.20.22）：stock_research_report_em 返回
    中文列名「报告名称/机构/日期/报告PDF链接」，其中报告PDF链接为
    pdf.dfcfw.com 直链，可直接下载；symbol 传 6 位代码即可。
    """
    import akshare as ak

    df = ak.stock_research_report_em(symbol=symbol)
    if df is None or df.empty:
        raise RuntimeError(f"研报中心未返回 {symbol} 的研报列表")

    def find(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    url_col = find("报告PDF链接", "pdf", "链接")
    if url_col is None:
        raise RuntimeError(f"研报列表缺少 PDF 链接列，实际列: {list(df.columns)}")
    title_col = find("报告名称", "标题", "题目")
    org_col = find("机构")
    date_col = find("日期", "时间")
    rows: list[dict[str, str]] = []
    for _, r in df.iterrows():
        url = str(r[url_col])
        if not url.lower().startswith("http"):
            continue  # 个别行无 PDF 直链（如招股书类），跳过
        rows.append(
            {
                "title": str(r[title_col]) if title_col else "",
                "org": str(r[org_col]) if org_col else "",
                "date": str(r[date_col])[:10] if date_col else "",
                "url": url,
            }
        )
    return rows


def download_sample_pdfs(raw_dir: Path, symbol: str, limit: int = 4) -> list[Path]:
    """下载样本研报 PDF 到 raw_dir（按披露时间倒序取最新 limit 份），写 sources.json。"""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)  # 独立调用时也保证目录存在
    rows: list[dict[str, str]] = fetch_report_rows(symbol)
    saved: list[Path] = []
    sources: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        if len(saved) >= limit:
            break
        safe: str = re.sub(r'[\\/:*?"<>|\s]+', "_", row["title"])[:60] or f"report_{i}"
        dest: Path = raw_dir / f"{i + 1:02d}_{safe}.pdf"
        if dest.exists():  # 已存在则跳过（幂等重跑）
            saved.append(dest)
            sources.append({"file": dest.name, **row})
            continue
        try:
            resp = requests.get(row["url"], headers=_HTTP_HEADERS, timeout=60)
            resp.raise_for_status()
            if len(resp.content) < 10_000 or resp.content[:5] != b"%PDF-":
                logger.warning("跳过非 PDF 响应：%s（%d 字节）", row["url"], len(resp.content))
                continue
            dest.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001 —— 单份下载失败跳过，凑够数量即可
            logger.warning("研报下载失败：%s（%s）", row["title"], exc)
            continue
        saved.append(dest)
        sources.append({"file": dest.name, **row})
        print(f"  已下载 [{len(saved)}/{limit}] {row['title'][:40]}（{row['org']}，{row['date']}）")
    if not saved:
        raise RuntimeError("样本研报下载全部失败：请手动放置 PDF 到 data/raw/ 后用 --skip-download")
    SOURCES_JSON.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def load_sources() -> dict[str, dict[str, str]]:
    """读取 sources.json：文件名 → 来源信息（不存在则空）。"""
    if not SOURCES_JSON.exists():
        return {}
    try:
        data: Any = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
        return {str(item["file"]): item for item in data}
    except Exception:  # noqa: BLE001 —— 来源信息非关键路径
        return {}


# ------------------------------------------------------------------ PDF 解析与切分


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句（保留标点），丢弃空白句。"""
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page(text: str) -> list[str]:
    """单页文本切分为带重叠的定长 chunk（字符数口径，句边界优先）。"""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    buf: str = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= CHUNK_SIZE:
            buf = f"{buf}{s}" if buf else s
            continue
        if buf:
            chunks.append(buf)
        buf = (buf[-CHUNK_OVERLAP:] if buf else "") + s  # 相邻 chunk 重叠 60 字符
        while len(buf) > CHUNK_SIZE:  # 超长句硬切
            chunks.append(buf[:CHUNK_SIZE])
            buf = buf[-CHUNK_OVERLAP:]
    if buf:
        chunks.append(buf)
    return chunks


def _infer_meta(filename: str, sources: dict[str, dict[str, str]]) -> dict[str, Any]:
    """来源元数据：优先 sources.json，否则从文件名正则解析（ticker/date）。"""
    info = sources.get(filename, {})
    title = info.get("title") or filename
    ticker = ""
    m = re.search(r"\d{6}", title)
    if m:
        ticker = m.group(0)
    date = info.get("date", "")
    if not date:
        m = re.search(r"20\d{2}[-./年]?\d{1,2}", filename)
        date = m.group(0) if m else "未知"
    return {
        "ticker": ticker,
        "date": date,
        "source": (info.get("org") or title)[:120],
    }


def parse_pdf(path: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """解析单个 PDF：按页抽取文本 → 切分 → 生成入库行（doc_id/chunk_text/metadata）。"""
    import pymupdf  # PyMuPDF 2.0+：fitz 别名已弃用

    doc = pymupdf.open(str(path))
    rows: list[dict[str, Any]] = []
    try:
        for page_no in range(len(doc)):
            text: str = doc[page_no].get_text().strip()
            if len(text) < 20:
                continue  # 跳过空白页/图片页
            for ci, ctext in enumerate(chunk_page(text)):
                row_meta = dict(meta)
                row_meta["page"] = page_no + 1
                rows.append(
                    {
                        "doc_id": f"{path.stem}#p{page_no + 1}c{ci}",
                        "chunk_text": ctext,
                        "metadata": row_meta,
                    }
                )
    finally:
        doc.close()
    return rows


# ------------------------------------------------------------------ Milvus 集合与写入


def build_collection(client: Any, dim: int, rebuild: bool) -> str:
    """创建集合（doc_id/chunk_text/embedding/metadata）+ HNSW 索引；已存在则复用。"""
    from pymilvus import DataType  # 延迟导入

    coll = config_module.CONFIG.milvus_collection
    if rebuild and client.has_collection(coll):
        client.drop_collection(coll)
    if client.has_collection(coll):
        print(f"集合 {coll} 已存在，复用（如需覆盖重建请加 --rebuild）")
        return coll
    schema = client.create_schema(auto_id=False)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("metadata", DataType.JSON)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",           # 选型理由见模块 docstring
        metric_type="COSINE",        # bge 向量已归一化，余弦等价内积
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(coll, schema=schema, index_params=index_params)
    client.load_collection(coll)
    return coll


def main(argv: list[str] | None = None) -> int:
    """主流程：下载 PDF → 解析切分 → bge 向量化 → 写入 Milvus → 样例检索验证。"""
    parser = argparse.ArgumentParser(description="构建 Milvus 研报知识库（data/raw PDF → 向量库）")
    parser.add_argument("--symbol", default=_DEFAULT_SYMBOL, help=f"研报对应股票代码（默认 {_DEFAULT_SYMBOL}）")
    parser.add_argument("--skip-download", action="store_true", help="不下载，仅处理 data/raw 已有 PDF")
    parser.add_argument("--rebuild", action="store_true", help="先删除集合再重建")
    args = parser.parse_args(argv)

    print("=== AgentCouncil 研报知识库构建（Milvus + bge）===")
    print(f"配置：Milvus {config_module.CONFIG.milvus_host}:{config_module.CONFIG.milvus_port}"
          f" / 集合 {config_module.CONFIG.milvus_collection} / 模型 {config_module.CONFIG.bge_model}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs and not args.skip_download:
        print(f"data/raw 为空，开始下载样本研报（东方财富研报中心，{args.symbol} 公开卖方研报）…")
        pdfs = download_sample_pdfs(RAW_DIR, args.symbol)
    if not pdfs:
        print("错误：data/raw 下没有 PDF。请运行 python data/ingest.py 自动下载，"
              "或手动放入公开研报/财报 PDF 后加 --skip-download", file=sys.stderr)
        return 1

    # 1) 解析与切分
    sources = load_sources()
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    for pdf in pdfs:
        meta = _infer_meta(pdf.name, sources)
        pdf_rows = parse_pdf(pdf, meta)
        if not pdf_rows:
            print(f"  警告：{pdf.name} 未抽取到文本（可能为扫描件），跳过")
            continue
        rows.extend(pdf_rows)
        print(f"  解析 {pdf.name}：{len(pdf_rows)} 个 chunk（来源：{meta['source']}）")
    if not rows:
        print("错误：所有 PDF 均无可抽取文本", file=sys.stderr)
        return 1
    print(f"共 {len(rows)} 个 chunk（切分参数：{CHUNK_SIZE} 字符 / 重叠 {CHUNK_OVERLAP}）")

    # 2) bge 向量化（首次运行会下载模型权重）
    from data.embedder import get_embedder

    embedder = get_embedder()
    texts = [r["chunk_text"] for r in rows]
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        vectors.extend(embedder.encode(texts[i : i + batch]))
    print(f"向量化完成：{len(vectors)} 个向量，维度 {embedder.dim}，耗时 {time.time() - t0:.1f}s")

    # 3) 写入 Milvus（HNSW 索引）
    client = get_milvus_client()
    coll = build_collection(client, embedder.dim, args.rebuild)
    for i in range(0, len(rows), 64):
        batch_rows = [
            {"doc_id": r["doc_id"], "chunk_text": r["chunk_text"],
             "embedding": vectors[j], "metadata": r["metadata"]}
            for j, r in enumerate(rows[i : i + 64], start=i)
        ]
        client.insert(coll, batch_rows)
    client.flush(coll)
    print(f"已写入集合 {coll}：{len(rows)} 行（索引 HNSW，metric=COSINE，M=16，efConstruction=200）")

    # 4) 样例检索验证（真实检索，验证端到端可用）
    probe = client.search(
        collection_name=coll,
        data=[embedder.encode(["白酒 行业 竞争格局 盈利"])[0]],
        limit=3,
        output_fields=["doc_id", "metadata"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )[0]
    print("样例检索（query=白酒 行业 竞争格局 盈利）：")
    for h in probe:
        print(f"  score={h['distance']:.4f}  {h['entity']['metadata'].get('source', '?')}"
              f"（{h['entity']['doc_id']}）")
    print(f"=== 建库完成（总耗时 {time.time() - t0:.1f}s）===")
    return 0


if __name__ == "__main__":
    sys.exit(main())