# AgentCouncil 服务镜像（FastAPI + SSE）
# 构建：docker build -t agentcouncil:latest .
# 运行：docker run --rm -p 8000:8000 --env-file .env agentcouncil:latest
#
# 说明：
# - 基础镜像 python:3.11-slim 与 pyproject.toml 的 target-version=py311 对齐；
# - torch 先按 CPU 源安装，避免默认拉取 CUDA 版本把镜像撑到 GB 级；
#   sentence-transformers 随后安装时复用该 torch；
# - bge 权重与 HuggingFace 缓存放 HF_HOME，建议用 volume 持久化，
#   否则每次重建容器都要重新下载约 1.3GB 权重。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# build-essential 用于少数需要本地编译的依赖；清理 apt 缓存控制镜像体积
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -i https://pypi.org/simple

COPY . .

EXPOSE 8000

# 健康检查直接打 /health（slim 镜像无 curl，用 Python 标准库实现）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]