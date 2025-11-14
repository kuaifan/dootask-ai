# UI 构建阶段
FROM node:20-alpine AS ui-builder
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui .
RUN npm run build

# Python 依赖构建阶段
FROM python:3.12-slim-bookworm AS builder

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/*

# 运行阶段
FROM python:3.12-slim-bookworm

WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# 设置环境变量
ENV PORT=5001 \
    WORKERS=4 \
    TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    REDIS_HOST=redis \
    REDIS_PORT=6379

# 安装 curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY main.py .
COPY helper/ helper/
COPY static/ static/
COPY --from=ui-builder /ui/dist ./static/ui
COPY requirements.txt .
COPY README.md .
COPY LICENSE .

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# 暴露端口
EXPOSE $PORT

# 健康检查
HEALTHCHECK --start-period=10s --interval=3m --timeout=3s \
    CMD curl -f http://localhost:$PORT/health || exit 1

# 启动命令
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers $WORKERS"]
