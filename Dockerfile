# 构建阶段
FROM python:3.11-slim-bookworm AS builder

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
RUN pip install --no-cache-dir -r requirements.txt gunicorn && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/*

# 运行阶段
FROM python:3.11-slim-bookworm

WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# 设置环境变量
ENV PORT=5001 \
    WORKERS=4 \
    TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    REDIS_HOST=redis \
    REDIS_PORT=6379

# 安装 wget
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY main.py .
COPY helper/ helper/
COPY static/ static/
COPY requirements.txt .
COPY README.md .
COPY LICENSE .
COPY start.sh .
# 创建非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# 暴露端口
EXPOSE $PORT

# 健康检查
HEALTHCHECK --start-period=10s --interval=3m --timeout=3s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:$PORT/health || exit 1

# 启动命令
# CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers $WORKERS --timeout $TIMEOUT --access-logfile - --error-logfile - main:app"]
CMD ["sh", "-c", "/app/start.sh", "$PORT", "$WORKERS", "$TIMEOUT"]