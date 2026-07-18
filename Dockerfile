# ═══════════════════════════════════════════════════════════════
# Laike RAG - Docker 镜像
# 构建: docker build -t laike-rag .
# 运行: docker run -p 8000:8000 --env-file .env laike-rag
# ═══════════════════════════════════════════════════════════════

FROM python:3.13-slim

LABEL maintainer="laike-rag"
LABEL description="抖音来客 AI 客服 RAG 问答系统"

WORKDIR /app

# Python 依赖（分层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# 项目文件
COPY . .

# 创建必要目录
RUN mkdir -p docs knowledge_base static

# 暴露端口
EXPOSE 8000

# 环境变量默认值
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000

# 启动服务
CMD ["sh", "-c", "uvicorn web_service:app --host ${WEB_HOST:-0.0.0.0} --port ${WEB_PORT:-8000}"]
