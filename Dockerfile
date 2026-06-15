# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG HF_ENDPOINT=https://hf-mirror.com

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_ENDPOINT=${HF_ENDPOINT} \
    HF_HOME=/app/models_cache \
    HF_HUB_CACHE=/app/models_cache/hub \
    HF_HUB_DISABLE_XET=1 \
    PORT=9000

WORKDIR /app

# 使用国内 Debian 镜像源；可通过 --build-arg APT_MIRROR=... 覆盖。
RUN sed -i \
        -e "s|http://deb.debian.org/debian|${APT_MIRROR}/debian|g" \
        -e "s|http://deb.debian.org/debian-security|${APT_MIRROR}/debian-security|g" \
        -e "s|http://security.debian.org/debian-security|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# 默认使用国内 PyPI 镜像；可通过 --build-arg PIP_INDEX_URL=... 覆盖。
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY run.py ./
COPY app/ ./app/
COPY data/ ./data/
COPY .env.example ./

RUN mkdir -p /app/chroma_db /app/models_cache /app/data/db /app/logs \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"9000\")}/api/health', timeout=3).read()" || exit 1

CMD ["python", "run.py"]
