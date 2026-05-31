# ============================================
# Stage 1: Build Vue.js frontend
# ============================================
FROM node:20-slim AS frontend-builder

RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app/frontend

# Copy package files first for better layer caching
COPY frontend/package.json frontend/pnpm-lock.yaml ./

RUN pnpm install --frozen-lockfile

# Copy frontend source and build
COPY frontend/ ./

RUN pnpm run build


# ============================================
# Stage 2: Python backend + serve everything
# ============================================
FROM python:3.11-slim

# Install build essentials needed for compiled Python packages (e.g. chroma-hnswlib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run.py ./
COPY app/ ./app/

# Copy data directory (laws, case database, etc.)
COPY data/ ./data/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist/ ./frontend/dist/

EXPOSE 9000

CMD ["python", "run.py"]
