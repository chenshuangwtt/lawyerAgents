#!/bin/bash
# 法律顾问 Agent 一键启动脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  法律顾问 Agent 启动${NC}"
echo -e "${GREEN}========================================${NC}"

# 1. 检查 Python 环境
if command -v conda &> /dev/null; then
    echo -e "${YELLOW}[1/4] 检测到 Conda，激活 myenv 环境...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate myenv 2>/dev/null || true
elif [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}[1/4] 未检测到虚拟环境，使用系统 Python${NC}"
fi

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}错误：未找到 Python，请先安装 Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}  Python: $($PYTHON --version)${NC}"

# 2. 检查 .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}[2/4] .env 不存在，从 .env.example 复制...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}  请编辑 .env 填入 API Key 后重新运行${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}[2/4] .env 已就绪${NC}"

# 3. 检查依赖
echo -e "${YELLOW}[3/4] 检查 Python 依赖...${NC}"
$PYTHON -c "import fastapi" 2>/dev/null || {
    echo -e "${YELLOW}  安装依赖...${NC}"
    $PYTHON -m pip install -r requirements.txt -q
}

# 4. 启动后端
echo -e "${GREEN}[4/4] 启动后端服务...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  服务地址: http://localhost:9000${NC}"
echo -e "${GREEN}  API 文档: http://localhost:9000/docs${NC}"
echo -e "${GREEN}  前端地址: http://localhost:5173${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}  按 Ctrl+C 停止服务${NC}"
echo ""

# 启动前端（后台）
if [ -d "frontend" ] && command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}  启动前端开发服务器...${NC}"
    (cd frontend && pnpm install --silent && pnpm dev) &
    FRONTEND_PID=$!
    trap "kill $FRONTEND_PID 2>/dev/null; exit" INT TERM
fi

# 启动后端（前台）
$PYTHON run.py
