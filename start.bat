@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ========================================
echo   法律顾问 Agent 启动
echo ========================================

:: 1. 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查 Conda
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [1/4] 检测到 Conda，激活 myenv 环境...
    call conda activate myenv 2>nul
)

echo [1/4] Python:
python --version

:: 2. 检查 .env
if not exist ".env" (
    if exist ".env.example" (
        echo [2/4] .env 不存在，从 .env.example 复制...
        copy .env.example .env >nul
        echo   请编辑 .env 填入 API Key 后重新运行
        pause
        exit /b 1
    )
)
echo [2/4] .env 已就绪

:: 3. 检查依赖
echo [3/4] 检查 Python 依赖...
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo   安装依赖...
    python -m pip install -r requirements.txt -q
)

:: 4. 启动
echo [4/4] 启动后端服务...
echo ========================================
echo   服务地址: http://localhost:9000
echo   API 文档: http://localhost:9000/docs
echo   前端地址: http://localhost:5173
echo ========================================
echo   按 Ctrl+C 停止服务
echo.

:: 启动前端（后台）
if exist "frontend\package.json" (
    where pnpm >nul 2>&1
    if !errorlevel! equ 0 (
        echo   启动前端开发服务器...
        start /b cmd /c "cd frontend && pnpm install --silent && pnpm dev"
    )
)

:: 启动后端（前台）
python run.py
