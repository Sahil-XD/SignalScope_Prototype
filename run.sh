#!/bin/bash
echo "==================================================="
echo "  SignalScope - AI Media Forensics and Verification"
echo "  SIH 2026 [Internal Hackathon] Team Syndicate"
echo "==================================================="
echo ""

# Ensure working directory is the repository root
cd "$(dirname "$0")"

# Prevent OpenMP collision
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PYTHON_CMD=""
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -x "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
elif [ -x "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10+ is not installed or not in PATH."
    exit 1
fi

echo "[*] Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

echo ""
echo "[*] Checking dependencies..."
$PYTHON_CMD -c "import fastapi, uvicorn, multipart, torch, torchvision, PIL, numpy, cv2" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[*] Installing missing dependencies from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[*] Trying CPU-optimized PyTorch fallback..."
        $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
else
    echo "[OK] All core dependencies verified."
fi

PORT=8000
if command -v lsof &> /dev/null; then
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "[*] Port 8000 in use. Switching to backup port 8001..."
        PORT=8001
    fi
fi

echo ""
echo "[*] Launching SignalScope on http://127.0.0.1:$PORT..."

# Open default browser after 2 seconds
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://127.0.0.1:$PORT") &
elif command -v open &> /dev/null; then
    (sleep 2 && open "http://127.0.0.1:$PORT") &
fi

$PYTHON_CMD -m uvicorn app.main:app --host 127.0.0.1 --port $PORT
