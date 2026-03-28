#!/bin/bash
# EUAI Startup Script

echo "=== Starting EUAI ==="

# Set OMP variables
export OMP_NUM_THREADS=1
export KMP_AFFINITY=disabled

# Paths
PROJECT_ROOT="/home/storage/EUAI"
BINARY="$PROJECT_ROOT/build/euai"
CONFIG="$PROJECT_ROOT/config"
MODEL="$PROJECT_ROOT/models/qwen2.5-coder-0.5b-instruct-q2_k.gguf"

# Check binary
if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Binary not found: $BINARY"
    echo "Run: cd $PROJECT_ROOT && make"
    exit 1
fi

# Check model
if [ ! -f "$MODEL" ]; then
    echo "[WARN] Model not found: $MODEL"
fi

echo "[INFO] Starting FastAPI server on port 11434..."

# Start API server
cd "$PROJECT_ROOT/euai"
python3 api/server.py &
API_PID=$!

echo "[INFO] API server started with PID $API_PID"
echo ""
echo "EUAI is running!"
echo "  Local API: http://localhost:11434"
echo "  Health:    http://localhost:11434/health"
echo ""
echo "To stop: $PROJECT_ROOT/euai/stop_euai.sh"
echo ""

# Save PID
echo $API_PID > /tmp/euai_api.pid
