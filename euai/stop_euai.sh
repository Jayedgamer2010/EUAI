#!/bin/bash
# EUAI Stop Script

echo "=== Stopping EUAI ==="

# Kill API server
if [ -f /tmp/euai_api.pid ]; then
    API_PID=$(cat /tmp/euai_api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo "Stopping API server (PID $API_PID)..."
        kill $API_PID 2>/dev/null || true
        sleep 1
    fi
    rm -f /tmp/euai_api.pid
fi

# Also kill any remaining server.py processes
pkill -f "api/server.py" 2>/dev/null || true

echo "EUAI stopped."
