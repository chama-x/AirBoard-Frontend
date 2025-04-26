#!/bin/bash
# Script to start the AirBoard backend server 
# Expected final location: /Users/chamaththiwanka/Desktop/Projects/start_airboard_backend.sh

BACKEND_DIR="/Users/chamaththiwanka/Desktop/Projects/AirBoard/backend_ml_model"
VENV_PYTHON="$BACKEND_DIR/venv/bin/python"

echo "Navigating to backend directory: $BACKEND_DIR"
cd "$BACKEND_DIR" || { echo "Failed to navigate to backend directory. Exiting."; exit 1; }

echo "Starting Uvicorn server (Press CTRL+C to quit)..."
"$VENV_PYTHON" -m uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

echo "Server stopped." 