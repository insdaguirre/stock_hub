#!/bin/bash
# start.sh - Startup script for FastAPI application on Render

echo "Starting application..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la

# Run the simplified application
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --log-level debug app:app 