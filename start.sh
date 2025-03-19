#!/bin/bash
# start.sh - Startup script for FastAPI application on Render

echo "Starting application..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Files in directory:"
ls -la

# Run the application
exec gunicorn -w 6 -k uvicorn.workers.UvicornWorker --timeout 120 --log-level debug main:app 