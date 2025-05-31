#!/usr/bin/bash
port=$1
workers=$2
timeout=$3

python3 mcp-server/dootask-task.py &

uvicorn main:app --workers $workers --port $port --timeout-keep-alive $timeout