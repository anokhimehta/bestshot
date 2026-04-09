#!/bin/bash
# Usage: ./run.sh <config_name>
# Example: ./run.sh cpu_sequential

CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: ./run.sh <config_name>"
    echo "Available configs:"
    echo "  cpu_sequential"
    echo "  cpu_batch"
    echo "  gpu_sequential"
    echo "  gpu_batch"
    echo "  gpu_onnx"
    exit 1
fi

# set gpu flags if needed
if [[ "$CONFIG_NAME" == gpu_* ]]; then
    GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video"
else
    GPU_FLAGS=""
fi

echo "======================================"
echo "  Running config: $CONFIG_NAME"
echo "======================================\\n"

# clean up old container
docker rm -f bestshot-api 2>/dev/null

# start server
echo "Starting server..."
docker run -d --network host \
  $GPU_FLAGS \
  --name bestshot-api \
  -e CONFIG_NAME=$CONFIG_NAME \
  bestshot-serve \
  uvicorn app:app --host 0.0.0.0 --port 8000

# wait for server
echo "Waiting for server..."
until curl -s http://127.0.0.1:8000/docs > /dev/null; do
    sleep 2
    echo "  still waiting..."
done
echo "Server ready!\\n"

# resource snapshot
echo ""
echo "Resource usage:"
docker stats bestshot-api --no-stream \
  --format "  CPU: {{.CPUPerc}}  MEM: {{.MemUsage}}"

# run benchmark
echo ""
echo "Running benchmark..."
docker run --rm --network host \
  $GPU_FLAGS \
  -e CONFIG_NAME=$CONFIG_NAME \
  bestshot-serve \
  sh -c "python benchmark.py"

# stop server
docker stop bestshot-api
echo ""
echo "Done: $CONFIG_NAME"s