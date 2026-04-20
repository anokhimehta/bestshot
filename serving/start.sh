#!/bin/bash
# Usage: ./start.sh
# Starts all BestShot services (assumes Docker image already built)
# Run this after a node reboot or to restart all services

cd ~/bestshot/serving

echo "======================================"
echo "  Starting BestShot Services"
echo "======================================"

# pull latest code
echo "Pulling latest code..."
cd ~/bestshot && git pull && cd serving

# clean up any old containers
docker rm -f bestshot-api 2>/dev/null
docker rm -f bestshot-sidecar 2>/dev/null

# start serving API
echo "Starting serving API..."
docker run -d --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --name bestshot-api \
  --env-file .env \
  -e CONFIG_NAME=gpu_sequential \
  --restart always \
  bestshot-serve \
  uvicorn app:app --host 0.0.0.0 --port 8000

# wait for server
echo "Waiting for server..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 2; echo "  still waiting..."
done
echo "Server ready!"

# start sidecar
echo "Starting sidecar..."
docker run -d --network host \
  --name bestshot-sidecar \
  --env-file .env \
  --restart always \
  bestshot-serve \
  python -u sidecar.py
echo "Sidecar started!"

# start monitoring
echo "Starting monitoring stack..."
docker compose -f docker/docker-compose-monitoring.yaml up -d prometheus grafana
echo "Prometheus: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):9090"
echo "Grafana:    http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"

echo ""
echo "======================================"
echo "  All services running!"
echo "======================================"
docker ps --format "  {{.Names}} — {{.Status}}"