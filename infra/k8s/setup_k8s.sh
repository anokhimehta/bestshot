#!/bin/bash
# deploy.sh
# Deploys all BestShot services in correct order
# Usage: bash infra/k8s/deploy.sh

set -e
echo "======================================"
echo " BestShot Deploy Script — proj19"
echo "======================================"

cd ~/bestshot

# Step 1 — Create namespaces first and wait
echo "[1/4] Creating namespaces..."
kubectl apply -f infra/k8s/platform/namespace.yaml
echo "Waiting for namespaces to be ready..."
sleep 5

# Step 2 — Deploy platform (MLflow)
echo "[2/4] Deploying platform services (MLflow)..."
kubectl apply -f infra/k8s/platform/mlflow-pvc.yaml
kubectl apply -f infra/k8s/platform/mlflow-deployment.yaml
kubectl apply -f infra/k8s/platform/mlflow-service.yaml
echo "✅ Platform services deployed!"

# Step 3 — Create Immich secret if it doesn't exist
echo "[3/4] Creating secrets..."
kubectl create secret generic immich-db-secret \
  --from-literal=DB_PASSWORD=BestShot2024! \
  --from-literal=DB_USERNAME=immich \
  --from-literal=DB_DATABASE_NAME=immich \
  -n bestshot-app \
  --dry-run=client -o yaml | kubectl apply -f -
echo "✅ Secrets created!"

# Step 4 — Deploy Immich
echo "[4/4] Deploying Immich..."
kubectl apply -f infra/k8s/app/immich-pvc.yaml
kubectl apply -f infra/k8s/app/immich-deployment.yaml
echo "✅ Immich deployed!"

echo ""
echo "======================================"
echo " All services deployed!"
echo " Checking status..."
echo "======================================"
kubectl get pods -n bestshot-platform
kubectl get pods -n bestshot-app

echo ""
echo " MLflow: http://$(curl -s ifconfig.me 2>/dev/null):30500"
echo " Immich:  http://$(curl -s ifconfig.me 2>/dev/null):30283"
echo "======================================"