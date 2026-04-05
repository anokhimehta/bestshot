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
echo "[1/3] Creating namespaces..."
kubectl apply -f infra/k8s/platform/00-namespace.yaml
echo "Waiting for namespaces to be ready..."
sleep 5

# Step 2 — Deploy platform (MLflow)
echo "[2/3] Deploying platform services (MLflow)..."
kubectl apply -f infra/k8s/platform/mlflow-pvc.yaml
kubectl apply -f infra/k8s/platform/mlflow-deployment.yaml
kubectl apply -f infra/k8s/platform/mlflow-service.yaml
echo "✅ Platform services deployed!"

# Step 3 — Deploy Immich
echo "[3/3] Deploying Immich..."
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
echo "======================================"
echo " NOTE: Before running this script,"
echo " make sure you have created the secret:"
echo ""
echo " kubectl create secret generic immich-db-secret \\"
echo "   --from-literal=DB_PASSWORD=<your_password> \\"
echo "   --from-literal=DB_USERNAME=immich \\"
echo "   --from-literal=DB_DATABASE_NAME=immich \\"
echo "   -n bestshot-app"
echo "======================================"