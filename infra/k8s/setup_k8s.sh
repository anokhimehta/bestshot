#!/bin/bash
# setup_k8s.sh
# CaC script — installs and configures single-node Kubernetes
# Run this on the Chameleon node after provisioning
# Usage: bash infra/k8s/setup_k8s.sh

set -e
echo "======================================"
echo " BestShot K8S Setup — proj19"
echo " Single node installation"
echo "======================================"

# Step 1 — Install K3s
echo ""
echo "[1/5] Installing K3s (single-node Kubernetes)..."
curl -sfL https://get.k3s.io | sh -
echo "Waiting for K3s to start..."
sleep 30

# Step 2 — Configure kubectl for cc user
echo ""
echo "[2/5] Configuring kubectl..."
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown cc ~/.kube/config
export KUBECONFIG=~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc

# Step 3 — Verify K8S is running
echo ""
echo "[3/5] Verifying Kubernetes..."
kubectl get nodes
echo "✅ Kubernetes is running!"

# Step 4 — Install metrics server
echo ""
echo "[4/5] Installing metrics server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
echo "✅ Metrics server installed!"

# Step 5 — Create persistent storage directories
echo ""
echo "[5/5] Setting up storage directories..."
sudo mkdir -p /mnt/block/mlflow
sudo mkdir -p /mnt/block/immich
sudo chown -R cc /mnt/block
echo "✅ Storage directories created!"

echo ""
echo "======================================"
echo " Setup complete!"
echo " Next steps:"
echo "   git clone https://github.com/anokhimehta/bestshot.git"
echo "   cd bestshot"
echo "   kubectl apply -f infra/k8s/platform/"
echo "   kubectl apply -f infra/k8s/app/"
echo "======================================"