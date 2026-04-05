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
echo "[1/6] Installing K3s (single-node Kubernetes)..."
curl -sfL https://get.k3s.io | sh -
echo "Waiting for K3s to start..."
sleep 30

# Step 2 — Fix kubeconfig permissions
echo ""
echo "[2/6] Fixing kubeconfig permissions..."
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

# Step 3 — Set up kubectl for cc user
echo ""
echo "[3/6] Configuring kubectl for cc user..."
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown cc ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG=~/.kube/config

# Make it permanent across logins
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
source ~/.bashrc

# Step 4 — Verify K8S is running
echo ""
echo "[4/6] Verifying Kubernetes..."
kubectl get nodes
echo "✅ Kubernetes is running!"

# Step 5 — Install metrics server
echo ""
echo "[5/6] Installing metrics server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
echo "✅ Metrics server installed!"

# Step 6 — Create persistent storage directories
echo ""
echo "[6/6] Setting up storage directories..."
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