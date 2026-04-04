#!/bin/bash
# setup_k8s.sh
# Run this on the master node after VMs are provisioned
# This is your CaC — configures Kubernetes on bare Ubuntu

set -e   # stop if any command fails

echo "======================================"
echo " BestShot K8S Setup Script"
echo " Project: proj19"
echo "======================================"

# Step 1 - Install K3s (lightweight Kubernetes)
echo "[1/6] Installing K3s..."
curl -sfL https://get.k3s.io | sh -
sleep 30   # wait for K3s to fully start

# Step 2 - Set up kubectl for cc user
echo "[2/6] Setting up kubectl..."
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown cc ~/.kube/config
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
export KUBECONFIG=~/.kube/config

# Step 3 - Verify K8S is running
echo "[3/6] Verifying K8S..."
kubectl get nodes

# Step 4 - Install metrics server (for kubectl top)
echo "[4/6] Installing metrics server..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Step 5 - Mount block volume for persistent storage
echo "[5/6] Setting up persistent storage..."
sudo mkdir -p /mnt/block/mlflow
sudo mkdir -p /mnt/block/immich

# Mount if not already mounted
if ! mountpoint -q /mnt/block; then
    sudo mount /dev/vdb1 /mnt/block
fi

sudo chown -R cc /mnt/block

# Step 6 - Done
echo "[6/6] Done!"
echo ""
echo "Next steps:"
echo "  kubectl apply -f k8s/platform/"
echo "  kubectl apply -f k8s/app/"
echo ""
echo "MLflow will be at:  http://$(curl -s ifconfig.me):30500"
echo "Immich will be at:  http://$(curl -s ifconfig.me):30283"