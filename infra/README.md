# BestShot Infrastructure — DevOps/Platform

## Overview
Single-node Kubernetes cluster on Chameleon KVM@TACC.
- One VM runs everything: K8S control plane + Immich + MLflow
- Block storage volume for persistent data
- Project suffix: proj19

## Services
| Service | URL | Port |
|---|---|---|
| Immich (open source base) | http://<NODE_IP>:30283 | 30283 |
| MLflow (platform service) | http://<NODE_IP>:30500 | 30500 |

## How to bring up the system

### Step 1 - Provision VM on Chameleon
Open `chameleon/provision.ipynb` in Chameleon Jupyter at KVM@TACC.
Run all cells in order.
Note the floating IP printed at the end.

### Step 2 - SSH into node
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING_IP>
```

### Step 3 - Format block volume
```bash
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/vdb1
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
sudo chown -R cc /mnt/block
echo "/dev/vdb1 /mnt/block ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

### Step 4 - Install Kubernetes
```bash
bash infra/k8s/setup_k8s.sh
```

### Step 5 - Clone repo on node
```bash
git clone https://github.com/anokhimehta/bestshot.git
cd bestshot
```

### Step 6 - Create secrets (never in git)
```bash
kubectl create secret generic immich-db-secret \
  --from-literal=DB_PASSWORD=BestShot2024! \
  --from-literal=DB_USERNAME=immich \
  --from-literal=DB_DATABASE_NAME=immich \
  -n bestshot-app
```

### Step 7 - Deploy platform (MLflow)
```bash
kubectl apply -f infra/k8s/platform/
```

### Step 8 - Deploy Immich
```bash
kubectl apply -f infra/k8s/app/
```

### Step 9 - Verify
```bash
kubectl get pods -n bestshot-platform
kubectl get pods -n bestshot-app
```

## Access
- Immich: http://<FLOATING_IP>:30283
- MLflow: http://<FLOATING_IP>:30500

## Secrets
Secrets are never committed to git. Create `immich-db-secret` with `kubectl create secret` as in Step 6 (`DB_PASSWORD`, `DB_USERNAME`, `DB_DATABASE_NAME`).

For a local YAML template only, see `k8s/app/immich-secrets.example.yaml` (present in your working copy; listed in `.gitignore` so it is **not** pushed to GitHub).