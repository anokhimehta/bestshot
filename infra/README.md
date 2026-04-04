# BestShot Infrastructure

## Overview
This folder contains all DevOps/Platform materials for the BestShot project.
- `chameleon/` - Jupyter notebook to provision VMs on Chameleon KVM@TACC
- `k8s/platform/` - Kubernetes manifests for MLflow (shared platform service)
- `k8s/app/` - Kubernetes manifests for Immich (open-source base service)

## How to bring up the system

### Step 1 - Provision infrastructure
Open `chameleon/provision.ipynb` in Chameleon Jupyter at KVM@TACC and run all cells.

### Step 2 - SSH into master node
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@
```

### Step 3 - Install Kubernetes (run on master node)
```bash
curl -sfL https://get.k3s.io | sh -
```

### Step 4 - Deploy platform services
```bash
kubectl apply -f k8s/platform/
```

### Step 5 - Create secrets (never committed to git)
```bash
kubectl create secret generic immich-db-secret \
  --from-literal=DB_PASSWORD=yourpassword \
  --from-literal=DB_USERNAME=immich \
  --from-literal=DB_DATABASE_NAME=immich \
  -n bestshot-app
```

### Step 6 - Deploy Immich
```bash
kubectl apply -f k8s/app/
```

## Access
- MLflow: http://<MASTER_IP>:30500
- Immich: http://<MASTER_IP>:30283