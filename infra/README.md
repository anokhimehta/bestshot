# BestShot Infrastructure — DevOps/Platform

## Overview
Single-node Kubernetes cluster on Chameleon KVM@TACC.
- One VM runs everything: K8S control plane + Immich + MLflow
- Block storage volume for persistent data
- Project suffix: proj19

Production policy:
- Use one production notebook: `serving/setup_node.ipynb`.
- Use one production execution path: `bash infra/bootstrap.sh` + Kubernetes manifests in `infra/k8s/`.
- Host-level Docker scripts are dev-only and should not be used for production rollout.

## Services
| Service | URL | Port |
|---|---|---|
| Immich (open source base) | http://<NODE_IP>:30283 | 30283 |
| MLflow (platform service) | http://<NODE_IP>:30500 | 30500 |

## How to bring up the system

### Step 1 - Provision VM:
Open `infra/chameleon/provision.ipynb` in Chameleon Jupyter at KVM@TACC.
Run all cells in order except last cell.
Note the floating IP printed at the end.

### Step 2 - SSH into node:
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING_IP>
```

### Step 3 - Format block volume:
```bash
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/vdb1
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
sudo chown -R cc /mnt/block
echo "/dev/vdb1 /mnt/block ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

### Step 4 - Clone repo and install Kubernetes:
```bash
git clone https://github.com/anokhimehta/bestshot.git
cd bestshot
bash infra/k8s/setup_k8s.sh
```

### Step 5 - Deploy platform services (MLflow):
```bash
kubectl apply -f infra/k8s/platform/
```

### Step 6 - Create secret (never committed to git):
```bash
kubectl create secret generic immich-db-secret \
  --from-literal=DB_PASSWORD=<password> \
  --from-literal=DB_USERNAME=immich \
  --from-literal=DB_DATABASE_NAME=immich \
  -n bestshot-app
```

### Step 7 - Deploy open source service (Immich):
```bash
kubectl apply -f infra/k8s/app/
```

### Step 8 - Verify:
```bash
kubectl get pods -n bestshot-platform
kubectl get pods -n bestshot-app
```

## Endpoints:
- Immich: http://<FLOATING_IP>:30283
- MLflow: http://<FLOATING_IP>:30500

## Secrets
Secrets are never committed to git. Create `immich-db-secret` with `kubectl create secret` as in Step 6 (`DB_PASSWORD`, `DB_USERNAME`, `DB_DATABASE_NAME`).

## Full System Architecture

### Environments
| Environment | URL | Model Stage |
|---|---|---|
| Staging | http://129.114.27.200:30801 | Staging |
| Canary | http://129.114.27.200:30802 | Production (small traffic) |
| Production | http://129.114.27.200:30803 | Production |
| MLflow | http://129.114.27.200:30500 | — |
| Immich | http://129.114.27.200:30283 | — |

### Automated Workflows
- Retraining: CronJob runs nightly at 2am (infra/k8s/jobs/retrain-cronjob.yaml)
- Promotion: CronJob runs at 2:30am, promotes if PLCC≥0.85, SRCC≥0.83
- Rollback: CronJob checks production health every 15 min, auto-rollback if down
- Health monitoring: CronJob runs every 5 min, alerts if any service fails
- Autoscaling: HPA scales production serving pods when CPU>70% or RAM>80%

### CI/CD
Push to main → GitHub Actions builds all images → pushes to ghcr.io → deploys to K8S
See .github/workflows/ci.yml

### Secrets Required (create manually — never in git)
kubectl create secret generic openstack-credentials \
  --from-literal=OS_APPLICATION_CREDENTIAL_ID=<id> \
  --from-literal=OS_APPLICATION_CREDENTIAL_SECRET=<secret> \
  -n bestshot-platform

kubectl create secret generic mlflow-artifact-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=<s3_access_key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<s3_secret_key> \
  -n bestshot-platform

`IMMICH_API_KEY` can be added after first bootstrap (once Immich UI is up):

kubectl create secret generic immich-sidecar-secret \
  --from-literal=IMMICH_API_KEY=<immich_api_key> \
  -n bestshot-app --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/immich-sidecar -n bestshot-app