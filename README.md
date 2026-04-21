# BestShot MLOps Project

BestShot extends Immich with ML-based photo quality scoring and feedback-driven retraining.
This repository contains versioned code + infrastructure so the project can be reproduced from scratch.

## One-path Setup

1. Use `serving/setup_node.ipynb` as the primary operator notebook.
2. On the node, clone the repository.
3. Export required environment variables.
4. Run `bash infra/bootstrap.sh`.
5. Verify pods and CronJobs.

```bash
git clone https://github.com/anokhimehta/bestshot.git
cd bestshot

export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_APPLICATION_CREDENTIAL_ID="<id>"
export OS_APPLICATION_CREDENTIAL_SECRET="<secret>"
export OS_REGION_NAME="CHI@TACC"
export BUCKET_NAME="<swift_bucket>"
export IMMICH_DB_PASSWORD="<db_password>"
export IMMICH_API_KEY="<immich_api_key>"
export MLFLOW_TRACKING_URI="http://<FLOATING_IP>:30500"

bash infra/bootstrap.sh

kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
kubectl get pods -n bestshot-app
```

## Runbook (Quick Recovery)

Use this when you need to recreate or refresh the environment on CHI@TACC.

```bash
# 1) SSH + repo
ssh -i ~/.ssh/id_rsa cc@<CHI_TACC_IP>
cd ~/bestshot

# 2) Required environment variables
export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_APPLICATION_CREDENTIAL_ID="<id>"
export OS_APPLICATION_CREDENTIAL_SECRET="<secret>"
export OS_REGION_NAME="CHI@TACC"
export BUCKET_NAME="<swift_bucket>"
export IMMICH_DB_PASSWORD="<db_password>"
export IMMICH_API_KEY="<immich_api_key>"
export MLFLOW_TRACKING_URI="http://<CHI_TACC_IP>:30500"

# 3) Re-apply everything
bash infra/bootstrap.sh

# 4) Verify
kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
kubectl get pods -n bestshot-app

# 5) Optional: generate fresh interaction events
source venv/bin/activate
python data/generator/simulate_users.py
```

## What Is Automated

- Sidecar integration runs via `infra/k8s/app/sidecar-deployment.yaml`.
- Retrain, promotion, and rollback run via Kubernetes CronJobs.
- Model promotion gates on quality thresholds before Production transition.
- Rollback checks production health and restores a previous model if needed.

## Detailed Docs

- Data operations, quality gates, storage map, and safeguarding: `docs/data-ops.md`
- CI/CD behavior, deploy tunnel secrets, and skip options: `docs/cicd.md`
- Recovery checklist and rerun commands: `docs/runbook.md`

## Notes For Grading

- All operational assets are in Git (code, manifests, scripts, workflow).
- Secrets are injected via environment variables and are not committed.
