# BestShot MLOps Project

BestShot extends Immich with ML-based photo quality scoring and feedback-driven retraining.
This repository contains versioned code + infrastructure so the project can be reproduced from scratch.

## One-path Setup

1. Use `serving/setup_node.ipynb` as the primary operator notebook.
2. On the node, clone the repository.
3. Export required environment variables.
4. Run `bash infra/bootstrap.sh`.
5. Verify pods and CronJobs.

Production policy:
- Kubernetes is the single production runtime.
- `serving/setup_node.ipynb` is the single production notebook.
- Host Docker flows (`serving/start.sh`, `serving/run.sh`, manual `docker run`) are dev-only.

```bash
git clone https://github.com/anokhimehta/bestshot.git
cd bestshot

export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_APPLICATION_CREDENTIAL_ID="<id>"
export OS_APPLICATION_CREDENTIAL_SECRET="<secret>"
export OS_REGION_NAME="CHI@TACC"
export BUCKET_NAME="<swift_bucket>"
export AWS_ACCESS_KEY_ID="<s3_access_key>"
export AWS_SECRET_ACCESS_KEY="<s3_secret_key>"
export IMMICH_DB_PASSWORD="<db_password>"
export MLFLOW_TRACKING_URI="http://<FLOATING_IP>:30500"

bash infra/bootstrap.sh

kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
kubectl get pods -n bestshot-app
```

`IMMICH_API_KEY` is optional during first bootstrap. If omitted, create it after Immich is up:

```bash
kubectl create secret generic immich-sidecar-secret \
  --from-literal=IMMICH_API_KEY="<immich_api_key>" \
  -n bestshot-app --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/immich-sidecar -n bestshot-app
```

## Immich Sidecar Automation

The sidecar integration in `serving/sidecar.py` is deployed as Kubernetes `Deployment`
(`infra/k8s/app/sidecar-deployment.yaml`), side-by-side with Immich and BestShot services.
It continuously:

- reads image metadata/events from Immich,
- computes quality scores through the serving endpoint,
- writes score metadata back to Immich,
- sends user-action feedback to `/feedback` for retraining signals.

### Retraining triggers

`retrain.py` checks three conditions before triggering a retrain. Any one being true is sufficient:

- New interaction events: >= 500 since last retrain
- Days elapsed: >= 7 days
- Negative feedback rate: >= 40% over >= 50 events

### Automated promotion and rollback

`infra/k8s/jobs/promote-cronjob.yaml` runs nightly to move qualified models from Staging to Production.
`infra/k8s/jobs/rollback-cronjob.yaml` checks production health and restores a previous model when required.

Manual commands (if needed):

```bash
bash infra/scripts/promote.sh
bash infra/scripts/rollback.sh
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
export AWS_ACCESS_KEY_ID="<s3_access_key>"
export AWS_SECRET_ACCESS_KEY="<s3_secret_key>"
export IMMICH_DB_PASSWORD="<db_password>"
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
- Bootstrap (`infra/bootstrap.sh`) automates K3s setup, namespace/secret creation, and manifest apply.

## Dev-only Paths (Not Production)

- `serving/start.sh` and `serving/run.sh` are for local benchmarking/debug only.
- `training/docker/docker-compose-mlflow.yaml` is for local experiments only.
- Do not use host Docker services when validating production behavior.

## Detailed Docs

- Data operations, quality gates, storage map, and safeguarding: `docs/data-ops.md`
- CI/CD behavior, deploy tunnel secrets, and skip options: `docs/cicd.md`
- Recovery checklist and rerun commands: `docs/runbook.md`

## Forced Training Trigger
- docker run --rm \
  --network host \
  --env-file ~/bestshot/.env \
  ghcr.io/anokhimehta/bestshot-training:latest \
  python training/train.py --config training/config/partial_finetune_highlr.yaml

## Notes For Grading

- All operational assets are in Git (code, manifests, scripts, workflow).
- Secrets are injected via environment variables and are not committed.
