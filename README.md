# BestShot MLOps Project

BestShot extends Immich with ML-based photo quality scoring and feedback-driven retraining.
This repository contains the full reproducible project material (code, infra manifests, CI/CD, automation scripts) so graders can rebuild the system from scratch.

## What Is Automated End-to-End

- Production feedback and interaction events are written to object storage.
- Retraining checks run automatically (`bestshot-retrain` CronJob).
- Training artifacts/metrics are logged to MLflow.
- Model promotion from `Staging` to `Production` is automated with quality gates.
- Rollback is automated when production health checks fail.
- Deployments are updated through Kubernetes automation (not manual SSH promotion steps).

## Repository Layout

```text
bestshot/
├── data/                      # ingestion, generator, feature + batch pipelines
├── serving/                   # FastAPI inference + feedback capture
├── training/                  # training, eval, retrain logic
├── infra/
│   ├── bootstrap.sh           # one-command environment bootstrap
│   ├── scripts/
│   │   ├── promote.py         # promotion/rollback core logic
│   │   ├── promote.sh         # manual promotion helper
│   │   └── rollback.sh        # manual rollback helper
│   └── k8s/
│       ├── app/
│       ├── environments/
│       ├── jobs/
│       ├── monitoring/
│       └── platform/
└── .github/workflows/ci.yml   # build + deploy workflow
```

## Reproduce From Scratch

### 1) Provision host

Use `infra/chameleon/provision.ipynb` to provision a node and note the floating IP.

### 2) Clone repo

```bash
git clone https://github.com/anokhimehta/bestshot.git
cd bestshot
```

### 3) Set required environment variables

```bash
export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_APPLICATION_CREDENTIAL_ID="<id>"
export OS_APPLICATION_CREDENTIAL_SECRET="<secret>"
export OS_REGION_NAME="CHI@TACC"
export BUCKET_NAME="<swift_bucket>"
export IMMICH_DB_PASSWORD="<db_password>"
export MLFLOW_TRACKING_URI="http://<FLOATING_IP>:30500"
```

### 4) Bootstrap everything with one command

```bash
bash infra/bootstrap.sh
```

This installs/configures K3s, creates namespaces/secrets, and deploys platform/app/environments/monitoring/jobs.

### 5) Verify deployment

```bash
kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
```

## Model Promotion and Rollback

### Automated promotion

`infra/k8s/jobs/promote-cronjob.yaml` runs nightly:

- reads the latest model in MLflow `Staging`
- checks quality gates (`PLCC >= 0.85` and `SRCC >= 0.83`)
- archives old production model
- promotes the staging model to `Production`

### Automated rollback

`infra/k8s/jobs/rollback-cronjob.yaml` runs every 15 minutes:

- checks production `/health`
- if unhealthy, runs rollback logic to restore the most recent archived model
- restarts serving deployments so the restored model is loaded

### Manual commands (if needed)

```bash
bash infra/scripts/promote.sh
bash infra/scripts/rollback.sh
```

## CI/CD

On push to `main`, GitHub Actions in `.github/workflows/ci.yml`:

- builds/pushes service images to GHCR
- applies Kubernetes manifests to the target cluster

## Notes For Grading

- All operational assets are versioned in Git (code, manifests, scripts, workflow).
- Secrets are injected from environment variables and are **not** committed.
- Promotion/rollback are automated workflows; humans can optionally trigger wrappers but do not need to SSH and manually mutate deployment state.
