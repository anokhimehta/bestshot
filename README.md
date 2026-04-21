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
export IMMICH_API_KEY="<immich_api_key>"
export MLFLOW_TRACKING_URI="http://<FLOATING_IP>:30500"
```

### 4) Bootstrap everything with one command

```bash
bash infra/bootstrap.sh
```

This installs/configures K3s, creates namespaces/secrets, and deploys platform/app/environments/monitoring/jobs.

**Primary deployment path:** after you provision the node, treat **`bash infra/bootstrap.sh`** on the host as the supported way to stand up or refresh Kubernetes workloads from this repo. GitHub Actions may also deploy when secrets are configured, but it is optional.

**New model versions:** day-two updates are driven by cluster **CronJobs** (retrain, promote, rollback) and rollout restarts—not by re-running bootstrap for every image tag. CI/CD mainly publishes images and can apply manifests when you choose to use it.

### 5) Verify deployment

```bash
kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
kubectl get pods -n bestshot-app
```

## Model Promotion and Rollback

## Immich Sidecar Automation

The sidecar integration in `serving/sidecar.py` is deployed as Kubernetes `Deployment`
`infra/k8s/app/sidecar-deployment.yaml` in namespace `bestshot-app`.

It continuously:

- polls Immich for new assets,
- calls the serving API for scoring,
- writes score metadata back to Immich,
- sends user-action feedback to `/feedback` for retraining signals.

### Retraining triggers

`retrain.py` checks three conditions before triggering a retrain. Any one being true is sufficient:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| New interaction events | ≥ 500 since last retrain | Ensures enough new data to improve the model |
| Days elapsed | ≥ 7 days | Time-based safety net for low-traffic periods |
| Negative feedback rate | ≥ 40% over ≥ 50 events | Signals model is actively failing users |

The minimum 50-event floor on negative feedback rate prevents retraining on statistically insignificant signals — a 60% negative rate from 5 users is noise, not a reliable signal.

### Quality gates and regression detection

After every training run, `train.py` evaluates the model on a held-out test set and applies two gates before registering to the MLflow Model Registry:

1. **Absolute quality gate** — model must meet minimum thresholds (PLCC ≥ 0.85, SRCC ≥ 0.83)
2. **Regression check** — model must not be more than 0.01 worse than the currently deployed Production model on either metric

Every run is tagged with a `promotion_status` in MLflow:

| Tag value | Meaning |
|-----------|---------|
| `approved` | Passed both gates, registered to Staging |
| `rejected_quality_gate` | Did not meet absolute thresholds |
| `rejected_regression` | Passed absolute thresholds but worse than Production |

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
- optionally applies Kubernetes manifests to the cluster (requires SSH + kubeconfig secrets below)

The **`deploy` job uses `continue-on-error: true` temporarily**, so a broken tunnel or missing secret does not fail the whole workflow while you rely on **`infra/bootstrap.sh`** after provisioning for cluster deployment.

For ongoing model lifecycle, **CronJobs** handle retrain / promote / rollback; serving pods pick up new models after promotion and restarts as defined in those jobs.

### GitHub Actions deploy secrets

The deploy job cannot dial your node’s public IP on port `6443` from GitHub’s network (you will see `dial tcp …:6443: i/o timeout`). The workflow therefore **SSH tunnels** `127.0.0.1:6443` on the runner to `127.0.0.1:6443` on the Chameleon node, then repoints kubeconfig at the tunnel.

Configure these repository secrets:

| Secret | Purpose |
|--------|---------|
| `KUBECONFIG_DATA` | Full kubeconfig from the node (same content as `~/.kube/config` after K3s install) |
| `K3S_SSH_HOST` | Floating IP or DNS of the node (e.g. `129.114.x.x`) |
| `K3S_SSH_USER` | SSH login (Chameleon CC images usually use `cc`) |
| `K3S_SSH_KEY` | Private key PEM used to SSH into the node (paste entire key including headers) |

Ensure on the node that `sshd` allows your key and that K3s is listening on `127.0.0.1:6443` (default K3s behavior).

### Faster runs and optional skips

- **Parallel builds:** the six images each run in their own matrix job, so wall-clock time is roughly one heavy image (usually training) instead of six steps in sequence.
- **Docker layer cache:** each image uses GitHub Actions cache (`type=gha`) so unchanged layers reuse across runs.
- **Pull requests:** images are **built but not pushed** to GHCR (saves time and avoids fork permission issues); deploy still only runs on `main`.
- **Manual workflow:** Actions → **BestShot CI/CD Pipeline** → **Run workflow**:
  - **Skip build** — only deploy (uses whatever `:latest` is already in GHCR).
  - **Skip deploy** — only build/push (no cluster changes).
- **Concurrency:** a newer push on the same branch cancels the older run so you are not waiting on obsolete jobs.

## Notes For Grading

- All operational assets are versioned in Git (code, manifests, scripts, workflow).
- Secrets are injected from environment variables and are **not** committed.
- Promotion/rollback are automated workflows; humans can optionally trigger wrappers but do not need to SSH and manually mutate deployment state.
