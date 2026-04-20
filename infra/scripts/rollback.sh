#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow.bestshot-platform.svc.cluster.local:5000}"

python "${ROOT_DIR}/infra/scripts/promote.py" rollback

kubectl rollout restart deployment/bestshot-serving-canary -n bestshot-canary
kubectl rollout restart deployment/bestshot-serving-production -n bestshot-production
