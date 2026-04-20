#!/usr/bin/env bash
set -euo pipefail

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required_env=(
  OS_AUTH_URL
  OS_APPLICATION_CREDENTIAL_ID
  OS_APPLICATION_CREDENTIAL_SECRET
  OS_REGION_NAME
  BUCKET_NAME
  IMMICH_DB_PASSWORD
  MLFLOW_TRACKING_URI
)

for var in "${required_env[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    die "Required env var missing: ${var}"
  fi
done

info "All required environment variables are present"

if ! command -v kubectl >/dev/null 2>&1; then
  info "kubectl not found yet (expected before K3s install)"
fi

if kubectl get nodes >/dev/null 2>&1; then
  info "K3s already installed, skipping install step"
else
  info "Installing K3s"
  curl -sfL https://get.k3s.io | sh -
  sleep 20
fi

sudo chmod 644 /etc/rancher/k3s/k3s.yaml
mkdir -p "${HOME}/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "${HOME}/.kube/config"
sudo chown "${USER}:${USER}" "${HOME}/.kube/config"
chmod 600 "${HOME}/.kube/config"
export KUBECONFIG="${HOME}/.kube/config"
if ! rg -q "KUBECONFIG=~/.kube/config" "${HOME}/.bashrc" 2>/dev/null; then
  echo 'export KUBECONFIG=~/.kube/config' >> "${HOME}/.bashrc"
fi

kubectl get nodes >/dev/null
info "Kubernetes is ready"

FLOATING_IP="$(curl -s ifconfig.me || true)"
if [[ -n "${FLOATING_IP}" ]]; then
  info "Configuring K3s TLS SAN for ${FLOATING_IP}"
  sudo bash -c "cat > /etc/rancher/k3s/config.yaml <<EOF
tls-san:
  - ${FLOATING_IP}
EOF"
  sudo systemctl restart k3s
  sleep 20
fi

info "Installing metrics server"
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' \
  >/dev/null 2>&1 || true

info "Creating namespaces"
kubectl apply -f "${ROOT_DIR}/infra/k8s/platform/00-namespace.yaml"
kubectl apply -f "${ROOT_DIR}/infra/k8s/environments/namespaces.yaml"

for ns in bestshot-platform bestshot-app bestshot-staging bestshot-canary bestshot-production; do
  kubectl create secret generic mlflow-credentials \
    --from-literal=MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    -n "${ns}" --dry-run=client -o yaml | kubectl apply -f -
done

for ns in bestshot-platform bestshot-staging bestshot-canary bestshot-production; do
  kubectl create secret generic openstack-credentials \
    --from-literal=OS_AUTH_URL="${OS_AUTH_URL}" \
    --from-literal=OS_AUTH_TYPE="v3applicationcredential" \
    --from-literal=OS_APPLICATION_CREDENTIAL_ID="${OS_APPLICATION_CREDENTIAL_ID}" \
    --from-literal=OS_APPLICATION_CREDENTIAL_SECRET="${OS_APPLICATION_CREDENTIAL_SECRET}" \
    --from-literal=OS_REGION_NAME="${OS_REGION_NAME}" \
    --from-literal=BUCKET_NAME="${BUCKET_NAME}" \
    -n "${ns}" --dry-run=client -o yaml | kubectl apply -f -
done

kubectl create secret generic immich-db-secret \
  --from-literal=DB_PASSWORD="${IMMICH_DB_PASSWORD}" \
  --from-literal=DB_USERNAME=immich \
  --from-literal=DB_DATABASE_NAME=immich \
  -n bestshot-app --dry-run=client -o yaml | kubectl apply -f -

info "Deploying platform, app, environments, monitoring, and jobs"
kubectl apply -f "${ROOT_DIR}/infra/k8s/platform/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/app/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/environments/staging/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/environments/canary/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/environments/production/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/monitoring/"
kubectl apply -f "${ROOT_DIR}/infra/k8s/jobs/"

info "Done. Quick checks:"
kubectl get pods --all-namespaces
kubectl get cronjobs -n bestshot-platform
