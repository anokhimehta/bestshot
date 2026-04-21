# Runbook

## Quick Recovery Checklist

Run this sequence on the CHI@TACC node to recreate or refresh the environment.

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
```

Optional (generate fresh interaction events):

```bash
source venv/bin/activate
python data/generator/simulate_users.py
```
