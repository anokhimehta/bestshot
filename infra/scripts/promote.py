#!/usr/bin/env python3
"""
promote.py
Checks model in Staging, promotes to Production if quality gates pass.
Rolls back to previous Production version if current Production is failing.
Run by: kubectl create job --from=cronjob/bestshot-retrain promote-manual -n bestshot-platform
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "bestshot-iqa"
PLCC_THRESHOLD = 0.85
SRCC_THRESHOLD = 0.83

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

def get_latest_version(stage):
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    return versions[0] if versions else None

def get_run_metrics(version):
    run = client.get_run(version.run_id)
    return run.data.metrics

def promote_staging_to_production():
    staging = get_latest_version("Staging")
    if not staging:
        print("No model in Staging — nothing to promote")
        return False

    metrics = get_run_metrics(staging)
    plcc = metrics.get("plcc", 0)
    srcc = metrics.get("srcc", 0)

    print(f"Staging model v{staging.version}: PLCC={plcc:.3f}, SRCC={srcc:.3f}")
    print(f"Thresholds: PLCC>={PLCC_THRESHOLD}, SRCC>={SRCC_THRESHOLD}")

    if plcc >= PLCC_THRESHOLD and srcc >= SRCC_THRESHOLD:
        # Archive current production
        current_prod = get_latest_version("Production")
        if current_prod:
            client.transition_model_version_stage(
                MODEL_NAME, current_prod.version, "Archived"
            )
            print(f"Archived old production version {current_prod.version}")

        # Promote staging to production
        client.transition_model_version_stage(
            MODEL_NAME, staging.version, "Production"
        )
        print(f"✅ Promoted version {staging.version} to Production")
        return True
    else:
        print(f"❌ Model failed quality gates — not promoted")
        return False

def rollback_production():
    """Roll back to previous archived version if production is failing."""
    archived = client.search_model_versions(
        f"name='{MODEL_NAME}' and current_stage='Archived'"
    )
    if not archived:
        print("No archived versions to roll back to")
        return False

    # Get most recent archived
    latest_archived = sorted(archived, key=lambda v: int(v.version), reverse=True)[0]

    # Archive current production
    current_prod = get_latest_version("Production")
    if current_prod:
        client.transition_model_version_stage(
            MODEL_NAME, current_prod.version, "Archived"
        )
        print(f"Archived failing production version {current_prod.version}")

    # Restore previous
    client.transition_model_version_stage(
        MODEL_NAME, latest_archived.version, "Production"
    )
    print(f"✅ Rolled back to version {latest_archived.version}")
    return True

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "promote"

    if action == "promote":
        success = promote_staging_to_production()
        sys.exit(0 if success else 1)
    elif action == "rollback":
        success = rollback_production()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)