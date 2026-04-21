"""
BestShot retrain.py

Checks whether retraining should be triggered based on:
  1. Number of new feedback events since last retrain (count-based)
  2. Days elapsed since last retrain (schedule-based)

If either threshold is met:
  1. Runs compile_dataset.py to produce a fresh versioned dataset
  2. Runs training_quality_checks.py to validate the dataset
  3. Triggers a Kubernetes training Job
  4. Saves a new last_retrain timestamp to Swift

Runs as a Kubernetes CronJob (see infra/k8s/jobs/retrain-cronjob.yaml)
Schedule: daily at 2am

Usage (manual):
    python retrain.py
    python retrain.py --force   # skip threshold checks, always retrain
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
import swiftclient

load_dotenv('.env')

UPLOAD_THRESHOLD = 500
DAYS_THRESHOLD = 7
NEGATIVE_FEEDBACK_THRESHOLD = 0.40
MIN_FEEDBACK_SAMPLE = 50

BUCKET = os.environ.get('BUCKET_NAME')
if not BUCKET:
    raise ValueError("BUCKET_NAME environment variable not set")
INTERACTIONS_LOG_KEY = 'interactions_log.jsonl'
LAST_RETRAIN_KEY = 'training/last_retrain.json'

def get_swift_conn():
    return swiftclient.Connection(
        auth_version='3',
        authurl=os.environ['OS_AUTH_URL'],
        os_options={
            'application_credential_id': os.environ['OS_APPLICATION_CREDENTIAL_ID'],
            'application_credential_secret': os.environ['OS_APPLICATION_CREDENTIAL_SECRET'],
            'region_name': os.environ['OS_REGION_NAME'],
            'auth_type': 'v3applicationcredential'
        }
    )

#LAST RETRAINING TIMESTAMP MANAGEMENT
def get_last_retrain_time(conn):
    """
    Load the timestamp of the last successful retrain from Swift.
    Returns datetime.min if no previous retrain exists (forces first run).
    """
    try:
        _, content = conn.get_object(BUCKET, LAST_RETRAIN_KEY)
        data = json.loads(content)
        return datetime.fromisoformat(data['last_retrain'])
    except Exception:
        print("No previous retrain timestamp found — treating as first run")
        return datetime.min


def save_retrain_timestamp(conn):
    """Save current timestamp as last retrain time to Swift."""
    data = json.dumps({
        "last_retrain": datetime.now().isoformat(),
        "triggered_by": "retrain.py"
    }).encode('utf-8')
    conn.put_object(BUCKET, LAST_RETRAIN_KEY, data)
    print(f"Saved retrain timestamp to Swift: {LAST_RETRAIN_KEY}")


#FEEDBACK EVENTS COUNT
def get_new_upload_count(interactions, since: datetime) -> int:
    """
    Count all events (uploads + feedback) since last retrain.
    Used as a proxy for overall system activity — triggers retraining
    once enough new data has accumulated.
    """
    count = 0
    for event in interactions:
        try:
            event_time = datetime.fromisoformat(event['timestamp'])
            if event_time > since:
                count += 1
        except (KeyError, ValueError):
            continue
    return count

def get_negative_feedback_rate(interactions, since):
    """
    Compute fraction of feedback events that are negative since last retrain.

    Negative feedback depends on which feature triggered it:
    - delete_suggestion: keep or favorite = model flagged a good photo (false positive)
    - best_shot: delete = model recommended the wrong photo as best
    """
    feedback = [
        e for e in interactions
        if 'action' in e
        and 'feature' in e
        and datetime.fromisoformat(e.get('timestamp', '1970-01-01')) > since
    ]

    if len(feedback) < MIN_FEEDBACK_SAMPLE:
        return 0.0, len(feedback)

    negative = []
    for e in feedback:
        feature = e['feature']
        action = e['action']
        if feature == 'delete_suggestion' and action in ('keep', 'favorite'):
            negative.append(e)
        elif feature == 'best_shot' and action == 'delete':
            negative.append(e)

    return len(negative) / len(feedback), len(feedback)


#THRESHOLD CHECKS
def should_retrain(conn, force: bool = False) -> tuple[bool, str]:
    """
    Returns (should_retrain: bool, reason: str)
    """
    if force:
        return True, "forced via --force flag"

    last_retrain = get_last_retrain_time(conn)

    # load once, reuse for all checks
    try:
        _, content = conn.get_object(BUCKET, INTERACTIONS_LOG_KEY)
        interactions = [
            json.loads(line) 
            for line in content.decode().splitlines() 
            if line.strip()
        ]
    except Exception:
        print("No interactions log found")
        interactions = []

    days_since = (datetime.now() - last_retrain).days
    upload_count = get_new_upload_count(interactions, since=last_retrain)
    negative_rate, feedback_sample = get_negative_feedback_rate(interactions, since=last_retrain)

    print(f"Last retrain: {last_retrain}")
    print(f"Days since last retrain: {days_since}")
    print(f"New events since retrain: {upload_count} (threshold: {UPLOAD_THRESHOLD})")
    print(f"Feedback sample size:     {feedback_sample} (min: {MIN_FEEDBACK_SAMPLE})")
    print(f"Negative feedback rate:   {negative_rate:.1%} (threshold: {NEGATIVE_FEEDBACK_THRESHOLD:.1%})")

    if feedback_sample >= MIN_FEEDBACK_SAMPLE and negative_rate >= NEGATIVE_FEEDBACK_THRESHOLD:
        return True, f"high negative feedback rate ({negative_rate:.1%} over {feedback_sample} events)"
    if upload_count >= UPLOAD_THRESHOLD:
        return True, f"upload threshold met ({upload_count} >= {UPLOAD_THRESHOLD})"
    if days_since >= DAYS_THRESHOLD:
        return True, f"schedule threshold met ({days_since} >= {DAYS_THRESHOLD} days)"

    return False, (
        f"no threshold met — "
        f"uploads={upload_count}/{UPLOAD_THRESHOLD}, "
        f"days={days_since}/{DAYS_THRESHOLD}, "
        f"negative_rate={negative_rate:.1%}/{NEGATIVE_FEEDBACK_THRESHOLD:.1%}"
    )

#PIPELINE EXECUTION
def run_compile_dataset():
    """
    Run compile_dataset.py to produce a fresh versioned dataset in Swift.
    Exits with non-zero code on failure.
    """
    print("\n--- Step 1: Running compile_dataset.py ---")
    result = subprocess.run(
        [sys.executable, "data/batch_pipeline/compile_dataset.py"],
        capture_output=False  # stream output directly
    )
    if result.returncode != 0:
        raise RuntimeError(f"compile_dataset.py failed with exit code {result.returncode}")
    print("compile_dataset.py completed successfully")


def trigger_training_job():
    """
    Launch a new Kubernetes training Job from the training CronJob template.
    The training Job will call train.py which automatically picks up the
    latest dataset version from Swift.
    """
    print("\n--- Step 3: Triggering training Job ---")

    result = subprocess.run([
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
        "--network", "host",
        "--shm-size=8g",
        "--env-file", "/home/cc/bestshot/training/.env",
        "ghcr.io/anokhimehta/bestshot-training:latest",
        "python", "training/train.py", "--config", "training/config/partial_finetune_highlr.yaml"
    ])
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed with exit code {result.returncode}")
    print("Training job completed successfully")
    
    # job_name = f"bestshot-training-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # create_result = subprocess.run(
    #     [
    #         "kubectl", "create", "job", job_name,
    #         "--from=cronjob/bestshot-training",
    #         "-n", "bestshot-platform"
    #     ],
    #     capture_output=True,
    #     text=True
    # )

    # if create_result.returncode != 0:
    #     raise RuntimeError(f"kubectl create failed: {create_result.stderr}")

    # print(f"Launched training job: {job_name}")
    # print(create_result.stdout)
    # print("Patched job command to run train.py")
    # return job_name


def main():
    parser = argparse.ArgumentParser(description="BestShot retrain script")
    parser.add_argument(
        '--force', 
        action='store_true', 
        help="Force retraining regardless of thresholds")
    args = parser.parse_args()

    conn = get_swift_conn()

    should_retrain_flag, reason = should_retrain(conn, force=args.force)
    print(f"\nRetrain decision: {should_retrain_flag} ({reason})")

    if not should_retrain_flag:
        return

    try:
        run_compile_dataset()
        trigger_training_job()
        save_retrain_timestamp(conn)
        print("\nRetraining process completed successfully.")
    except Exception as e:
        print(f"\nERROR during retraining process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
