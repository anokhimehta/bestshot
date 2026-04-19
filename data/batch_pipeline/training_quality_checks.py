import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import swiftclient
from dotenv import load_dotenv

load_dotenv()

conn = swiftclient.Connection(
    auth_version='3',
    authurl=os.environ['OS_AUTH_URL'],
    os_options={
        'application_credential_id': os.environ['OS_APPLICATION_CREDENTIAL_ID'],
        'application_credential_secret': os.environ['OS_APPLICATION_CREDENTIAL_SECRET'],
        'region_name': os.environ['OS_REGION_NAME'],
        'auth_type': 'v3applicationcredential'
    }
)

BUCKET = os.environ.get('BUCKET_NAME', 'ak12754-data-proj19')
MIN_TRAIN_SAMPLES = 500
MIN_EVAL_SAMPLES = 100
MAX_CLASS_IMBALANCE = 4.0

def check_minimum_samples(df, split):
    min_required = MIN_TRAIN_SAMPLES if split == 'train' else MIN_EVAL_SAMPLES
    count = len(df)
    if count < min_required:
        return False, f"Only {count} samples, minimum required is {min_required}"
    return True, f"{count} samples — sufficient"

def check_class_balance(df):
    if 'label' not in df.columns:
        return False, "No label column found"
    counts = df['label'].value_counts()
    if len(counts) < 2:
        return False, f"Only {len(counts)} class found — need at least 2"
    max_count = counts.max()
    min_count = counts.min()
    if min_count == 0:
        return False, "One class has zero samples"
    imbalance_ratio = max_count / min_count
    if imbalance_ratio > MAX_CLASS_IMBALANCE:
        return False, f"Class imbalance ratio {imbalance_ratio:.2f} exceeds threshold {MAX_CLASS_IMBALANCE}"
    return True, f"Class distribution: {counts.to_dict()}"

def check_user_diversity(df):
    if 'user_id' not in df.columns:
        return True, "No user_id column — skipping diversity check"
    user_counts = df['user_id'].value_counts()
    total = len(df)
    max_user_pct = user_counts.max() / total * 100
    if max_user_pct > 50:
        return False, f"Single user represents {max_user_pct:.1f}% of dataset — too dominant"
    return True, f"{len(user_counts)} unique users, max contribution {max_user_pct:.1f}%"

def check_score_distribution(df):
    if 'quality_score' not in df.columns:
        return False, "No quality_score column found"
    scores = df['quality_score'].dropna()
    if scores.min() < 0 or scores.max() > 10:
        return False, f"Scores out of range: min={scores.min():.2f}, max={scores.max():.2f}"
    mean = scores.mean()
    std = scores.std()
    return True, f"Score distribution: mean={mean:.2f}, std={std:.2f}, min={scores.min():.2f}, max={scores.max():.2f}"

def check_no_leakage(train_df, eval_df):
    train_paths = set(train_df['image_path'].tolist())
    eval_paths = set(eval_df['image_path'].tolist())
    overlap = train_paths.intersection(eval_paths)
    if overlap:
        return False, f"Found {len(overlap)} images in both train and eval sets — data leakage!"
    return True, "No overlap between train and eval sets"

def run_training_quality_checks(version):
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': version,
        'checks': {},
        'summary': {
            'overall': 'PASS',
            'passed': 0,
            'failed': 0
        }
    }

    print(f"Running training quality checks for dataset v{version}...")

    print("Downloading train.csv and eval.csv...")
    _, train_content = conn.get_object(BUCKET, f'labels/v{version}/train.csv')
    _, eval_content = conn.get_object(BUCKET, f'labels/v{version}/eval.csv')

    with open('/tmp/train.csv', 'wb') as f:
        f.write(train_content)
    with open('/tmp/eval.csv', 'wb') as f:
        f.write(eval_content)

    train_df = pd.read_csv('/tmp/train.csv')
    eval_df = pd.read_csv('/tmp/eval.csv')

    print(f"Train samples: {len(train_df)}, Eval samples: {len(eval_df)}")

    print("Check 1: Minimum sample size...")
    passed, message = check_minimum_samples(train_df, 'train')
    report['checks']['train_minimum_samples'] = {'passed': passed, 'message': message}
    print(f"  Train samples: {'PASS' if passed else 'FAIL'} — {message}")

    passed, message = check_minimum_samples(eval_df, 'eval')
    report['checks']['eval_minimum_samples'] = {'passed': passed, 'message': message}
    print(f"  Eval samples: {'PASS' if passed else 'FAIL'} — {message}")

    print("Check 2: Class balance...")
    passed, message = check_class_balance(train_df)
    report['checks']['class_balance'] = {'passed': passed, 'message': message}
    print(f"  Class balance: {'PASS' if passed else 'FAIL'} — {message}")

    print("Check 3: User diversity...")
    passed, message = check_user_diversity(train_df)
    report['checks']['user_diversity'] = {'passed': passed, 'message': message}
    print(f"  User diversity: {'PASS' if passed else 'FAIL'} — {message}")

    print("Check 4: Score distribution...")
    passed, message = check_score_distribution(train_df)
    report['checks']['score_distribution'] = {'passed': passed, 'message': message}
    print(f"  Score distribution: {'PASS' if passed else 'FAIL'} — {message}")

    print("Check 5: Data leakage check...")
    passed, message = check_no_leakage(train_df, eval_df)
    report['checks']['no_leakage'] = {'passed': passed, 'message': message}
    print(f"  No leakage: {'PASS' if passed else 'FAIL'} — {message}")

    all_checks = [v['passed'] for v in report['checks'].values()]
    report['summary']['passed'] = sum(all_checks)
    report['summary']['failed'] = len(all_checks) - sum(all_checks)
    report['summary']['overall'] = 'PASS' if all(all_checks) else 'FAIL'

    print(f"\nSummary: {report['summary']['passed']}/{len(all_checks)} checks passed")
    print(f"Overall: {report['summary']['overall']}")

    if report['summary']['overall'] == 'FAIL':
        print("WARNING: Training quality checks FAILED — retraining should NOT be triggered!")
    else:
        print("SUCCESS: Training quality checks PASSED — safe to trigger retraining!")

    report_json = json.dumps(report, indent=2).encode('utf-8')
    conn.put_object(
        BUCKET,
        f'labels/quality_reports/training_quality_report_v{version}.json',
        report_json
    )
    print(f"Report uploaded to: labels/quality_reports/training_quality_report_v{version}.json")

    return report

if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "1"
    report = run_training_quality_checks(version)
    if report['summary']['overall'] == 'FAIL':
        sys.exit(1)
    else:
        sys.exit(0)
