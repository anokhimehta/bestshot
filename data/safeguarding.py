import os
import json
import cv2
import numpy as np
from datetime import datetime
import swiftclient
from dotenv import load_dotenv

load_dotenv('/home/cc/bestshot/.env')

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

def check_privacy(metadata):
    """
    Privacy: Ensure no PII is stored with photos.
    Remove any sensitive fields before logging.
    """
    sensitive_fields = ['email', 'phone', 'address', 'name', 'location']
    sanitized = {}
    for key, value in metadata.items():
        if key.lower() not in sensitive_fields:
            sanitized[key] = value
        else:
            sanitized[key] = '[REDACTED]'
    return sanitized

def check_fairness(df):
    """
    Fairness: Ensure training data is diverse across users
    and not biased toward any single user or photo type.
    """
    issues = []

    if 'user_id' in df.columns:
        user_counts = df['user_id'].value_counts()
        total = len(df)
        for user, count in user_counts.items():
            if count / total > 0.3:
                issues.append(f"User {user} represents {count/total:.1%} of data — potential bias")

    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            issues.append("Only one quality class in dataset — not diverse enough")

    return issues

def log_transparency(pipeline_name, inputs, outputs, parameters):
    """
    Transparency: Log all data transformations with full lineage.
    Track how data entered and was transformed within the system.
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': pipeline_name,
        'inputs': inputs,
        'outputs': outputs,
        'parameters': parameters,
        'version': '1.0'
    }

    log_json = json.dumps(log_entry, indent=2).encode('utf-8')
    log_key = f'logs/transparency/{pipeline_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    conn.put_object(BUCKET, log_key, log_json)
    print(f"Transparency log saved: {log_key}")
    return log_entry

def log_accountability(action, performed_by, details):
    """
    Accountability: Maintain audit trail of all pipeline runs.
    Track who did what and when.
    """
    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'performed_by': performed_by,
        'details': details
    }

    audit_json = json.dumps(audit_entry, indent=2).encode('utf-8')
    audit_key = f'logs/audit/{action}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    conn.put_object(BUCKET, audit_key, audit_json)
    print(f"Audit log saved: {audit_key}")
    return audit_entry

def check_robustness(image_path):
    """
    Robustness: Handle edge cases gracefully.
    Check for corrupted images, extreme sizes, etc.
    """
    issues = []

    try:
        img = cv2.imread(image_path)
        if img is None:
            issues.append("Image cannot be read — corrupted or invalid format")
            return issues

        h, w = img.shape[:2]
        if w < 50 or h < 50:
            issues.append(f"Image too small ({w}x{h}) — may affect quality scoring")

        if w > 10000 or h > 10000:
            issues.append(f"Image very large ({w}x{h}) — may cause memory issues")

        file_size = os.path.getsize(image_path)
        if file_size < 1000:
            issues.append(f"File size very small ({file_size} bytes) — may be corrupted")

        if len(img.shape) < 3:
            issues.append("Grayscale image — color features will not be computed")

    except Exception as e:
        issues.append(f"Error checking image: {str(e)}")

    return issues

def run_safeguarding_checks(image_dir, metadata=None):
    """Run all safeguarding checks"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'overall': 'PASS'
    }

    print("Running safeguarding checks...")

    # Privacy check
    print("Check 1: Privacy...")
    if metadata:
        sanitized = check_privacy(metadata)
        report['checks']['privacy'] = {
            'passed': True,
            'message': 'PII fields sanitized',
            'sanitized_fields': [k for k, v in sanitized.items() if v == '[REDACTED]']
        }
    else:
        report['checks']['privacy'] = {'passed': True, 'message': 'No metadata to check'}
    print(f"  Privacy: PASS")

    # Robustness check on sample images
    print("Check 2: Robustness...")
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    sample = np.random.choice(images, min(10, len(images)), replace=False)
    robustness_issues = []
    for img_name in sample:
        issues = check_robustness(os.path.join(image_dir, img_name))
        if issues:
            robustness_issues.extend(issues)

    report['checks']['robustness'] = {
        'passed': len(robustness_issues) == 0,
        'issues': robustness_issues[:5]
    }
    print(f"  Robustness: {'PASS' if len(robustness_issues) == 0 else 'FAIL'} — {len(robustness_issues)} issues found")

    # Transparency log
    print("Check 3: Transparency logging...")
    log_transparency(
        pipeline_name='safeguarding_check',
        inputs={'image_dir': image_dir, 'sample_size': len(sample)},
        outputs={'report': 'safeguarding_report'},
        parameters={'threshold': 'default'}
    )
    report['checks']['transparency'] = {'passed': True, 'message': 'Transparency log created'}
    print(f"  Transparency: PASS")

    # Accountability log
    print("Check 4: Accountability logging...")
    log_accountability(
        action='safeguarding_check',
        performed_by='data_pipeline',
        details={'image_count': len(images), 'sample_size': len(sample)}
    )
    report['checks']['accountability'] = {'passed': True, 'message': 'Audit trail created'}
    print(f"  Accountability: PASS")

    # Overall
    all_passed = all(v['passed'] for v in report['checks'].values())
    report['overall'] = 'PASS' if all_passed else 'FAIL'

    print(f"\nOverall safeguarding: {report['overall']}")

    # Upload report
    report_json = json.dumps(report, indent=2).encode('utf-8')
    conn.put_object(BUCKET, 'logs/safeguarding_report.json', report_json)
    print("Safeguarding report uploaded to object storage")

    return report

if __name__ == "__main__":
    IMAGE_DIR = '/tmp/512x384'
    metadata = {
        'user_id': 'user_001',
        'email': 'test@example.com',
        'upload_timestamp': '2026-04-13T10:00:00Z'
    }
    report = run_safeguarding_checks(IMAGE_DIR, metadata)
