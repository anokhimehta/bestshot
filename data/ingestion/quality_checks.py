import os
import cv2
import numpy as np
import pandas as pd
import json
import hashlib
import swiftclient
from datetime import datetime
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

def check_image_readability(image_path):
    """Check if image can be read and is valid"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Image cannot be read"
        if img.size == 0:
            return False, "Image is empty"
        if len(img.shape) < 2:
            return False, "Invalid image dimensions"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def check_mos_score(mos):
    """Validate MOS score is in valid range 1-5"""
    try:
        mos = float(mos)
        if mos < 1.0 or mos > 5.0:
            return False, f"MOS score {mos} out of range 1-5"
        return True, "OK"
    except:
        return False, f"Invalid MOS score: {mos}"

def check_csv_schema(df):
    """Validate CSV has all required columns"""
    required_columns = ['image_name', 'MOS', 'SD']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"

def check_duplicates(df):
    """Check for duplicate image names"""
    duplicates = df[df.duplicated(['image_name'])]['image_name'].tolist()
    if duplicates:
        return False, f"Found {len(duplicates)} duplicate images"
    return True, "OK"

def compute_image_hash(image_path):
    """Compute MD5 hash of image for duplicate detection"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_ingestion_quality_checks(image_dir, scores_csv_path):
    """Run all quality checks on ingested data"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'summary': {
            'total_images': 0,
            'passed': 0,
            'failed': 0,
            'warnings': []
        }
    }

    print("Running ingestion quality checks...")

    # Check 1 — CSV Schema
    print("Check 1: Validating CSV schema...")
    df = pd.read_csv(scores_csv_path)
    passed, message = check_csv_schema(df)
    report['checks']['csv_schema'] = {'passed': passed, 'message': message}
    print(f"  CSV schema: {'PASS' if passed else 'FAIL'} — {message}")

    # Check 2 — MOS Score Range
    print("Check 2: Validating MOS scores...")
    invalid_mos = []
    for _, row in df.iterrows():
        passed, message = check_mos_score(row['MOS'])
        if not passed:
            invalid_mos.append({'image': row['image_name'], 'error': message})
    
    mos_passed = len(invalid_mos) == 0
    report['checks']['mos_scores'] = {
        'passed': mos_passed,
        'invalid_count': len(invalid_mos),
        'invalid_samples': invalid_mos[:5]
    }
    print(f"  MOS scores: {'PASS' if mos_passed else 'FAIL'} — {len(invalid_mos)} invalid scores")

    # Check 3 — Duplicates
    print("Check 3: Checking for duplicates...")
    passed, message = check_duplicates(df)
    report['checks']['duplicates'] = {'passed': passed, 'message': message}
    print(f"  Duplicates: {'PASS' if passed else 'FAIL'} — {message}")

    # Check 4 — Image Readability (sample 100 images)
    print("Check 4: Checking image readability (sampling 100 images)...")
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    sample = np.random.choice(images, min(100, len(images)), replace=False)
    
    unreadable = []
    for img_name in sample:
        img_path = os.path.join(image_dir, img_name)
        passed, message = check_image_readability(img_path)
        if not passed:
            unreadable.append({'image': img_name, 'error': message})
    
    readability_passed = len(unreadable) == 0
    report['checks']['image_readability'] = {
        'passed': readability_passed,
        'sample_size': len(sample),
        'unreadable_count': len(unreadable),
        'unreadable_samples': unreadable[:5]
    }
    print(f"  Image readability: {'PASS' if readability_passed else 'FAIL'} — {len(unreadable)}/{len(sample)} unreadable")

    # Summary
    total_images = len(images)
    all_checks = [v['passed'] for v in report['checks'].values()]
    report['summary']['total_images'] = total_images
    report['summary']['passed'] = sum(all_checks)
    report['summary']['failed'] = len(all_checks) - sum(all_checks)
    report['summary']['overall'] = 'PASS' if all(all_checks) else 'FAIL'

    print(f"\nSummary: {report['summary']['passed']}/{len(all_checks)} checks passed")
    print(f"Overall: {report['summary']['overall']}")

    # Upload report to object storage
    report_json = json.dumps(report, indent=2).encode('utf-8')
    conn.put_object(BUCKET, 'labels/quality_reports/ingestion_quality_report.json', report_json)
    print(f"\nReport uploaded to object storage: labels/quality_reports/ingestion_quality_report.json")

    return report

if __name__ == "__main__":
    IMAGE_DIR = '/tmp/512x384'
    SCORES_CSV = '/tmp/koniq10k_scores.csv'
    
    # Download scores CSV from object storage first
    print("Downloading scores CSV from object storage...")
    _, content = conn.get_object(BUCKET, 'koniq10k/koniq10k_scores_and_distributions.csv')
    with open(SCORES_CSV, 'wb') as f:
        f.write(content)
    print("Downloaded successfully!")
    
    # Run checks
    report = run_ingestion_quality_checks(IMAGE_DIR, SCORES_CSV)
