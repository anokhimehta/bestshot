import os
import json
import numpy as np
import pandas as pd
import cv2
import swiftclient
from datetime import datetime
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
DRIFT_THRESHOLD = 0.2

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(variance / 1000.0, 1.0)

def compute_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    dark_pixels = hist[:50].sum()
    bright_pixels = hist[200:].sum()
    if dark_pixels > 0.5:
        score = 1.0 - dark_pixels
    elif bright_pixels > 0.5:
        score = 1.0 - bright_pixels
    else:
        score = 1.0 - abs(dark_pixels - bright_pixels)
    return round(float(score), 4)

def get_training_distribution():
    """Get feature distribution from training dataset v5"""
    try:
        _, content = conn.get_object(BUCKET, 'labels/v5/train.csv')
        with open('/tmp/train_ref.csv', 'wb') as f:
            f.write(content)
        df = pd.read_csv('/tmp/train_ref.csv')
        return {
            'mean_score': float(df['quality_score'].mean()),
            'std_score': float(df['quality_score'].std()),
            'low_pct': float((df['label'] == 'low').mean()),
            'medium_pct': float((df['label'] == 'medium').mean()),
            'high_pct': float((df['label'] == 'high').mean())
        }
    except Exception as e:
        print(f"Could not load training distribution: {e}")
        return None

def get_production_features(image_dir, sample_size=100):
    """Compute features from recent production images"""
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if len(images) == 0:
        return None

    sample = np.random.choice(images, min(sample_size, len(images)), replace=False)
    sharpness_scores = []
    exposure_scores = []

    for img_name in sample:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        sharpness_scores.append(compute_sharpness(img))
        exposure_scores.append(compute_exposure(img))

    return {
        'mean_sharpness': float(np.mean(sharpness_scores)),
        'std_sharpness': float(np.std(sharpness_scores)),
        'mean_exposure': float(np.mean(exposure_scores)),
        'std_exposure': float(np.std(exposure_scores)),
        'sample_size': len(sharpness_scores)
    }

def check_score_drift(training_dist, production_features):
    """Check if production data has drifted from training distribution"""
    alerts = []

    # Check sharpness drift
    if production_features['mean_sharpness'] < 0.1:
        alerts.append({
            'type': 'sharpness_drift',
            'severity': 'HIGH',
            'message': f"Mean sharpness {production_features['mean_sharpness']:.3f} is very low — incoming photos may be blurrier than training data"
        })

    # Check exposure drift
    if production_features['mean_exposure'] < 0.3:
        alerts.append({
            'type': 'exposure_drift',
            'severity': 'HIGH',
            'message': f"Mean exposure {production_features['mean_exposure']:.3f} is low — incoming photos may be darker than training data"
        })

    # Check score distribution drift from training
    if training_dist:
        expected_mean = training_dist['mean_score']
        if abs(production_features['mean_sharpness'] * 10 - expected_mean) > DRIFT_THRESHOLD * 10:
            alerts.append({
                'type': 'score_distribution_drift',
                'severity': 'MEDIUM',
                'message': f"Production feature distribution differs significantly from training distribution"
            })

    return alerts

def run_drift_monitoring(image_dir):
    """Run drift monitoring on production data"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'OK',
        'alerts': [],
        'production_features': {},
        'training_distribution': {},
        'drift_detected': False
    }

    print("Running production drift monitoring...")

    # Get training distribution
    print("Loading training distribution...")
    training_dist = get_training_distribution()
    if training_dist:
        report['training_distribution'] = training_dist
        print(f"  Training mean score: {training_dist['mean_score']:.2f}")
        print(f"  Training label distribution: low={training_dist['low_pct']:.2%}, medium={training_dist['medium_pct']:.2%}, high={training_dist['high_pct']:.2%}")

    # Get production features
    print("Computing production features...")
    production_features = get_production_features(image_dir)
    if production_features is None:
        print("No production images found!")
        report['status'] = 'NO_DATA'
        return report

    report['production_features'] = production_features
    print(f"  Production mean sharpness: {production_features['mean_sharpness']:.3f}")
    print(f"  Production mean exposure: {production_features['mean_exposure']:.3f}")
    print(f"  Sample size: {production_features['sample_size']}")

    # Check for drift
    print("Checking for drift...")
    alerts = check_score_drift(training_dist, production_features)
    report['alerts'] = alerts
    report['drift_detected'] = len(alerts) > 0

    if alerts:
        report['status'] = 'DRIFT_DETECTED'
# Write drift alert flag to object storage
        alert_flag = {
            'drift_detected': True,
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'action_required': 'Consider retraining model'
        }
        alert_json = json.dumps(alert_flag, indent=2).encode('utf-8')
        conn.put_object(BUCKET, 'labels/drift_alert.json', alert_json)
        print("⚠️  Drift alert written to object storage: labels/drift_alert.json")
        print(f"\n⚠️  DRIFT DETECTED — {len(alerts)} alert(s):")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['message']}")
    else:
        report['status'] = 'OK'
        print("\n✅ No drift detected — production data matches training distribution")

    # Upload report to object storage
    report_json = json.dumps(report, indent=2).encode('utf-8')
    conn.put_object(
        BUCKET,
        f'labels/quality_reports/drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        report_json
    )
    print(f"\nDrift report uploaded to object storage")

    return report

if __name__ == "__main__":
    IMAGE_DIR = '/tmp/512x384'
    report = run_drift_monitoring(IMAGE_DIR)
    print(f"\nFinal status: {report['status']}")
