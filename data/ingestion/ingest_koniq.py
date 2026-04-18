import os
import subprocess
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

BUCKET = 'ak12754-data-proj19'

# Step 1 — Upload scores CSV to object storage
print("Uploading KonIQ-10k scores CSV to object storage...")
with open('/tmp/koniq10k_scores.csv', 'rb') as f:
    conn.put_object(BUCKET, 'koniq10k/koniq10k_scores_and_distributions.csv', f)

print("Scores CSV uploaded successfully!")
print(f"Bucket: {BUCKET}")
print(f"Path: koniq10k/koniq10k_scores_and_distributions.csv")

# Step 2 — Run data quality checks after ingestion
print("\nRunning data quality checks on ingested data...")
result = subprocess.run(
    ['/home/cc/bestshot/venv/bin/python',
     '/home/cc/bestshot/repo/data/ingestion/quality_checks.py'],
    capture_output=True,
    text=True
)
print(result.stdout)

if result.returncode == 0:
    print("✅ Ingestion quality checks PASSED — data is ready for training")
else:
    print("❌ Ingestion quality checks FAILED — check quality report!")
    print(result.stderr)
