import os
import swiftclient
from dotenv import load_dotenv

load_dotenv('/home/cc/bestshot/.env')

# Connect to object storage
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

# Upload scores CSV to object storage
print("Uploading KonIQ-10k scores CSV to object storage...")
with open('/tmp/koniq10k_scores.csv', 'rb') as f:
    conn.put_object(BUCKET, 'koniq10k/koniq10k_scores_and_distributions.csv', f)

print("Scores CSV uploaded successfully!")
print(f"Bucket: {BUCKET}")
print(f"Path: koniq10k/koniq10k_scores_and_distributions.csv")
