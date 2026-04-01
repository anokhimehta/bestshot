import os
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
IMAGE_DIR = '/tmp/koniq_images/512x384'

images = os.listdir(IMAGE_DIR)
total = len(images)
print(f"Found {total} images to upload...")

for i, filename in enumerate(images):
    if filename.endswith('.jpg'):
        filepath = os.path.join(IMAGE_DIR, filename)
        with open(filepath, 'rb') as f:
            conn.put_object(BUCKET, f'koniq10k/images/{filename}', f)
        if i % 100 == 0:
            print(f"Uploaded {i}/{total} images...")

print("All images uploaded successfully!")
