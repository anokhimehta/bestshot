import os
import random
import time
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import swiftclient

load_dotenv('/home/cc/bestshot/.env')

# Immich configuration
IMMICH_URL = os.environ.get('IMMICH_URL', 'http://129.114.27.10:30283')
IMMICH_API_KEY = os.environ.get('IMMICH_API_KEY', 'd76f56yfIR4xekGvAEGntgDZerSnczlBra6dG6og')
HEADERS = {'x-api-key': IMMICH_API_KEY}

# Object storage configuration
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
IMAGE_DIR = '/tmp/512x384'
LOG_FILE = '/tmp/user_interactions.json'

USERS = [f'user_{i:03d}' for i in range(1, 11)]

def check_immich_health():
    """Check if Immich is accessible"""
    try:
        response = requests.get(f'{IMMICH_URL}/api/server/ping', timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_photo_to_immich(image_path, filename):
    """Upload a photo to Immich via API"""
    try:
        with open(image_path, 'rb') as f:
            files = {'assetData': (filename, f, 'image/jpeg')}
            data = {
                'deviceAssetId': f'bestshot-{filename}-{int(time.time())}',
                'deviceId': 'bestshot-generator',
                'fileCreatedAt': datetime.now().isoformat(),
                'fileModifiedAt': datetime.now().isoformat(),
            }
            response = requests.post(
                f'{IMMICH_URL}/api/assets',
                headers=HEADERS,
                files=files,
                data=data,
                timeout=30
            )
            if response.status_code in [200, 201]:
                asset_id = response.json().get('id')
                return asset_id
            else:
                print(f"  Upload failed: {response.status_code}")
                return None
    except Exception as e:
        print(f"  Upload error: {e}")
        return None

def simulate_feedback(asset_id, action):
    """Simulate user feedback on a photo in Immich"""
    try:
        if action == 'favorite':
            response = requests.put(
                f'{IMMICH_URL}/api/assets/{asset_id}',
                headers={**HEADERS, 'Content-Type': 'application/json'},
                json={'isFavorite': True},
                timeout=10
            )
            return response.status_code in [200, 201]
        elif action == 'delete':
            response = requests.delete(
                f'{IMMICH_URL}/api/assets',
                headers={**HEADERS, 'Content-Type': 'application/json'},
                json={'ids': [asset_id]},
                timeout=10
            )
            return response.status_code in [200, 204]
        else:
            return True
    except Exception as e:
        print(f"  Feedback error: {e}")
        return False

def main():
    print(f"Starting simulation with {len(USERS)} users...")
    print(f"Immich URL: {IMMICH_URL}")

    # Check Immich health
    if not check_immich_health():
        print("WARNING: Immich is not accessible! Continuing anyway...")
    else:
        print("Immich is healthy!")

    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    interactions = []

    print("Press Ctrl+C to stop\n")

    round_num = 0
    while True:
        round_num += 1
        print(f"Round {round_num} — simulating user activity...")

        for user_id in USERS:
            num_uploads = random.randint(1, 3)
            selected = random.sample(images, num_uploads)

            for image_name in selected:
                image_path = os.path.join(IMAGE_DIR, image_name)

                # Upload to Immich
                asset_id = upload_photo_to_immich(image_path, image_name)

                if asset_id:
                    print(f"  {user_id} uploaded {image_name} → asset_id: {asset_id}")
                else:
                    print(f"  {user_id} uploaded {image_name} (no asset_id)")
                    asset_id = f'local_{image_name}'

                upload_event = {
                    'request_id': f'req_{datetime.now().strftime("%Y%m%d%H%M%S")}_{image_name}',
                    'photo_id': image_name,
                    'asset_id': asset_id,
                    'user_id': user_id,
                    'upload_timestamp': datetime.now().isoformat(),
                    'storage_path': f'production/user_uploads/{user_id}/{image_name}',
                    'resolution': '512x384',
                    'file_type': 'jpg',
                }
                interactions.append(upload_event)

                time.sleep(0.5)

                # Simulate feedback
                action = random.choices(
                    ['keep', 'delete', 'favorite'],
                    weights=[0.5, 0.3, 0.2]
                )[0]

                if asset_id and not asset_id.startswith('local_'):
                    simulate_feedback(asset_id, action)

                print(f"  {user_id} action: {action} on {image_name}")

                feedback_event = {
                    'event_id': f'evt_{datetime.now().strftime("%Y%m%d%H%M%S")}_{image_name}',
                    'photo_id': image_name,
                    'asset_id': asset_id,
                    'user_id': user_id,
                    'action': action,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 'explicit'
                }
                interactions.append(feedback_event)

        # Save interactions log
        with open(LOG_FILE, 'w') as f:
            json.dump(interactions, f, indent=2)

        # Upload to object storage
        with open(LOG_FILE, 'rb') as f:
            conn.put_object(BUCKET, 'production/interactions_log.json', f)

        print(f"\nRound {round_num} complete. Total interactions: {len(interactions)}")
        print("Waiting 30 seconds before next round...\n")
        time.sleep(30)

if __name__ == "__main__":
    main()
