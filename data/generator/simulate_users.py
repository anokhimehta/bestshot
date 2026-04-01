import os
import random
import time
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import swiftclient

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
IMAGE_DIR = '/tmp/koniq_images/512x384'
LOG_FILE = '/tmp/user_interactions.json'

# Simulated users
USERS = [f'user_{i:03d}' for i in range(1, 11)]

def simulate_upload(user_id, image_path):
    """Simulate a user uploading a photo"""
    filename = os.path.basename(image_path)
    storage_path = f'production/user_uploads/{user_id}/{filename}'
    
    with open(image_path, 'rb') as f:
        conn.put_object(BUCKET, storage_path, f)
    
    event = {
        'request_id': f'req_{datetime.now().strftime("%Y%m%d%H%M%S")}_{filename}',
        'photo_id': filename,
        'user_id': user_id,
        'upload_timestamp': datetime.now().isoformat(),
        'storage_path': storage_path,
        'resolution': '512x384',
        'file_type': 'jpg',
        'exif': {
            'capture_timestamp': datetime.now().isoformat(),
            'camera_model': random.choice(['iPhone 14', 'Samsung S23', 'Pixel 7'])
        }
    }
    return event

def simulate_feedback(photo_id, user_id):
    """Simulate user feedback on a photo"""
    action = random.choices(
        ['keep', 'delete', 'favorite'],
        weights=[0.5, 0.3, 0.2]
    )[0]
    
    return {
        'event_id': f'evt_{datetime.now().strftime("%Y%m%d%H%M%S")}_{photo_id}',
        'photo_id': photo_id,
        'user_id': user_id,
        'action': action,
        'timestamp': datetime.now().isoformat(),
        'confidence': 'explicit'
    }

def main():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    interactions = []
    
    print(f"Starting simulation with {len(USERS)} users...")
    print("Press Ctrl+C to stop\n")
    
    round_num = 0
    while True:
        round_num += 1
        print(f"Round {round_num} — simulating user activity...")
        
        for user_id in USERS:
            # Each user uploads 1-3 photos
            num_uploads = random.randint(1, 3)
            selected = random.sample(images, num_uploads)
            
            for image_name in selected:
                image_path = os.path.join(IMAGE_DIR, image_name)
                
                # Simulate upload
                upload_event = simulate_upload(user_id, image_path)
                interactions.append(upload_event)
                print(f"  {user_id} uploaded {image_name}")
                
                # Simulate feedback after short delay
                time.sleep(0.5)
                feedback_event = simulate_feedback(image_name, user_id)
                interactions.append(feedback_event)
                print(f"  {user_id} action: {feedback_event['action']} on {image_name}")
        
        # Save interactions log
        with open(LOG_FILE, 'w') as f:
            json.dump(interactions, f, indent=2)
        
        # Upload interactions log to object storage
        with open(LOG_FILE, 'rb') as f:
            conn.put_object(
                BUCKET,
                'production/interactions_log.json',
                f
            )
        
        print(f"\nRound {round_num} complete. Total interactions: {len(interactions)}")
        print("Waiting 30 seconds before next round...\n")
        time.sleep(30)

if __name__ == "__main__":
    main()
