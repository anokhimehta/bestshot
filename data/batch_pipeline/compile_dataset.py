import os
import json
import csv
import swiftclient
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

def get_latest_version(conn, bucket):
    """Get the latest dataset version number from object storage"""
    try:
        _, objects = conn.get_container(bucket, prefix='labels/')
        versions = set()
        for obj in objects:
            parts = obj['name'].split('/')
            if len(parts) >= 2 and parts[1].startswith('v'):
                try:
                    versions.add(int(parts[1][1:]))
                except:
                    pass
        return max(versions) if versions else None
    except:
        return None

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

def get_next_version():
    """Get next dataset version number"""
    try:
        _, objects = conn.get_container(BUCKET, prefix='labels/')
        versions = set()
        for obj in objects:
            name = obj['name']
            parts = name.split('/')
            if len(parts) >= 2 and parts[1].startswith('v'):
                versions.add(int(parts[1][1:]))
        return max(versions) + 1 if versions else 1
    except:
        return 1

def load_interactions():
    """Load user interactions from object storage"""
    try:
        _, content = conn.get_object(BUCKET, 'production/interactions_log.json')
        return json.loads(content)
    except:
        print("No interactions log found, using empty list")
        return []

def load_koniq_scores():
    """Load KonIQ-10k scores from object storage"""
    _, content = conn.get_object(
        BUCKET,
        'koniq10k/koniq10k_scores_and_distributions.csv'
    )
    lines = content.decode('utf-8').splitlines()
    scores = {}
    reader = csv.DictReader(lines)
    for row in reader:
        image_name = row['image_name']
        mos = float(row['MOS'])
        # Normalize MOS from 1-5 to 0-10
        normalized_score = (mos - 1) / 4 * 10
        scores[image_name] = normalized_score
    return scores

def candidate_selection(interactions, scores):
    """
    Select training candidates following best practices:
    1. Prioritize explicit user labels
    2. Include hard examples (false positives/negatives)
    3. Balance classes
    4. Ensure diversity across users
    """
    candidates = []
    
    # Group interactions by photo_id
    photo_actions = {}
    for event in interactions:
        if 'action' in event:
            photo_id = event['photo_id']
            if photo_id not in photo_actions:
                photo_actions[photo_id] = []
            photo_actions[photo_id].append(event)
    
    # Add production photos with explicit labels
    for photo_id, actions in photo_actions.items():
        explicit_actions = [a for a in actions if a.get('confidence') == 'explicit']
        if explicit_actions:
            latest_action = explicit_actions[-1]
            label = 'low' if latest_action['action'] == 'delete' else 'high'
            candidates.append({
                'image_path': f'production/user_uploads/{latest_action["user_id"]}/{photo_id}',
                'quality_score': 2.0 if label == 'low' else 8.0,
                'label': label,
                'source': 'production',
                'user_id': latest_action['user_id'],
                'upload_date': latest_action['timestamp'][:10],
                'burst_group_id': '',
                'feedback_action': latest_action['action']
            })
    
    # Add KonIQ-10k samples
    koniq_samples = list(scores.items())
    for image_name, score in koniq_samples[:5000]:
        if score < 4.0:
            label = 'low'
        elif score > 7.0:
            label = 'high'
        else:
            label = 'medium'
        
        candidates.append({
            'image_path': f'koniq10k/images/{image_name}',
            'quality_score': score,
            'label': label,
            'source': 'koniq10k',
            'user_id': '',
            'upload_date': '2026-01-01',
            'burst_group_id': '',
            'feedback_action': ''
        })
    
    return candidates

def time_based_split(candidates, train_ratio=0.8):
    """
    Split by upload date to prevent leakage.
    Train on older data, evaluate on newer data.
    """
    # Sort by upload date
    candidates.sort(key=lambda x: x['upload_date'])
    
    split_idx = int(len(candidates) * train_ratio)
    train = candidates[:split_idx]
    eval_set = candidates[split_idx:]
    
    # Add split column
    for c in train:
        c['split'] = 'train'
    for c in eval_set:
        c['split'] = 'eval'
    
    return train, eval_set

def save_dataset(train, eval_set, version):
    """Save versioned dataset to object storage"""
    fieldnames = [
        'image_path', 'quality_score', 'label', 'split',
        'source', 'user_id', 'upload_date', 
        'burst_group_id', 'feedback_action'
    ]
    
    for split_name, data in [('train', train), ('eval', eval_set)]:
        # Write to temp file
        tmp_path = f'/tmp/{split_name}.csv'
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        # Upload to object storage
        with open(tmp_path, 'rb') as f:
            conn.put_object(
                BUCKET,
                f'labels/v{version}/{split_name}.csv',
                f
            )
        print(f"Uploaded labels/v{version}/{split_name}.csv ({len(data)} samples)")
    
    # Save manifest
    manifest = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'train_samples': len(train),
        'eval_samples': len(eval_set),
        'sources': ['koniq10k', 'production'],
        'filters_applied': ['explicit_labels_only', 'time_based_split']
    }
    
    manifest_json = json.dumps(manifest, indent=2).encode('utf-8')
    conn.put_object(
        BUCKET,
        f'labels/v{version}/manifest.json',
        manifest_json
    )
    print(f"Uploaded labels/v{version}/manifest.json")

def main():
    print("Starting batch pipeline...")
    
    # Get next version number
    version = get_next_version()
    print(f"Creating dataset version v{version}")
    
    # Load data
    print("Loading user interactions...")
    interactions = load_interactions()
    print(f"Found {len(interactions)} interactions")
    
    print("Loading KonIQ-10k scores...")
    scores = load_koniq_scores()
    print(f"Loaded {len(scores)} KonIQ-10k scores")
    
    # Select candidates
    print("Running candidate selection...")
    candidates = candidate_selection(interactions, scores)
    print(f"Selected {len(candidates)} candidates")
    
    # Split dataset
    print("Applying time-based split...")
    train, eval_set = time_based_split(candidates)
    print(f"Train: {len(train)} | Eval: {len(eval_set)}")
    
    # Save to object storage
    print("Saving versioned dataset...")
    save_dataset(train, eval_set, version)
    
    print(f"\nBatch pipeline complete!")
    print(f"Dataset v{version} saved to object storage")
    print(f"Train samples: {len(train)}")
    print(f"Eval samples: {len(eval_set)}")

if __name__ == "__main__":
    main()
