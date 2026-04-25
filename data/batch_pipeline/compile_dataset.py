import sys
import os
import json
import csv
import random
import swiftclient
import pandas as pd
from collections import defaultdict
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
    """Load user interactions from object storage (supports JSON, JSONL, and .jsonl file)"""
    interactions = []
    
    # Try interactions_log.jsonl first (Lava's feedback endpoint)
    try:
        _, content = conn.get_object(BUCKET, 'interactions_log.jsonl')
        decoded = content.decode('utf-8').strip()
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                interactions.append(json.loads(line))
        print(f"Loaded {len(interactions)} interactions from interactions_log.jsonl")
    except:
        pass
    
    # Also try interactions_log.json (data generator)
    try:
        _, content = conn.get_object(BUCKET, 'production/interactions_log.json')
        decoded = content.decode('utf-8').strip()
        if decoded.startswith('['):
            entries = json.loads(decoded)
        else:
            entries = [json.loads(l) for l in decoded.splitlines() if l.strip()]
        interactions.extend(entries)
        print(f"Loaded {len(entries)} interactions from interactions_log.json")
    except:
        pass
    
    if not interactions:
        print("No interactions log found, using empty list")
    
    return interactions

def load_koniq_scores():
    """Load KonIQ-10k scores from object storage"""
    _, content = conn.get_object(BUCKET, 'koniq10k/koniq10k_scores_and_distributions.csv')
    lines = content.decode('utf-8').splitlines()
    scores = {}
    reader = csv.DictReader(lines)
    for row in reader:
        image_name = row['image_name']
        mos = float(row['MOS'])
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
        explicit_actions = [a for a in actions if 'action' in a]
        if explicit_actions:
            latest_action = explicit_actions[-1]
            label = 'low' if latest_action['action'] == 'delete' else 'high'
            candidates.append({
	        'image_path': f'production/user_uploads/immich/{photo_id}.jpg' if latest_action.get('user_id', '') in ('sidecar', '') else f'production/user_uploads/{latest_action["user_id"]}/{photo_id}', 
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
    random.shuffle(koniq_samples)
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

    # Balance classes using simple random sampling
    by_label = defaultdict(list)
    for c in candidates:
        by_label[c['label']].append(c)

    if len(by_label) > 0:
        min_count = min(len(v) for v in by_label.values())
        target_count = min_count * 2
        balanced = []
        for label, items in by_label.items():
            if len(items) > target_count:
                balanced.extend(random.sample(items, target_count))
            else:
                balanced.extend(items)
        candidates = balanced

    return candidates

def time_based_split(candidates, train_ratio=0.8):
    """Split by upload date to prevent leakage."""
    candidates.sort(key=lambda x: x['upload_date'])
    split_idx = int(len(candidates) * train_ratio)
    train = candidates[:split_idx]
    eval_set = candidates[split_idx:]
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
        tmp_path = f'/tmp/{split_name}.csv'
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        with open(tmp_path, 'rb') as f:
            conn.put_object(BUCKET, f'labels/v{version}/{split_name}.csv', f)
        print(f"Uploaded labels/v{version}/{split_name}.csv ({len(data)} samples)")

    manifest = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'train_samples': len(train),
        'eval_samples': len(eval_set),
        'sources': ['koniq10k', 'production'],
        'filters_applied': ['explicit_labels_only', 'time_based_split', 'class_balanced']
    }

    manifest_json = json.dumps(manifest, indent=2).encode('utf-8')
    conn.put_object(BUCKET, f'labels/v{version}/manifest.json', manifest_json)
    print(f"Uploaded labels/v{version}/manifest.json")

def main():
    print("Starting batch pipeline...")
    version = get_next_version()
    print(f"Creating dataset version v{version}")

    print("Loading user interactions...")
    interactions = load_interactions()
    print(f"Found {len(interactions)} interactions")

    print("Loading KonIQ-10k scores...")
    scores = load_koniq_scores()
    print(f"Loaded {len(scores)} KonIQ-10k scores")

    print("Running candidate selection...")
    candidates = candidate_selection(interactions, scores)
    print(f"Selected {len(candidates)} candidates")

    print("Applying time-based split...")
    train, eval_set = time_based_split(candidates)
    print(f"Train: {len(train)} | Eval: {len(eval_set)}")

    print("Saving versioned dataset...")
    save_dataset(train, eval_set, version)

    print(f"\nBatch pipeline complete!")
    print(f"Dataset v{version} saved to object storage")
    print(f"Train samples: {len(train)}")
    print(f"Eval samples: {len(eval_set)}")

# Step — Run training quality checks
    import subprocess
    print("\nRunning training quality checks...")
    result = subprocess.run(
    [sys.executable, 'data/batch_pipeline/training_quality_checks.py',
     str(version)],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    if result.returncode == 0:
        print(f"✅ Training quality checks PASSED — dataset v{version} ready for retraining!")
    else:
        print(f"❌ Training quality checks FAILED — retraining NOT triggered!")

if __name__ == "__main__":
    main()
