# BestShot Data Pipeline

This directory contains all data pipeline components for the BestShot ML feature.

## Setup

1. Clone the repository
2. Create a `.env` file in the root directory with the following variables:
```
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_AUTH_TYPE=v3applicationcredential
OS_APPLICATION_CREDENTIAL_ID=<your_credential_id>
OS_APPLICATION_CREDENTIAL_SECRET=<your_credential_secret>
OS_REGION_NAME=CHI@TACC
BUCKET_NAME=ak12754-data-proj19
```

3. Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Components

### 1. Ingestion Pipeline
Downloads KonIQ-10k dataset and uploads to Chameleon object storage.
Also generates synthetic augmented images.
```bash
python data/ingestion/ingest_koniq.py
python data/ingestion/upload_images.py
python data/ingestion/expand_synthetic.py
```

### 2. Data Generator
Simulates real user uploads and interactions with the service.
```bash
python data/generator/simulate_users.py
```

### 3. Online Feature Computation
FastAPI service that computes sharpness, exposure, and face quality scores in real time.
```bash
python data/features/feature_service.py
```

Test with:
```bash
curl -X POST "http://localhost:8000/compute_features" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

### 4. Batch Pipeline
Compiles versioned training and evaluation datasets from production data.
```bash
python data/batch_pipeline/compile_dataset.py
```

## Object Storage Structure
```
ak12754-data-proj19/
├── koniq10k/
│   ├── images/          # 10,073 real images at 512x384
│   ├── synthetic/       # 4,000 augmented images
│   └── koniq10k_scores_and_distributions.csv
├── production/
│   ├── user_uploads/    # simulated user photo uploads
│   └── interactions_log.json
├── labels/
│   ├── v1/
│   │   ├── train.csv
│   │   ├── eval.csv
│   │   └── manifest.json
│   └── v2/
│       ├── train.csv
│       ├── eval.csv
│       └── manifest.json
└── features/
```

## Data Flow
```
User uploads photo
→ Stored in object storage
→ Feature service computes scores
→ Quality score stored in database
→ User feedback logged
→ Batch pipeline compiles training data
→ Model retrained
```
