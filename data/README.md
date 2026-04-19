# BestShot — Data Pipeline

**Role:** Data  
**Owner:** Anurag Kunde  
**Instance:** CHI@TACC (`ak12754-data-proj19`)

## Overview

The data pipeline is responsible for:
1. Ingesting and validating the KonIQ-10k dataset
2. Simulating production user traffic via Immich API
3. Computing online features (sharpness, exposure, face quality)
4. Compiling versioned training datasets from production feedback
5. Monitoring production data drift
6. Safeguarding (privacy, fairness, transparency, accountability, robustness)

## Directory Structure
data/
├── ingestion/
│   ├── ingest_koniq.py         # Upload KonIQ-10k scores CSV to object storage
│   ├── upload_images.py        # Upload 10,073 KonIQ-10k images to object storage
│   ├── expand_synthetic.py     # Generate 4,000 synthetic augmented images
│   ├── quality_checks.py       # Data quality checks at ingestion (4 checks)
│   ├── Dockerfile
│   └── requirements.txt
├── batch_pipeline/
│   ├── compile_dataset.py      # Build versioned train/eval datasets
│   ├── training_quality_checks.py  # Quality gates before retraining (6 checks)
│   ├── Dockerfile
│   └── requirements.txt
├── features/
│   ├── feature_service.py      # FastAPI on port 8000: sharpness, exposure, face quality
│   ├── drift_monitor.py        # Monitor production data distribution drift
│   ├── Dockerfile
│   └── requirements.txt
├── generator/
│   ├── simulate_users.py       # Emulate production user traffic via Immich API
│   ├── Dockerfile
│   └── requirements.txt
└── safeguarding.py             # Privacy, robustness, transparency, accountability

## Dataset

**KonIQ-10k** — 10,073 everyday Flickr photos with crowdsourced MOS scores (1-5 scale).

- Selected over AVA (photography contest images — not representative of typical user photos)
- Selected over UHD-IQA (ultra-high-definition — not representative of smartphone photos)
- MOS normalized to 0-10: `score = (MOS - 1) / 4 * 10`
- Images stored at 512x384, EfficientNet-B3 resizes to 300x300 internally
- 4,000 synthetic images generated via blur, overexposure, underexposure, noise augmentations

## Object Storage Structure
ak12754-data-proj19/
├── koniq10k/
│   ├── images/                      # 10,073 real KonIQ-10k images
│   ├── synthetic/                   # 4,000 augmented images
│   └── koniq10k_scores_and_distributions.csv
├── production/
│   ├── user_uploads/                # Photos uploaded via Immich
│   └── interactions_log.json        # User feedback events (keep/delete/favorite)
├── labels/
│   ├── v1/ ... v36/                 # Versioned datasets
│   │   ├── train.csv
│   │   ├── eval.csv
│   │   └── manifest.json
│   ├── quality_reports/             # Ingestion + training quality reports
│   └── drift_alert.json             # Written when drift is detected
└── logs/
├── transparency/                # Data transformation lineage logs
└── audit/                       # Accountability audit trail

## Environment Variables

Create a `.env` file:
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_AUTH_TYPE=v3applicationcredential
OS_APPLICATION_CREDENTIAL_ID=<your_credential_id>
OS_APPLICATION_CREDENTIAL_SECRET=<your_credential_secret>
OS_REGION_NAME=CHI@TACC
BUCKET_NAME=ak12754-data-proj19
IMMICH_URL=http://129.114.26.156:30283
IMMICH_API_KEY=<your_api_key>
MLFLOW_URL=http://129.114.25.247:8000

## How to Run

### Ingestion Pipeline
```bash
# Upload scores CSV and images to object storage
python data/ingestion/ingest_koniq.py
python data/ingestion/upload_images.py

# Generate synthetic images
python data/ingestion/expand_synthetic.py

# Run quality checks standalone
python data/ingestion/quality_checks.py
```

### Data Generator (Production Simulation)
```bash
# Simulates 10 users uploading photos to Immich and giving feedback
# Runs continuously, updates interactions_log.json every 30 seconds
python data/generator/simulate_users.py
```

### Batch Pipeline
```bash
# Compiles versioned train/eval datasets from interactions_log.json
# Runs training quality checks automatically after compilation
# Runs automatically every 6 hours via cron
python data/batch_pipeline/compile_dataset.py

# Run training quality checks standalone
python data/batch_pipeline/training_quality_checks.py <version>
```

### Feature Service
```bash
# Start FastAPI feature service on port 8000
uvicorn data.features.feature_service:app --host 0.0.0.0 --port 8000
```

### Drift Monitoring
```bash
# Monitor production data drift vs training distribution
# Runs automatically every 6 hours via cron
python data/features/drift_monitor.py
```

### Safeguarding
```bash
python data/safeguarding.py
```

## Automation

Cron jobs run automatically every 6 hours on CHI@TACC instance:
0 */6 * * * python data/batch_pipeline/compile_dataset.py
0 */6 * * * python data/features/drift_monitor.py

## Data Quality Checks

### Point 1 — At Ingestion (4 checks)
| Check | Description | Status |
|-------|-------------|--------|
| CSV Schema | Verifies required columns exist | ✅ PASS |
| MOS Score Range | Validates scores are 1-5 | ✅ PASS |
| Duplicates | No duplicate image names | ✅ PASS |
| Image Readability | Sample 100 images are readable | ✅ PASS |

### Point 2 — Training Dataset (6 checks)
| Check | Description | Threshold |
|-------|-------------|-----------|
| Train Minimum Samples | Enough training data | 500+ |
| Eval Minimum Samples | Enough eval data | 100+ |
| Class Balance | No class dominates | ratio < 4.0 |
| User Diversity | No single user dominates | < 50% |
| Score Distribution | Scores in valid range | 0-10 |
| No Leakage | No overlap between train/eval | 0 overlap |

### Point 3 — Production Drift Monitoring
- Monitors sharpness and exposure distributions every 6 hours
- Alerts if mean sharpness < 0.1 (photos getting blurrier)
- Alerts if mean exposure < 0.3 (photos getting darker)
- Writes `labels/drift_alert.json` when drift detected

## Batch Pipeline Logic

1. Load user interactions from `production/interactions_log.json`
2. Load KonIQ-10k scores from object storage
3. Candidate selection:
   - Prioritize explicit user labels
   - Random shuffle KonIQ-10k samples (avoid ordering bias)
   - Balance classes (max imbalance ratio: 4.0)
4. Time-based split (80% train, 20% eval) — older photos → train, newer → eval
5. Run training quality checks
6. Save versioned dataset to `labels/v{N}/`

## Label Mapping

User feedback is mapped to photo quality labels:
delete   → low quality
keep     → high quality
favorite → high quality

Note: The `feature` field (best_shot/review_for_deletion) is used by the training
pipeline to assess model performance, not by the data pipeline for label assignment.

## Safeguarding

| Principle | Implementation |
|-----------|---------------|
| Privacy | PII fields (email, phone, name, address) redacted from metadata |
| Fairness | Class balancing + user diversity checks in batch pipeline |
| Transparency | All data transformations logged to `logs/transparency/` |
| Accountability | Audit trail of all pipeline runs in `logs/audit/` |
| Robustness | Image validation at ingestion (size, readability, corruption) |

## Integration with Other Roles

| Role | Interface |
|------|-----------|
| Training (Anokhi) | Reads `labels/v{N}/train.csv` and `eval.csv` from object storage. Calls `compile_dataset.py` via subprocess before retraining. |
| Serving (Lava) | Calls `feature_service.py` on port 8000 for sharpness/exposure scores. Writes feedback to `production/interactions_log.json`. |
| DevOps (Dhanush) | Deploys data containers as K8s CronJobs. Manages Swift credentials as K8s secrets. |
