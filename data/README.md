# BestShot — Data Pipeline

**Role:** Data  
**Owner:** Anurag Kunde  
**Bucket:** ak12754-data-proj19 (CHI@TACC)

## Overview

The data pipeline is responsible for:
1. Ingesting and validating the KonIQ-10k dataset
2. Simulating production user traffic via Immich API
3. Computing online features (sharpness, exposure, face quality)
4. Compiling versioned training datasets from production feedback
5. Monitoring production data drift
6. Safeguarding (privacy, fairness, transparency, accountability, robustness)

## Directory Structure

- `ingestion/ingest_koniq.py` — Upload KonIQ-10k scores CSV to object storage
- `ingestion/upload_images.py` — Upload 10,073 KonIQ-10k images to object storage
- `ingestion/expand_synthetic.py` — Generate 4,000 synthetic augmented images
- `ingestion/quality_checks.py` — Data quality checks at ingestion (4 checks)
- `batch_pipeline/compile_dataset.py` — Build versioned train/eval datasets
- `batch_pipeline/training_quality_checks.py` — Quality gates before retraining (6 checks)
- `features/feature_service.py` — FastAPI on port 8000: sharpness, exposure, face quality
- `features/drift_monitor.py` — Monitor production data distribution drift
- `generator/simulate_users.py` — Emulate production user traffic via Immich API
- `safeguarding.py` — Privacy, robustness, transparency, accountability

## Dataset

**KonIQ-10k** — 10,073 everyday Flickr photos with crowdsourced MOS scores (1-5 scale).

- Selected over AVA (photography contest images — not representative of typical user photos)
- Selected over UHD-IQA (ultra-high-definition — not representative of smartphone photos)
- MOS normalized to 0-10: `score = (MOS - 1) / 4 * 10`
- Images stored at 512x384, EfficientNet-B3 resizes to 300x300 internally
- 4,000 synthetic images generated via blur, overexposure, underexposure, noise augmentations

## Object Storage Structure

Bucket: `ak12754-data-proj19`

- `koniq10k/images/` — 10,073 real KonIQ-10k images
- `koniq10k/synthetic/` — 4,000 augmented images
- `koniq10k/koniq10k_scores_and_distributions.csv`
- `production/user_uploads/` — Photos uploaded via Immich
- `production/interactions_log.json` — User feedback events (keep/delete/favorite)
- `labels/v1/ ... v36/` — Versioned datasets (train.csv, eval.csv, manifest.json)
- `labels/quality_reports/` — Ingestion and training quality reports
- `labels/drift_alert.json` — Written when drift is detected
- `logs/transparency/` — Data transformation lineage logs
- `logs/audit/` — Accountability audit trail

## Environment Variables
Create a `.env` file:
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_AUTH_TYPE=v3applicationcredential
OS_APPLICATION_CREDENTIAL_ID=<your_credential_id>
OS_APPLICATION_CREDENTIAL_SECRET=<your_credential_secret>
OS_REGION_NAME=CHI@TACC
BUCKET_NAME=ak12754-data-proj19
IMMICH_URL=http://129.114.108.98:30283
IMMICH_API_KEY=<your_api_key>
MLFLOW_URL=http://129.114.25.247:8000

## How to Run

### Ingestion Pipeline

```bash
python data/ingestion/ingest_koniq.py
python data/ingestion/upload_images.py
python data/ingestion/expand_synthetic.py
python data/ingestion/quality_checks.py
```

### Data Generator

```bash
python data/generator/simulate_users.py
```

Simulates 10 users uploading photos to Immich and giving feedback. Runs continuously, updates interactions_log.json every 30 seconds.

### Batch Pipeline

```bash
python data/batch_pipeline/compile_dataset.py
```

Runs automatically every 6 hours via cron. Compiles versioned train/eval datasets and runs training quality checks automatically.

### Feature Service

```bash
uvicorn data.features.feature_service:app --host 0.0.0.0 --port 8000
```

### Drift Monitoring

```bash
python data/features/drift_monitor.py
```

Runs automatically every 6 hours via cron.

### Safeguarding

```bash
python data/safeguarding.py
```

## Automation

Cron jobs run automatically every 6 hours on CHI@TACC instance:

```bash
0 */6 * * * python data/batch_pipeline/compile_dataset.py
0 */6 * * * python data/features/drift_monitor.py
```

## Data Quality Checks

### Point 1 — At Ingestion (4 checks)

| Check | Description | Status |
|-------|-------------|--------|
| CSV Schema | Verifies required columns exist | PASS |
| MOS Score Range | Validates scores are 1-5 | PASS |
| Duplicates | No duplicate image names | PASS |
| Image Readability | Sample 100 images are readable | PASS |

### Point 2 — Training Dataset (6 checks)

| Check | Description | Threshold |
|-------|-------------|-----------|
| Train Minimum Samples | Enough training data | 500+ |
| Eval Minimum Samples | Enough eval data | 100+ |
| Class Balance | No class dominates | ratio < 4.0 |
| User Diversity | No single user dominates | < 50% |
| Score Distribution | Scores in valid range | 0-10 |
| No Leakage | No overlap between train and eval | 0 overlap |

### Point 3 — Production Drift Monitoring

- Monitors sharpness and exposure distributions every 6 hours
- Alerts if mean sharpness < 0.1 (photos getting blurrier)
- Alerts if mean exposure < 0.3 (photos getting darker)
- Writes `labels/drift_alert.json` when drift detected

## Batch Pipeline Logic

1. Load user interactions from `production/interactions_log.json` (supports both JSON array and JSONL)
2. Load KonIQ-10k scores from object storage
3. Candidate selection — random shuffle, class balancing (max ratio 4.0)
4. Time-based split (80% train, 20% eval) — older photos go to train, newer to eval
5. Run training quality checks automatically
6. Save versioned dataset to `labels/v{N}/`

## Label Mapping

User feedback mapped to photo quality labels:

- `delete` → low quality
- `keep` → high quality
- `favorite` → high quality

Note: The `feature` field (best_shot/review_for_deletion) is used by the training pipeline to assess model performance, not by the data pipeline for label assignment.

## Safeguarding

| Principle | Implementation |
|-----------|---------------|
| Privacy | PII fields (email, phone, name, address) redacted from metadata |
| Fairness | Class balancing and user diversity checks in batch pipeline |
| Transparency | All data transformations logged to `logs/transparency/` |
| Accountability | Audit trail of all pipeline runs in `logs/audit/` |
| Robustness | Image validation at ingestion (size, readability, corruption) |

## Integration with Other Roles

| Role | Interface |
|------|-----------|
| Training (Anokhi) | Reads `labels/v{N}/train.csv` and `eval.csv` from object storage. Calls `compile_dataset.py` via subprocess before retraining. |
| Serving (Lava) | Calls `feature_service.py` on port 8000 for sharpness/exposure scores. Writes feedback to `production/interactions_log.json`. |
| DevOps (Dhanush) | Deploys data containers as K8s CronJobs. Manages Swift credentials as K8s secrets. |

