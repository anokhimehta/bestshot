# BestShot Training

Training pipeline for BestShot's image quality assessment model.

## Overview

The training pipeline fine-tunes an EfficientNet-B3 model on the KonIQ-10k dataset, augmented with user feedback collected in production. It is triggered automatically by `retrain.py`, which runs as a Kubernetes CronJob, but can also be run manually for testing.

## Directory Structure

```
training/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── config/              # Training configs (baseline, finetune, smoketest, etc.)
├── retrain.py           # Retraining trigger script
├── train.py             # Model training script
└── evaluate.py          # Offline evaluation script

data/
└── batch_pipeline/
    ├── compile_dataset.py         # Builds versioned dataset from Swift
    └── training_quality_checks.py # Validates dataset before training
```

## Environment Setup

Create a `training/.env` file with the following variables:

```
OS_AUTH_URL=https://chi.tacc.chameleoncloud.org:5000/v3
OS_AUTH_TYPE=v3applicationcredential
OS_APPLICATION_CREDENTIAL_ID=<your-credential-id>
OS_APPLICATION_CREDENTIAL_SECRET=<your-credential-secret>
OS_REGION_NAME=CHI@TACC
BUCKET_NAME=<your-swift-bucket>
```

## Building the Docker Image

Build from the repo root (not from inside `training/`):

```bash
cd ~/bestshot
docker build -t bestshot-train -f training/docker/Dockerfile .
```

## Running retrain.py

`retrain.py` checks whether retraining should be triggered based on:
- Number of new feedback events since last retrain (threshold: 500)
- Days elapsed since last retrain (threshold: 7 days)
- Negative feedback rate (threshold: 40% over at least 50 events)

If any threshold is met, it runs `compile_dataset.py`, validates the dataset, and triggers a Kubernetes training Job.

**Normal run (threshold checks apply):**
```bash
docker run --rm -it --env-file training/.env bestshot-train python3 training/retrain.py
```

**Force run (skip threshold checks):**
```bash
docker run --rm -it --env-file training/.env bestshot-train python3 training/retrain.py --force
```

> Note: The final step of `retrain.py` triggers a Kubernetes Job via `kubectl`. This will fail if run outside the cluster — this is expected. The compile and quality check steps will still run and can be validated this way.

## Running train.py Directly

To run a training job manually with a specific config:

```bash
docker run --rm -it --env-file training/.env bestshot-train python3 training/train.py --config training/config/baseline.yaml
```

Available configs: `baseline`, `smoketest`, `head_only`, `partial_finetune`, `full_finetune`.
