# BestShot — Serving

Inference API and Immich integration sidecar for the BestShot image quality assessment system.

## Overview

Two processes run together to form the serving layer:

| Process | File | Description |
|---------|------|-------------|
| **API** | `app.py` | FastAPI server — scores images and collects user feedback |
| **Sidecar** | `sidecar.py` | Polls Immich, sends images to the API, writes scores back, sorts into albums |

```
Immich (new photo uploaded)
    └─→ sidecar.py (polls every 30s)
            └─→ POST /predict
                    └─→ model.py (MLflow model + feature extraction)
                            └─→ scores + decisions written back to Immich
                                    └─→ album sorting (Best Photos / Review for Deletion)
                                            └─→ POST /feedback (on user action)
                                                    └─→ Swift object storage (interactions_log.jsonl)
```

## Endpoints

### `GET /health`
Returns `{"status": "ok"}`. Used by the startup script to wait for readiness.

### `POST /predict`
Scores a batch of images.

**Request:**
```json
{
  "request_id": "abc123",
  "images": [
    {
      "image_id": "photo-1",
      "image_bytes": "<base64-encoded image>",
      "image_path": "",
      "metadata": {}
    }
  ]
}
```

**Response:**
```json
{
  "request_id": "abc123",
  "results": [
    {
      "image_id": "photo-1",
      "scores": {
        "koniq_score": 7.21,
        "sharpness": 6.84,
        "exposure": 8.10,
        "face_quality": 5.93,
        "composite_score": 7.12
      },
      "decisions": {
        "quality_label": "high_quality",
        "is_best_shot": false,
        "review_flag": false
      }
    }
  ]
}
```

**Scoring logic (`model.py`):**
- `koniq_score` — raw output of the EfficientNet-B3 IQA model loaded from MLflow
- `sharpness` — Laplacian variance on the full image (0–10)
- `exposure` — brightness + contrast score (0–10)
- `face_quality` — Laplacian variance on the center crop (0–10)
- `composite_score` = `0.4 × koniq + 0.3 × sharpness + 0.2 × exposure + 0.1 × face_quality`

**Decision thresholds:**
- `is_best_shot`: composite > 8.5
- `review_flag` (suggest deletion): composite < 5.0
- `quality_label`: "high_quality" if composite > 7.0, else "low_quality"

### `POST /feedback`
Records a user action (keep / delete / favorite) against a model prediction. Logs to Swift object storage at `interactions_log.jsonl`, which the training pipeline reads during retraining.

**Request:**
```json
{
  "photo_id": "photo-1",
  "asset_id": "immich-asset-uuid",
  "user_id": "user-123",
  "action": "keep",
  "feature": "best_shot",
  "prediction": {
    "quality_label": "high_quality",
    "composite_score": 8.7,
    "is_best_shot": true,
    "review_flag": false
  }
}
```

Valid `action` values: `keep`, `delete`, `favorite`  
Valid `feature` values: `best_shot`, `deletion_suggestion`

A disagreement is recorded when the user's action contradicts the model's decision (e.g. model flagged best shot, user deleted it). Disagreements feed into Prometheus metrics and retraining triggers.

## Sidecar (`sidecar.py`)

Runs as a separate process alongside the API. On each poll cycle it:

1. Fetches new Immich assets uploaded since last check (`/api/search/metadata`)
2. Downloads each image (`/api/assets/{id}/original`)
3. Calls `POST /predict` on the local API
4. Writes the composite score and label back to the Immich asset description
5. Sorts the asset into one of two auto-managed albums:
   - **⭐ BestShot — Best Photos** — `is_best_shot = true`
   - **🗑️ BestShot — Review for Deletion** — `review_flag = true`
6. Monitors album membership each cycle — if an asset disappears from an album, sends feedback to `POST /feedback`
7. Checks for favorited assets and sends `favorite` feedback accordingly

## Model Loading

The model is loaded from MLflow at startup (`model.py`). It resolves the model by alias (configurable via env var), with a fallback to the latest registered version.

```
MLFLOW_TRACKING_URI  →  MLflow server
MLFLOW_MODEL_ALIAS   →  alias to load (default: value of MODEL_STAGE, then "production")
MODEL_STAGE          →  fallback alias ("production" / "staging" / "canary")
```

GPU inference uses ROCm (`torch.device("cuda")` — ROCm exposes itself as CUDA). Falls back to CPU if unavailable or if `DEVICE=cpu` is set.

## Inference Configs

Select a config via `CONFIG_NAME` env var. Configs live in `configs/`:

| Config | Device | Mode |
|--------|--------|------|
| `gpu_sequential` | GPU | One image at a time (default) |
| `gpu_batch` | GPU | Batched inference |
| `cpu_sequential` | CPU | One image at a time |
| `cpu_batch` | CPU | Batched inference |

## Running

### Prerequisites

Copy `.env.template` (repo root) to `serving/.env` and fill in your values.

Required env vars:
```
IMMICH_URL
IMMICH_API_KEY
MLFLOW_TRACKING_URI
OS_AUTH_URL
OS_APPLICATION_CREDENTIAL_ID
OS_APPLICATION_CREDENTIAL_SECRET
OS_REGION_NAME
BUCKET_NAME
```

Optional:
```
SERVING_URL         # default: http://localhost:8000
POLL_INTERVAL       # sidecar poll interval in seconds (default: 30)
CONFIG_NAME         # inference config (default: gpu_sequential)
MODEL_STAGE         # MLflow alias to load (default: production)
DEVICE              # force cpu or cuda
```

### Start all services

```bash
cd serving
./start.sh
```

This pulls the latest code, starts the API container, waits for `/health`, starts the sidecar, and launches Prometheus + Grafana.

### Start manually (without Docker)

```bash
# API
uvicorn app:app --host 0.0.0.0 --port 8000

# Sidecar (separate terminal)
python sidecar.py
```

### Docker

```bash
# Build
docker build -f docker/Dockerfile.serve -t bestshot-serve .

# API (GPU)
docker run -d --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --name bestshot-api \
  --env-file .env \
  -e CONFIG_NAME=gpu_sequential \
  bestshot-serve \
  uvicorn app:app --host 0.0.0.0 --port 8000

# Sidecar
docker run -d --network host \
  --name bestshot-sidecar \
  --env-file .env \
  bestshot-serve \
  python -u sidecar.py
```

The pre-built image is also available at `ghcr.io/anokhimehta/bestshot-serving:latest`.

## Monitoring

Prometheus metrics are exposed at `GET /metrics` (auto-instrumented by `prometheus-fastapi-instrumentator`).

Custom metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `bestshot_composite_score` | Histogram | Distribution of composite quality scores |
| `bestshot_koniq_score` | Histogram | Distribution of raw model scores |
| `bestshot_best_shot_total` | Counter | Images flagged as best shot |
| `bestshot_review_flag_total` | Counter | Images flagged for deletion review |
| `bestshot_quality_label_total` | Counter | high_quality vs low_quality decisions |
| `bestshot_feedback_action_total` | Counter | User actions (keep / delete / favorite) |
| `bestshot_disagreement_total` | Counter | User actions that contradict model predictions |

Start the monitoring stack:
```bash
docker compose -f docker/docker-compose-monitoring.yaml up -d
```

- Prometheus: `http://<server-ip>:9090`
- Grafana: `http://<server-ip>:3000`

## Benchmarks

Tested with an untrained model, 20 requests × 7 images:

| Metric | Value |
|--------|-------|
| Avg latency per request | 790 ms |
| Throughput | 1.26 req/sec |

See `benchmark_results.txt` and `benchmark_pipeline.ipynb` for the full benchmark setup.

## File Reference

```
serving/
├── app.py                  # FastAPI server (/health, /predict, /feedback)
├── sidecar.py              # Immich integration (polling, scoring, album sorting, feedback)
├── model.py                # Model loading from MLflow + inference + feature extraction
├── config.py               # Config loader (reads CONFIG_NAME env var)
├── configs/                # Inference mode configs (gpu_sequential, cpu_batch, etc.)
├── benchmark.py            # Latency/throughput benchmark script
├── start.sh                # Startup script (API + sidecar + monitoring)
├── docker/
│   ├── Dockerfile.serve            # Docker image (ROCm + FastAPI)
│   └── docker-compose-monitoring.yaml  # Prometheus + Grafana stack
├── monitoring/
│   └── prometheus.yml      # Prometheus scrape config
└── setup_node.ipynb        # Node setup walkthrough notebook
```
