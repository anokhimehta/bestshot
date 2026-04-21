# Data Operations and Feedback Loop

## Step 2: Run data generator (on CHI@TACC instance)

```bash
ssh -i ~/.ssh/id_rsa cc@<CHI_TACC_IP>
cd bestshot
source venv/bin/activate
python data/generator/simulate_users.py
```

The batch pipeline and drift monitor run automatically every 6 hours via cron.

## Automated pipelines (every 6 hours)

| Pipeline | Script | Purpose |
|----------|--------|---------|
| Batch pipeline | `data/batch_pipeline/compile_dataset.py` | Compiles versioned train/eval datasets from feedback |
| Drift monitor | `data/features/drift_monitor.py` | Monitors production data distribution |

## Data quality gates

| Check Point | Checks | Action on Failure |
|-------------|--------|-------------------|
| Ingestion | Schema, MOS range, duplicates, readability | Logs report, blocks pipeline |
| Training dataset | Sample size, class balance, diversity, no leakage | Blocks retraining |
| Production drift | Sharpness `< 0.1`, exposure `< 0.3` | Writes `drift_alert.json` |

## Object storage (`ak12754-data-proj19`)

| Path | Contents |
|------|----------|
| `koniq10k/images/` | 10,073 KonIQ-10k training images |
| `koniq10k/synthetic/` | 4,000 augmented images |
| `production/interactions_log.json` | Data generator feedback events |
| `interactions_log.jsonl` | Real user feedback from Immich endpoint |
| `labels/v1/...v65/` | Versioned train/eval datasets |
| `labels/quality_reports/` | Quality check reports |
| `labels/drift_alert.json` | Written when drift is detected |
| `logs/transparency/` | Data transformation lineage |
| `logs/audit/` | Accountability audit trail |

## Feedback loop

User interacts with photo in Immich (`keep`/`delete`/`favorite`)  
↓  
Sidecar detects interaction and calls `/feedback`  
↓  
Serving endpoint writes to `interactions_log.jsonl`  
↓  
Batch pipeline reads `interactions_log.json` + `interactions_log.jsonl`  
↓  
Compiles versioned train/eval datasets (`labels/v{N}/`)  
↓  
Training quality checks run automatically  
↓  
If pass, `training/retrain.py` triggers retraining

## Safeguarding

| Principle | Implementation |
|-----------|---------------|
| Privacy | PII fields redacted from metadata before logging |
| Fairness | Class balancing and user diversity checks in batch pipeline |
| Transparency | All transformations logged to `logs/transparency/` |
| Accountability | Audit trail in `logs/audit/` |
| Robustness | Image validation at ingestion (size, readability, corruption) |
