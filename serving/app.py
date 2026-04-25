from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter, Gauge
from datetime import datetime
import json
import os
import swiftclient

try:
    from serving.model import load_model, predict_batch
except ModuleNotFoundError:
    from model import load_model, predict_batch

app = FastAPI(title="BestShot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prometheus instrumentation - automatically tracks latency, request count, error rate at /metrics
Instrumentator().instrument(app).expose(app)

# Custom prediction metrics 

# Distribution of composite quality scores (0-10)
composite_score_histogram = Histogram(
    "bestshot_composite_score",
    "Distribution of composite quality scores",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)

# Distribution of raw koniq scores from the model (0-10)
koniq_score_histogram = Histogram(
    "bestshot_koniq_score",
    "Distribution of KonIQ model scores",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)

# Count of images flagged as best shot
best_shot_counter = Counter(
    "bestshot_best_shot_total",
    "Total images flagged as best shot"
)

# Count of images flagged for review/deletion
review_flag_counter = Counter(
    "bestshot_review_flag_total",
    "Total images flagged for review"
)

# Count of high vs low quality labels
quality_label_counter = Counter(
    "bestshot_quality_label_total",
    "Count of quality label decisions",
    ["label"]  # label = "high_quality" or "low_quality"
)

# Feedback metrics
# Count of feedback actions
feedback_action_counter = Counter(
    "bestshot_feedback_action_total",
    "Count of user feedback actions",
    ["action"]  # action = keep, delete, favorite
)

# Track disagreements: model said X but user did opposite
disagreement_counter = Counter(
    "bestshot_disagreement_total",
    "Count of cases where user action disagreed with model prediction"
)

# Swift config
SWIFT_AUTH_URL   = os.getenv("OS_AUTH_URL")
SWIFT_AUTH_TYPE  = os.getenv("OS_AUTH_TYPE", "v3applicationcredential")
SWIFT_CRED_ID    = os.getenv("OS_APPLICATION_CREDENTIAL_ID")
SWIFT_CRED_SEC   = os.getenv("OS_APPLICATION_CREDENTIAL_SECRET")
SWIFT_REGION     = os.getenv("OS_REGION_NAME")
BUCKET_NAME      = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is required but not set")
INTERACTIONS_KEY = "interactions_log.jsonl"

# Model startup 
model = load_model()

# Swift helpers 
def get_swift_conn():
    return swiftclient.Connection(
        authurl=SWIFT_AUTH_URL,
        auth_version="3",
        os_options={
            "auth_type":                     SWIFT_AUTH_TYPE,
            "application_credential_id":     SWIFT_CRED_ID,
            "application_credential_secret": SWIFT_CRED_SEC,
            "region_name":                   SWIFT_REGION,
        }
    )


def append_to_interactions_log(conn, entry: dict):
    """Append a single line to interactions_log.jsonl — no full read needed."""
    new_line = json.dumps(entry) + "\n"
    try:
        _, content = conn.get_object(BUCKET_NAME, INTERACTIONS_KEY)
        updated = content.decode() + new_line
    except swiftclient.exceptions.ClientException as e:
        if "404" in str(e) or "Not Found" in str(e):
            print("[feedback] interactions_log.jsonl not found, creating new file")
            updated = new_line
        else:
            print(f"[feedback] Swift read error: {e}")
            raise
    conn.put_object(
        BUCKET_NAME,
        INTERACTIONS_KEY,
        updated,
        content_type="application/json"
    )

# Endpoints
@app.get("/health")
def health():
    """Health check — also used by run.sh startup wait loop."""
    return {"status": "ok", "model": "bestshot-iqa"}


@app.post("/predict")
def predict(request: dict):
    images = request.get("images", [])
    batch_outputs = predict_batch(model, images)

    results = []
    for img, output in zip(images, batch_outputs):
        scores    = output["scores"]
        decisions = output["decisions"]

        # record metrics for every image scored
        composite_score_histogram.observe(scores.get("composite_score", 0))
        koniq_score_histogram.observe(scores.get("koniq_score", 0))
        quality_label_counter.labels(label=decisions.get("quality_label", "unknown")).inc()

        if decisions.get("is_best_shot"):
            best_shot_counter.inc()
        if decisions.get("review_flag"):
            review_flag_counter.inc()

        results.append({
            "image_id":  img.get("image_id"),
            "scores":    scores,
            "decisions": decisions,
        })

    return {
        "request_id": request.get("request_id", ""),
        "results":    results,
    }


@app.post("/feedback")
def feedback(request: dict):
    prediction = request.get("prediction", {})
    action     = request.get("action")
    feature    = request.get("feature")  # "best_shot" or "deletion_suggestion"

    valid_actions  = {"keep", "delete", "favorite"}
    valid_features = {"best_shot", "deletion_suggestion"}

    if action not in valid_actions:
        return {
            "status":  "error",
            "message": f"Invalid action. Must be one of: {valid_actions}"
        }

    if feature not in valid_features:
        return {
            "status":  "error",
            "message": f"Invalid feature. Must be one of: {valid_features}"
        }

    entry = {
        "event_type": "user_feedback",
        "event_id":   f"evt_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{request.get('photo_id', '')}",
        "photo_id":   request.get("photo_id"),
        "asset_id":   request.get("asset_id"),
        "user_id":    request.get("user_id"),
        "action":     action,
        "feature":    feature,
        "timestamp":  datetime.utcnow().isoformat(),
        "model_prediction": {
            "quality_label":   prediction.get("quality_label"),
            "composite_score": prediction.get("composite_score"),
            "is_best_shot":    prediction.get("is_best_shot"),
            "review_flag":     prediction.get("review_flag"),
        }
    }

    # record feedback metrics
    feedback_action_counter.labels(action=action).inc()

    # disagreement logic depends on feature
    if feature == "deletion_suggestion":
        # model suggested deletion
        # keep → negative (user disagrees, don't delete)
        # delete → positive (user agrees)
        # favorite → negative (user strongly disagrees)
        disagreed = action in {"keep", "favorite"}

    elif feature == "best_shot":
        # model suggested this is the best shot
        # keep → positive (user agrees)
        # delete → negative (user strongly disagrees, trigger retrain)
        # favorite → positive (user agrees)
        disagreed = action == "delete"

    if disagreed:
        disagreement_counter.inc()

    # write to Swift bucket (with fallback to local file if Swift is unavailable)
    try:
        conn = get_swift_conn()
        append_to_interactions_log(conn, entry)
        print(f"[feedback] Logged: {entry['event_id']} feature={feature} action={action} disagreement={disagreed}")
        return {"status": "ok", "logged": entry}

    except Exception as e:
        import traceback
        print(f"[feedback] Swift write failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        with open("feedback_log_fallback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {
            "status":  "fallback",
            "message": f"Swift unavailable: {str(e)}",
            "logged":  entry,
        }