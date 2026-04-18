from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

# load model at startup
model = load_model()

# swift config from env
SWIFT_AUTH_URL   = os.getenv("OS_AUTH_URL")
SWIFT_AUTH_TYPE  = os.getenv("OS_AUTH_TYPE", "v3applicationcredential")
SWIFT_CRED_ID    = os.getenv("OS_APPLICATION_CREDENTIAL_ID")
SWIFT_CRED_SEC   = os.getenv("OS_APPLICATION_CREDENTIAL_SECRET")
SWIFT_REGION     = os.getenv("OS_REGION_NAME")
BUCKET_NAME      = os.getenv("BUCKET_NAME", "bestshot-feedback")
INTERACTIONS_KEY = "interactions_log.jsonl"

def get_swift_conn():
    """Create a Swift connection using application credentials."""
    return swiftclient.Connection(
        authurl=SWIFT_AUTH_URL,
        auth_version="3",
        os_options={
            "auth_type":                    SWIFT_AUTH_TYPE,
            "application_credential_id":    SWIFT_CRED_ID,
            "application_credential_secret": SWIFT_CRED_SEC,
            "region_name":                  SWIFT_REGION,
        }
    )

def append_to_interactions_log(conn, entry: dict):
    """Append a single line to interactions_log.jsonl — no full read needed."""
    new_line = json.dumps(entry) + "\n"
    
    try:
        # read existing content
        _, content = conn.get_object(BUCKET_NAME, INTERACTIONS_KEY)
        updated = content.decode() + new_line
    except swiftclient.exceptions.ClientException as e:
        if "404" in str(e) or "Not Found" in str(e):
            # file doesn't exist yet — start fresh
            print("[feedback] interactions_log.jsonl not found, creating new file")
            updated = new_line
        else:
            print(f"[feedback] Swift read error: {e}")
            raise
    
    conn.put_object(
        BUCKET_NAME,
        "interactions_log.jsonl",
        updated,
        content_type="application/json"
    )

@app.post("/predict")
def predict(request: dict):
    images = request.get("images", [])
    batch_outputs = predict_batch(model, images)

    results = []
    for img, output in zip(images, batch_outputs):
        results.append({
            "image_id": img.get("image_id"),
            "scores":   output["scores"],
            "decisions": output["decisions"]
        })

    return {
        "request_id": request.get("request_id", ""),
        "results":    results
    }


@app.post("/feedback")
def feedback(request: dict):
    prediction = request.get("prediction", {})

    entry = {
        "event_id":   f"evt_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{request.get('photo_id', '')}",
        "photo_id":   request.get("photo_id"),
        "asset_id":   request.get("asset_id"),
        "user_id":    request.get("user_id"),
        "action":     request.get("action"),
        "timestamp":  datetime.utcnow().isoformat(),
        "confidence": request.get("confidence", "explicit"),
        "model_prediction": {
            "quality_label":   prediction.get("quality_label"),
            "composite_score": prediction.get("composite_score"),
            "is_best_shot":    prediction.get("is_best_shot"),
            "review_flag":     prediction.get("review_flag"),
        }
    }

    valid_actions = {"keep", "delete", "favorite"}
    if entry["action"] not in valid_actions:
        return {
            "status": "error",
            "message": f"Invalid action. Must be one of: {valid_actions}"
        }

    try:
        conn = get_swift_conn()
        append_to_interactions_log(conn, entry)
        print(f"[feedback] Logged: {entry['event_id']} action={entry['action']}")
        return {"status": "ok", "logged": entry}

    except Exception as e:
        import traceback
        print(f"[feedback] Swift write failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        # fallback — write locally so feedback isn't lost
        with open("feedback_log_fallback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {
            "status": "fallback",
            "message": f"Swift unavailable: {str(e)}",
            "logged": entry
        }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "bestshot-iqa"}