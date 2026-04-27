"""
sidecar.py — BestShot Immich Integration Sidecar

Polls Immich for new assets, scores them via the serving API,
writes scores back to Immich, and sorts into albums.

Environment variables (set in .env):
    IMMICH_URL      — e.g. http://129.114.108.211:30283
    IMMICH_API_KEY  — generated in Immich admin panel
    SERVING_URL     — e.g. http://localhost:8000
    POLL_INTERVAL   — seconds between polls (default 30)
    
Usage:
    python sidecar.py
    
    # or run inside Docker
    docker run --rm --network host --env-file .env bestshot-serve python sidecar.py
"""

import os
import time
import base64
import json
import requests
import swiftclient
from datetime import datetime, timezone
from datetime import timedelta

# helpers
def _ensure_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return str(value).encode("utf-8")

# Config from env 

IMMICH_URL    = os.getenv("IMMICH_URL", "http://localhost:2283").rstrip("/")
IMMICH_KEY    = os.getenv("IMMICH_API_KEY")
SERVING_URL   = os.getenv("SERVING_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
PREDICT_TIMEOUT = int(os.getenv("PREDICT_TIMEOUT", "120"))
BUCKET        = os.getenv("BUCKET_NAME", "ak12754-data-proj19")

if not IMMICH_KEY:
    raise ValueError("IMMICH_API_KEY environment variable is required")

HEADERS = {
    "x-api-key":   IMMICH_KEY,
    "Accept":      "application/json",
    "Content-Type": "application/json"
}

# Swift connection for persistent storage
try:
    swift_conn = swiftclient.Connection(
        auth_version='3',
        authurl=os.getenv('OS_AUTH_URL'),
        os_options={
            'application_credential_id': os.getenv('OS_APPLICATION_CREDENTIAL_ID'),
            'application_credential_secret': os.getenv('OS_APPLICATION_CREDENTIAL_SECRET'),
            'region_name': os.getenv('OS_REGION_NAME', 'CHI@TACC'),
            'auth_type': 'v3applicationcredential'
        }
    )
    print("[sidecar] Swift connection initialized")
except Exception as e:
    print(f"[sidecar] WARNING: Swift connection failed: {e}")
    swift_conn = None

# album names — created automatically on first run
BEST_SHOTS_ALBUM_NAME = "⭐ BestShot — Best Photos"
REVIEW_ALBUM_NAME     = "🗑️ BestShot — Review for Deletion"

# Immich helpers
def save_score_event(asset_id: str, result: dict):
    """Save score event to interactions_log.jsonl in Swift."""
    if not swift_conn:
        return
    entry = {
        "event_type": "score",
        "asset_id":   asset_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "scores":     result.get("scores", {}),
        "decisions":  result.get("decisions", {})
    }
    try:
        new_line = json.dumps(entry) + "\n"
        try:
            _, content = swift_conn.get_object(BUCKET, "interactions_log.jsonl")
            existing_text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
            updated = existing_text + new_line
        except Exception:
            updated = new_line
        swift_conn.put_object(
            BUCKET,
            "interactions_log.jsonl",
            _ensure_bytes(updated),
            content_type="application/json",
        )
        print(f"[sidecar] Saved score event for {asset_id}")
    except Exception as e:
        print(f"[sidecar] Failed to save score event: {e}")


def load_scores_from_log() -> dict:
    """Load saved scores from interactions_log.jsonl on startup."""
    scores = {}
    if not swift_conn:
        return scores
    try:
        _, content = swift_conn.get_object(BUCKET, "interactions_log.jsonl")
        text = content.decode("utf-8") if isinstance(content, (bytes, bytearray)) else str(content)
        for line in text.strip().split("\n"):
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event_type") == "score":
                asset_id = entry.get("asset_id")
                if asset_id:
                    scores[asset_id] = {
                        "scores":    entry.get("scores", {}),
                        "decisions": entry.get("decisions", {})
                    }
        print(f"[sidecar] Loaded {len(scores)} saved scores from interactions log")
    except Exception as e:
        print(f"[sidecar] No existing scores found: {e}")
    return scores


def get_or_create_album(name: str) -> str:
    """Get album ID by name, or create it if it doesn't exist."""
    resp = requests.get(f"{IMMICH_URL}/api/albums", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        print(f"[sidecar] Failed to list albums: {resp.status_code}")
        return None

    for album in resp.json():
        if album.get("albumName") == name:
            print(f"[sidecar] Found album '{name}': {album['id']}")
            return album["id"]

    # create it
    resp = requests.post(
        f"{IMMICH_URL}/api/albums",
        headers=HEADERS,
        json={"albumName": name},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code in (200, 201):
        album_id = resp.json()["id"]
        print(f"[sidecar] Created album '{name}': {album_id}")
        return album_id

    print(f"[sidecar] Failed to create album '{name}': {resp.status_code} {resp.text}")
    return None

def get_asset_exif(asset_id: str) -> dict:
    """Get EXIF data for an asset."""
    resp = requests.get(
        f"{IMMICH_URL}/api/assets/{asset_id}",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 200:
        return resp.json().get("exifInfo", {})
    return {}

def get_new_assets(since: str) -> list:
    """Get assets uploaded after a given ISO timestamp."""
    resp = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        headers=HEADERS,
        json={
            "updatedAfter": since,
            "type": "IMAGE",
            "withExif": False
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        print(f"[sidecar] Failed to search assets: {resp.status_code} {resp.text}")
        return []

    items = resp.json().get("assets", {}).get("items", [])
    print(f"[sidecar] Found {len(items)} new assets")
    return items


def download_image(asset_id: str) -> bytes:
    """Download original image bytes from Immich."""
    resp = requests.get(
        f"{IMMICH_URL}/api/assets/{asset_id}/original",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 200:
        return resp.content
    print(f"[sidecar] Failed to download {asset_id}: {resp.status_code}")
    return None


def save_to_object_storage(asset_id: str, image_bytes: bytes):
    """Save image to Chameleon object storage for persistent storage and retraining."""
    if not swift_conn:
        return
    try:
        storage_path = f'production/user_uploads/immich/{asset_id}.jpg'
        swift_conn.put_object(BUCKET, storage_path, _ensure_bytes(image_bytes))
        print(f"[sidecar] Saved {asset_id} to object storage: {storage_path}")
    except Exception as e:
        print(f"[sidecar] Failed to save {asset_id} to object storage: {e}")


def write_score_to_immich(asset_id: str, result: dict):
    """Write quality score back to Immich asset description."""
    scores    = result.get("scores", {})
    decisions = result.get("decisions", {})
    composite = scores.get("composite_score", 0)
    label     = decisions.get("quality_label", "unknown")
    best_shot = decisions.get("is_best_shot", False)
    review    = decisions.get("review_flag", False)

    desc = f"BestShot: {label} | score: {composite:.1f}/10"
    if best_shot:
        desc += " ⭐"
    if review:
        desc += " 🗑️"

    resp = requests.put(
        f"{IMMICH_URL}/api/assets/{asset_id}",
        headers=HEADERS,
        json={"description": desc},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 200:
        print(f"[sidecar] Wrote score to {asset_id}: {desc}")
    else:
        print(f"[sidecar] Failed to write score to {asset_id}: {resp.status_code}")


def add_to_album(asset_id: str, album_id: str):
    """Add asset to an Immich album."""
    if not album_id:
        return
    resp = requests.put(
        f"{IMMICH_URL}/api/albums/{album_id}/assets",
        headers=HEADERS,
        json={"ids": [asset_id]},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 200:
        print(f"[sidecar] Added {asset_id} to album {album_id}")
    else:
        print(f"[sidecar] Failed to add to album: {resp.status_code}")


def get_album_assets(album_id: str) -> set:
    """Get current asset IDs in an album."""
    if not album_id:
        return set()
    resp = requests.get(
        f"{IMMICH_URL}/api/albums/{album_id}",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 200:
        assets = resp.json().get("assets", [])
        return {a["id"] for a in assets}
    return set()


def send_feedback(asset_id: str, result: dict, action: str, feature: str):
    """Send feedback to serving /feedback endpoint."""
    scores    = result.get("scores", {})
    decisions = result.get("decisions", {})
    
    # skip if we don't have prediction data — happens for pre-existing album assets
    if not scores or not decisions:
        print(f"[sidecar] Skipping feedback for {asset_id} — no prediction data")
        return
    
    payload = {
        "asset_id":  asset_id,
        "photo_id":  asset_id,
        "user_id":   "immich_user",
        "action":    action,
        "feature":   feature,
        "prediction": {
            "quality_label":   decisions.get("quality_label"),
            "composite_score": scores.get("composite_score"),
            "is_best_shot":    decisions.get("is_best_shot"),
            "review_flag":     decisions.get("review_flag"),
        }
    }

    try:
        resp = requests.post(
            f"{SERVING_URL}/feedback",
            json=payload,
            timeout=10
        )
        if resp.status_code == 200:
            print(f"[sidecar] Feedback sent: asset={asset_id} action={action} feature={feature}")
        else:
            print(f"[sidecar] Feedback failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[sidecar] Feedback error: {e}")


def is_favorited(asset_id: str) -> bool:
    """Check if an asset is favorited in Immich."""
    try:
        resp = requests.get(
            f"{IMMICH_URL}/api/assets/{asset_id}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json().get("isFavorite", False)
    except Exception as e:
        print(f"[sidecar] Error checking favorite for {asset_id}: {e}")
    return False


# Serving helpers

def score_image(asset_id: str, image_bytes: bytes) -> dict:
    """Send image bytes to serving /predict and get scores."""
    b64 = base64.b64encode(image_bytes).decode()

    resp = requests.post(
        f"{SERVING_URL}/predict",
        json={
            "request_id": asset_id,
            "user_id":    "sidecar",
            "images": [{
                "image_id":    asset_id,
                "image_bytes": b64,
                "image_path":  "",
                "metadata":    {}
            }]
        },
        timeout=PREDICT_TIMEOUT,
    )

    if resp.status_code != 200:
        print(f"[sidecar] Predict failed for {asset_id}: {resp.status_code} {resp.text}")
        return None

    results = resp.json().get("results", [])
    return results[0] if results else None

def group_bursts(assets_with_exif: list) -> dict:
    """
    Group assets into burst groups based on timestamp proximity.
    Returns dict: burst_group_id → list of asset_ids
    """
    if not assets_with_exif:
        return {}

    def get_ts(item):
        ts_str = item.get("exif", {}).get("dateTimeOriginal") or item.get("exif", {}).get("dateTime", "")
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except:
            return None

    # sort by timestamp
    sortable = [(a, get_ts(a)) for a in assets_with_exif if get_ts(a)]
    sortable.sort(key=lambda x: x[1])

    burst_groups = {}
    current_group = []

    for i, (asset, ts) in enumerate(sortable):
        if i == 0:
            current_group = [(asset, ts)]
            continue

        prev_ts = sortable[i-1][1]
        diff = abs((ts - prev_ts).total_seconds())

        if diff <= 4.0:
            current_group.append((asset, ts))
        else:
            if len(current_group) > 1:
                group_id = current_group[0][0]["id"]
                burst_groups[group_id] = [a["id"] for a, _ in current_group]
            current_group = [(asset, ts)]

    # last group
    if len(current_group) > 1:
        group_id = current_group[0][0]["id"]
        burst_groups[group_id] = [a["id"] for a, _ in current_group]

    return burst_groups

def pick_best_from_burst(burst_asset_ids: list, scored_results: dict) -> str:
    """Pick asset with highest composite score from burst group."""
    return max(
        burst_asset_ids,
        key=lambda aid: scored_results.get(aid, {}).get("scores", {}).get("composite_score", 0)
    )

# Main loop

def run():
    print(f"[sidecar] Starting BestShot sidecar")
    print(f"[sidecar] Immich:  {IMMICH_URL}")
    print(f"[sidecar] Serving: {SERVING_URL}")
    print(f"[sidecar] Poll interval: {POLL_INTERVAL}s")

    # wait for serving API to be ready
    print("[sidecar] Waiting for serving API...")
    while True:
        try:
            health = requests.get(f"{SERVING_URL}/health", timeout=5)
            if health.status_code == 200:
                print(f"[sidecar] Serving API: OK")
                break
        except Exception:
            print("[sidecar] Serving API not ready yet, retrying in 5s...")
            time.sleep(5)

    # verify Immich connection
    immich_ping = requests.get(f"{IMMICH_URL}/api/server/ping", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    if immich_ping.status_code != 200:
        print(f"[sidecar] WARNING: Immich not reachable: {immich_ping.status_code}")
    else:
        print(f"[sidecar] Immich: OK")

    # set up albums on startup
    best_shots_album_id = get_or_create_album(BEST_SHOTS_ALBUM_NAME)
    review_album_id     = get_or_create_album(REVIEW_ALBUM_NAME)

    processed_ids   = set()
    album_snapshots = {"best_shot": {}, "deletion_suggestion": {}}
    favorited_ids   = set()
    # Look back a bit on startup so we do not miss recent uploads
    # if sidecar restarts between polling windows.
    lookback_minutes = int(os.getenv("POLL_LOOKBACK_MINUTES", "180"))
    last_check = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()

    saved_scores = load_scores_from_log()

    # pre-load existing album assets using saved scores
    existing_best   = get_album_assets(best_shots_album_id)
    existing_review = get_album_assets(review_album_id)

    for asset_id in existing_best:
        if asset_id in saved_scores:
            album_snapshots["best_shot"][asset_id] = saved_scores[asset_id]
            processed_ids.add(asset_id)
            print(f"[sidecar] Restored score for {asset_id} from log")

    for asset_id in existing_review:
        if asset_id in saved_scores:
            album_snapshots["deletion_suggestion"][asset_id] = saved_scores[asset_id]
            processed_ids.add(asset_id)
            print(f"[sidecar] Restored score for {asset_id} from log")

    while True:
        print(f"\n[sidecar] Polling at {datetime.now(timezone.utc).isoformat()}")
        now = datetime.now(timezone.utc).isoformat()

        assets = get_new_assets(last_check)

        # download + score all new assets
        assets_with_exif = []
        scored_results   = {}

        for asset in assets:
            asset_id = asset.get("id")

            if asset_id in processed_ids:
                continue

            try:
                image_bytes = download_image(asset_id)
                if not image_bytes:
                    continue

                # save raw image to object storage
                save_to_object_storage(asset_id, image_bytes)

                # score it
                result = score_image(asset_id, image_bytes)
                if not result:
                    continue

                # get exif for burst detection
                exif = get_asset_exif(asset_id)
                asset["exif"] = exif
                assets_with_exif.append(asset)
                scored_results[asset_id] = result

            except Exception as e:
                print(f"[sidecar] Error processing {asset_id}: {e}")

        #  burst detection
        burst_groups = group_bursts(assets_with_exif)
        print(f"[sidecar] Detected {len(burst_groups)} burst groups")

        # build reverse lookup: asset_id → burst info
        asset_to_burst = {}
        for group_id, members in burst_groups.items():
            best_id = pick_best_from_burst(members, scored_results)
            print(f"[sidecar] Burst group {group_id}: {len(members)} photos, best={best_id}")
            for asset_id in members:
                asset_to_burst[asset_id] = {
                    "burst_group_id": group_id,
                    "is_burst_best":  asset_id == best_id
                }

        # write scores + sort into albums
        for asset in assets_with_exif:
            asset_id = asset.get("id")
            result   = scored_results.get(asset_id)
            if not result:
                continue

            decisions = result.get("decisions", {})

            # apply burst info to decisions
            burst_info = asset_to_burst.get(asset_id)
            if burst_info:
                decisions["is_burst"]       = True
                decisions["burst_group_id"] = burst_info["burst_group_id"]
                if burst_info["is_burst_best"]:
                    decisions["is_best_shot"] = True   # always mark winner as best shot
                else:
                    decisions["is_best_shot"] = False  # suppress for non-winners
                result["decisions"] = decisions

            # save score event to Swift
            save_score_event(asset_id, result)

            # write score back to Immich
            write_score_to_immich(asset_id, result)

            # sort into albums
            if decisions.get("is_best_shot"):
                add_to_album(asset_id, best_shots_album_id)
                album_snapshots["best_shot"][asset_id] = result
            elif decisions.get("review_flag"):
                add_to_album(asset_id, review_album_id)
                album_snapshots["deletion_suggestion"][asset_id] = result

            processed_ids.add(asset_id)

        # detect user feedback from album changes 
        current_best   = get_album_assets(best_shots_album_id)
        current_review = get_album_assets(review_album_id)

        # best shots — asset removed = user deleted = negative feedback
        for aid in list(album_snapshots["best_shot"].keys()):
            if aid not in current_best:
                send_feedback(aid, album_snapshots["best_shot"][aid], "delete", "best_shot")
                del album_snapshots["best_shot"][aid]

        # review album — asset removed = user kept = negative feedback
        for aid in list(album_snapshots["deletion_suggestion"].keys()):
            if aid not in current_review:
                send_feedback(aid, album_snapshots["deletion_suggestion"][aid], "keep", "deletion_suggestion")
                del album_snapshots["deletion_suggestion"][aid]

        # update snapshots with current album contents
        for aid in current_best:
            if aid not in album_snapshots["best_shot"]:
                album_snapshots["best_shot"][aid] = {}
        for aid in current_review:
            if aid not in album_snapshots["deletion_suggestion"]:
                album_snapshots["deletion_suggestion"][aid] = {}

        # check for favorites in Best Shots album
        print(f"[sidecar] Checking favorites for {len(current_best)} assets in Best Shots")
        for aid in current_best:
            if aid not in favorited_ids and is_favorited(aid):
                send_feedback(aid, album_snapshots["best_shot"].get(aid, {}), "favorite", "best_shot")
                favorited_ids.add(aid)

        # check for favorites in Review for Deletion album
        print(f"[sidecar] Checking favorites for {len(current_review)} assets in Review")
        for aid in current_review:
            if aid not in favorited_ids and is_favorited(aid):
                send_feedback(aid, album_snapshots["deletion_suggestion"].get(aid, {}), "favorite", "deletion_suggestion")
                favorited_ids.add(aid)

        last_check = now
        print(f"[sidecar] Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
