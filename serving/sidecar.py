"""
sidecar.py — BestShot Immich Integration Sidecar

Polls Immich for new assets, scores them via the serving API,
writes scores back to Immich, and sorts into albums.

Environment variables (set in .env):
    IMMICH_URL      — e.g. http://129.114.108.98:30283
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
from datetime import datetime, timezone

# Config from env 

IMMICH_URL    = os.getenv("IMMICH_URL", "http://localhost:2283").rstrip("/")
IMMICH_KEY    = os.getenv("IMMICH_API_KEY")
SERVING_URL   = os.getenv("SERVING_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))

if not IMMICH_KEY:
    raise ValueError("IMMICH_API_KEY environment variable is required")

HEADERS = {
    "x-api-key":   IMMICH_KEY,
    "Accept":      "application/json",
    "Content-Type": "application/json"
}

# album names — created automatically on first run
BEST_SHOTS_ALBUM_NAME = "⭐ BestShot — Best Photos"
REVIEW_ALBUM_NAME     = "🗑️ BestShot — Review for Deletion"

# Immich helpers

def get_or_create_album(name: str) -> str:
    """Get album ID by name, or create it if it doesn't exist."""
    resp = requests.get(f"{IMMICH_URL}/api/albums", headers=HEADERS)
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
        json={"albumName": name}
    )
    if resp.status_code in (200, 201):
        album_id = resp.json()["id"]
        print(f"[sidecar] Created album '{name}': {album_id}")
        return album_id

    print(f"[sidecar] Failed to create album '{name}': {resp.status_code} {resp.text}")
    return None


def get_new_assets(since: str) -> list:
    """Get assets uploaded after a given ISO timestamp."""
    resp = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        headers=HEADERS,
        json={
            "updatedAfter": since,
            "type": "IMAGE",
            "withExif": False
        }
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
        headers=HEADERS
    )
    if resp.status_code == 200:
        return resp.content
    print(f"[sidecar] Failed to download {asset_id}: {resp.status_code}")
    return None


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
        json={"description": desc}
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
        json={"ids": [asset_id]}
    )
    if resp.status_code == 200:
        print(f"[sidecar] Added {asset_id} to album {album_id}")
    else:
        print(f"[sidecar] Failed to add to album: {resp.status_code}")

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
        }
    )

    if resp.status_code != 200:
        print(f"[sidecar] Predict failed for {asset_id}: {resp.status_code} {resp.text}")
        return None

    results = resp.json().get("results", [])
    return results[0] if results else None

# Main loop

def run():
    print(f"[sidecar] Starting BestShot sidecar")
    print(f"[sidecar] Immich:  {IMMICH_URL}")
    print(f"[sidecar] Serving: {SERVING_URL}")
    print(f"[sidecar] Poll interval: {POLL_INTERVAL}s")

    # verify connections on startup
    health = requests.get(f"{SERVING_URL}/health")
    if health.status_code != 200:
        print(f"[sidecar] WARNING: Serving API not healthy: {health.status_code}")
    else:
        print(f"[sidecar] Serving API: OK")

    immich_ping = requests.get(f"{IMMICH_URL}/api/server/ping", headers=HEADERS)
    if immich_ping.status_code != 200:
        print(f"[sidecar] WARNING: Immich not reachable: {immich_ping.status_code}")
    else:
        print(f"[sidecar] Immich: OK")

    # set up albums on startup
    best_shots_album_id = get_or_create_album(BEST_SHOTS_ALBUM_NAME)
    review_album_id     = get_or_create_album(REVIEW_ALBUM_NAME)

    processed_ids = set()
    last_check = datetime.now(timezone.utc).isoformat()

    while True:
        print(f"\n[sidecar] Polling at {datetime.now(timezone.utc).isoformat()}")
        now = datetime.now(timezone.utc).isoformat()

        assets = get_new_assets(last_check)

        for asset in assets:
            asset_id = asset.get("id")

            if asset_id in processed_ids:  # ← skip already processed
                continue

            try:
                # download image
                image_bytes = download_image(asset_id)
                if not image_bytes:
                    continue

                # score it
                result = score_image(asset_id, image_bytes)
                if not result:
                    continue

                decisions = result.get("decisions", {})

                # write score back to Immich
                write_score_to_immich(asset_id, result)

                # sort into albums
                if decisions.get("is_best_shot"):
                    add_to_album(asset_id, best_shots_album_id)
                elif decisions.get("review_flag"):
                    add_to_album(asset_id, review_album_id)
                
                processed_ids.add(asset_id) # mark as processed after successful handling

            except Exception as e:
                print(f"[sidecar] Error processing {asset_id}: {e}")

        last_check = now
        print(f"[sidecar] Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
