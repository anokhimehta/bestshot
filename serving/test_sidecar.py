#!/usr/bin/env python3
"""
test_sidecar.py — Test sidecar connections before running full pipeline

Tests:
    1. Serving API health check
    2. Immich connectivity
    3. Immich API key validity
    4. Album creation
    5. Asset search
    6. Image download (first asset found)
    7. Full predict round trip with real image bytes

Usage:
    python test_sidecar.py

    # or with custom URLs
    IMMICH_URL=http://... IMMICH_API_KEY=... SERVING_URL=http://... python test_sidecar.py
"""

import os
import sys
import base64
import requests
from datetime import datetime, timezone, timedelta

IMMICH_URL  = os.getenv("IMMICH_URL", "http://localhost:2283").rstrip("/")
IMMICH_KEY  = os.getenv("IMMICH_API_KEY")
SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8000").rstrip("/")

HEADERS = {
    "x-api-key":    IMMICH_KEY,
    "Accept":       "application/json",
    "Content-Type": "application/json"
}

passed = 0
failed = 0

def test(name: str, fn):
    global passed, failed
    print(f"\n{'─'*50}")
    print(f"TEST: {name}")
    try:
        result = fn()
        print(f"  ✅ PASS: {result}")
        passed += 1
        return result
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed += 1
        return None

# ── Test 1: Serving health ────────────────────────────────────────────────────

def test_serving_health():
    resp = requests.get(f"{SERVING_URL}/health", timeout=10)
    assert resp.status_code == 200, f"Status {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"Unexpected response: {data}"
    return f"Serving is healthy — model: {data.get('model')}"

test("Serving API health", test_serving_health)

# ── Test 2: Immich connectivity ───────────────────────────────────────────────

def test_immich_ping():
    resp = requests.get(f"{IMMICH_URL}/api/server/ping", timeout=10)
    assert resp.status_code == 200, f"Status {resp.status_code} — is Immich running at {IMMICH_URL}?"
    return f"Immich is reachable at {IMMICH_URL}"

test("Immich connectivity", test_immich_ping)

# ── Test 3: API key validity ──────────────────────────────────────────────────

def test_immich_auth():
    assert IMMICH_KEY, "IMMICH_API_KEY not set"
    resp = requests.get(f"{IMMICH_URL}/api/users/me", headers=HEADERS, timeout=10)
    assert resp.status_code == 200, f"Status {resp.status_code} — API key may be invalid"
    data = resp.json()
    return f"Authenticated as: {data.get('email', 'unknown')}"

test("Immich API key", test_immich_auth)

# ── Test 4: List albums ───────────────────────────────────────────────────────

def test_list_albums():
    resp = requests.get(f"{IMMICH_URL}/api/albums", headers=HEADERS, timeout=10)
    assert resp.status_code == 200, f"Status {resp.status_code}"
    albums = resp.json()
    names = [a.get("albumName") for a in albums]
    return f"Found {len(albums)} albums: {names[:5]}"

test("List Immich albums", test_list_albums)

# ── Test 5: Create test album ─────────────────────────────────────────────────

def test_create_album():
    resp = requests.post(
        f"{IMMICH_URL}/api/albums",
        headers=HEADERS,
        json={"albumName": "BestShot Test Album — DELETE ME"},
        timeout=10
    )
    assert resp.status_code in (200, 201), f"Status {resp.status_code}: {resp.text}"
    album_id = resp.json()["id"]
    return f"Created album ID: {album_id}"

album_result = test("Create Immich album", test_create_album)

# ── Test 6: Search for assets ─────────────────────────────────────────────────

def test_search_assets():
    since = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    resp = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        headers=HEADERS,
        json={"updatedAfter": since, "type": "IMAGE"},
        timeout=15
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    items = resp.json().get("assets", {}).get("items", [])
    return f"Found {len(items)} assets in last year"

asset_result = test("Search Immich assets", test_search_assets)

# ── Test 7: Download an image ─────────────────────────────────────────────────

first_asset_id = None

def test_download_image():
    global first_asset_id
    since = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    resp = requests.post(
        f"{IMMICH_URL}/api/search/metadata",
        headers=HEADERS,
        json={"updatedAfter": since, "type": "IMAGE"},
        timeout=15
    )
    items = resp.json().get("assets", {}).get("items", [])
    assert items, "No assets found — upload at least one photo to Immich first"

    first_asset_id = items[0]["id"]
    img_resp = requests.get(
        f"{IMMICH_URL}/api/assets/{first_asset_id}/original",
        headers=HEADERS,
        timeout=30
    )
    assert img_resp.status_code == 200, f"Download failed: {img_resp.status_code}"
    size_kb = len(img_resp.content) / 1024
    return f"Downloaded asset {first_asset_id} ({size_kb:.1f} KB)"

test("Download Immich image", test_download_image)

# ── Test 8: Full predict round trip ──────────────────────────────────────────

def test_predict_with_image_bytes():
    assert first_asset_id, "Skipping — no asset downloaded in previous test"

    # download image
    img_resp = requests.get(
        f"{IMMICH_URL}/api/assets/{first_asset_id}/original",
        headers=HEADERS,
        timeout=30
    )
    assert img_resp.status_code == 200

    b64 = base64.b64encode(img_resp.content).decode()

    # send to serving
    pred_resp = requests.post(
        f"{SERVING_URL}/predict",
        json={
            "request_id": first_asset_id,
            "user_id": "test_sidecar",
            "images": [{
                "image_id":    first_asset_id,
                "image_bytes": b64,
                "image_path":  "",
                "metadata":    {}
            }]
        },
        timeout=30
    )
    assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.status_code} {pred_resp.text}"
    results = pred_resp.json().get("results", [])
    assert results, "Empty results"

    r = results[0]
    scores    = r.get("scores", {})
    decisions = r.get("decisions", {})
    return (
        f"composite={scores.get('composite_score')} "
        f"label={decisions.get('quality_label')} "
        f"best_shot={decisions.get('is_best_shot')}"
    )

test("Full predict round trip (Immich → Serving)", test_predict_with_image_bytes)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    print("\nFix the failing tests before running sidecar.py")
    sys.exit(1)
else:
    print("\nAll tests passed — sidecar is ready to run!")
    print(f"\nTo start the sidecar:")
    print(f"  python sidecar.py")
    print(f"  # or")
    print(f"  docker run --rm --network host --env-file .env bestshot-serve python sidecar.py")
