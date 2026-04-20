#!/bin/bash
# Usage: ./test_feedback.sh
# Tests the feedback endpoint with different actions

# To make this script executeable and run it:
#   chmod +x serving/test_feedback.sh
#   Start the BestShot API server (if not already running): ./run.sh gpu_sequential
#   Then run this test script: ./serving/test_feedback.sh

# Expected output:
#   - Server health check passes (no ouput means success)
#   - Tests 1-4: all return status: ok (feedback logged to Swift)
#   - Test 5: returns error for invalid action
#   - Test 6: returns error for invalid feature
#   - Swift verification shows total entry count and last 3 logged entries


BASE_URL="http://127.0.0.1:8000"

echo "======================================"
echo "  Testing BestShot Feedback Endpoint"
echo "======================================"

# check server is running
if ! curl -s $BASE_URL/health > /dev/null; then
    echo "ERROR: Server not running at $BASE_URL"
    echo "Start it first with ./run.sh gpu_sequential"
    exit 1
fi
echo "Server is up!"
echo ""

# test 1 — deletion_suggestion: keep = negative feedback (disagreement)
echo "Test 1: User keeps image model flagged for deletion (disagreement)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-001",
    "photo_id": "test_img1.jpg",
    "user_id": "test_user",
    "feature": "deletion_suggestion",
    "action": "keep",
    "prediction": {
      "quality_label": "low_quality",
      "composite_score": 3.8,
      "is_best_shot": false,
      "review_flag": true
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 2 — deletion_suggestion: delete = positive feedback (agreement)
echo "Test 2: User deletes image model flagged for deletion (agreement)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-002",
    "photo_id": "test_img2.jpg",
    "user_id": "test_user",
    "feature": "deletion_suggestion",
    "action": "delete",
    "prediction": {
      "quality_label": "low_quality",
      "composite_score": 3.2,
      "is_best_shot": false,
      "review_flag": true
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 3 — best_shot: favorite = positive feedback (agreement)
echo "Test 3: User favorites image model flagged as best shot (agreement)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-003",
    "photo_id": "test_img3.jpg",
    "user_id": "test_user",
    "feature": "best_shot",
    "action": "favorite",
    "prediction": {
      "quality_label": "high_quality",
      "composite_score": 9.1,
      "is_best_shot": true,
      "review_flag": false
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 4 — best_shot: delete = negative feedback (disagreement → retrain signal)
echo "Test 4: User deletes image model flagged as best shot (disagreement)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-004",
    "photo_id": "test_img4.jpg",
    "user_id": "test_user",
    "feature": "best_shot",
    "action": "delete",
    "prediction": {
      "quality_label": "high_quality",
      "composite_score": 8.5,
      "is_best_shot": true,
      "review_flag": false
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 5 — invalid action
echo "Test 5: Invalid action (should return error)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-005",
    "photo_id": "test_img5.jpg",
    "user_id": "test_user",
    "feature": "best_shot",
    "action": "invalid_action"
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]} — {r.get(\"message\", \"\")}')"

# test 6 — invalid feature
echo "Test 6: Invalid feature (should return error)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-006",
    "photo_id": "test_img6.jpg",
    "user_id": "test_user",
    "feature": "invalid_feature",
    "action": "keep"
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]} — {r.get(\"message\", \"\")}')"

echo ""
echo "======================================"
echo "  Verifying entries in Swift bucket"
echo "======================================"

docker exec bestshot-api python3 -c "
import swiftclient, os, json
conn = swiftclient.Connection(
    authurl=os.getenv('OS_AUTH_URL'),
    auth_version='3',
    os_options={
        'auth_type': os.getenv('OS_AUTH_TYPE'),
        'application_credential_id': os.getenv('OS_APPLICATION_CREDENTIAL_ID'),
        'application_credential_secret': os.getenv('OS_APPLICATION_CREDENTIAL_SECRET'),
        'region_name': os.getenv('OS_REGION_NAME'),
    }
)
_, content = conn.get_object(os.getenv('BUCKET_NAME'), 'interactions_log.jsonl')
lines = [l for l in content.decode().strip().split('\n') if l]
print(f'Total entries in Swift: {len(lines)}')
print('Last 3 entries:')
for line in lines[-3:]:
    e = json.loads(line)
    eid = e.get('event_id', '')
    feat = e.get('feature', '')
    act = e.get('action', '')
    score = e.get('model_prediction', {}).get('composite_score', '')
    print(f'  {eid} | feature={feat} | action={act} | score={score}')
"

echo ""
echo "Done!"