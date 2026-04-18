#!/bin/bash
# Usage: ./test_feedback.sh
# Tests the feedback endpoint with different actions

# To make this script executeable and run it:
#   chmod +x serving/test_feedback.sh
#   ./serving/test_feedback.sh

# Expected output:
#   Server is up!
#   Test 1: User keeps image model flagged as low quality...
#       status: ok
#   Test 2: User deletes image model scored as high quality...
#       status: ok
#   Test 3: User favorites an image...
#       status: ok
#   Test 4: Invalid action (should return error)...
#       status: error — Invalid action. Must be one of: {'keep', 'delete', 'favorite'} 
# ... Done!


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

# test 1 — keep (model said low quality, user disagreed)
echo "Test 1: User keeps image model flagged as low quality..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-001",
    "photo_id": "test_img1.jpg",
    "user_id": "test_user",
    "action": "keep",
    "prediction": {
      "quality_label": "low_quality",
      "composite_score": 3.8,
      "is_best_shot": false,
      "review_flag": true
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 2 — delete (model said high quality, user disagreed)
echo "Test 2: User deletes image model scored as high quality..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-002",
    "photo_id": "test_img2.jpg",
    "user_id": "test_user",
    "action": "delete",
    "prediction": {
      "quality_label": "high_quality",
      "composite_score": 8.2,
      "is_best_shot": true,
      "review_flag": false
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 3 — favorite
echo "Test 3: User favorites an image..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-003",
    "photo_id": "test_img3.jpg",
    "user_id": "test_user",
    "action": "favorite",
    "prediction": {
      "quality_label": "high_quality",
      "composite_score": 9.1,
      "is_best_shot": true,
      "review_flag": false
    }
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'  status: {r[\"status\"]}')"

# test 4 — invalid action
echo "Test 4: Invalid action (should return error)..."
curl -s -X POST $BASE_URL/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test-004",
    "photo_id": "test_img4.jpg",
    "user_id": "test_user",
    "action": "invalid_action"
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
# show last 3
print('Last 3 entries:')
for line in lines[-3:]:
    entry = json.loads(line)
    print(f'  {entry[\"event_id\"]} | action={entry[\"action\"]} | score={entry[\"model_prediction\"][\"composite_score\"]}')
"

echo ""
echo "Done!"