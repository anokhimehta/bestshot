# Configuration for CPU sequential benchmark
# This config simulates 100 users uploading 20 images each sequentially (total 2000 requests) to test the untrained model's performance on CPU without batching.
# note: same 7 photos will be used for all requests to keep it simple and consistents
CONFIG = {
    "mode": "sequential",
    "device": "cpu",
    "model_type": "pytorch",
    "num_users": 100,
    "uploads_per_user": 20,
    "max_workers": 1,
    "url": "http://127.0.0.1:8000/predict",
}
