# Configuration for CPU batch testing
# This config simulates 10 concurrent users uploading 20 images each (total 200 requests) to test the untrained model's performance on CPU with batching.
# note: same 7 photos will be used for all requests to keep it simple and consistent
CONFIG = {
    "mode": "cpu_batch",
    "device": "cpu",
    "model_type": "pytorch",
    "num_users": 10,          # 10 concurrent users
    "uploads_per_user": 20,   # each sends 20 requests
    "max_workers": 10,        # 10 threads running simultaneously
    "url": "http://127.0.0.1:8000/predict",
}