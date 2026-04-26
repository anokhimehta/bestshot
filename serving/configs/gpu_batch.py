# Configuration for GPU batch processing
# This config simulates 100 users uploading 20 images each in batches (total 2000 requests) to test the trained model's performance on GPU with batching.
CONFIG = {
    "mode": "batch_gpu",
    "device": "gpu",
    "model_type": "pytorch",
    "num_users": 100,
    "uploads_per_user": 20,
    "max_workers": 20,
    "url": "http://127.0.0.1:8000/predict",
}