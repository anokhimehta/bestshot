# Configuration for GPU sequential processing
# This config simulates 100 users uploading 20 images each sequentially (total 2000 requests) to test the trained model's performance on GPU without batching.
CONFIG = {
    "mode": "gpu_sequential",
    "device": "gpu",
    "model_type": "pytorch",
    "num_users": 100,
    "uploads_per_user": 20,
    "max_workers": 1, 
    "url": "http://127.0.0.1:8000/predict",
}