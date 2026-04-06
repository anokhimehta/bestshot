import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
import onnxruntime as ort
from config import CONFIG 

class Model:
    def __init__(self):
        # Pull model type and device from config to determine which inference engine to initialize
        self.model_type = CONFIG.get("model_type", "pytorch")
        self.device_type = CONFIG.get("device", "cpu")
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])

        # Initialize the selected inferencing engine
        if self.model_type == "onnx":
            self._init_onnx()
        else:
            self._init_pytorch()

    def _init_pytorch(self):
        #ROCm/pytorch setup
        self.device = torch.device("cuda" if self.device_type == "gpu" and torch.cuda.is_available() else "cpu")
        self.model = models.efficientnet_b3(weights=None)
        self.model.to(self.device)
        self.model.eval()
        print(f"PyTorch Engine active on: {self.device}")

    def _init_onnx(self):
        # AMD-optimized ONNX setup
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider'] if self.device_type == "gpu" else ['CPUExecutionProvider']
        
        # Load the frozen model file
        self.session = ort.InferenceSession("model.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX Engine active with providers: {self.session.get_providers()}")

    def load_images(self, image_paths): # Preprocesses images into a batch tensor, this is called by the predict method to prepare the input for inference
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            # We keep tensors on CPU initially for easier conversion to NumPy for ONNX
            img_tensor = self.transform(img) 
            images.append(img_tensor)
        return torch.stack(images)

    def predict(self, image_paths):
        # Load and prepare the batch
        x_tensor = self.load_images(image_paths)
        
        # Inferencing with the selected engine, simulating the double-pass workload
        if self.model_type == "onnx":
            # Convert to NumPy for ONNX Runtime
            onnx_inputs = {self.input_name: x_tensor.numpy()}
            # Simulate the double-pass workload by running the model twice
            _ = self.session.run(None, onnx_inputs) 
            _ = self.session.run(None, onnx_inputs)
        else:
            # Move to GPU if needed and run the model twice to simulate the workload
            x_gpu = x_tensor.to(self.device)
            with torch.no_grad():
                _ = self.model(x_gpu)
                _ = self.model(x_gpu)

        # Calculate mock scores and decisions to simulate the output structure
        results = []
        for _ in range(len(image_paths)):
            scores = {
                "koniq_score": random.uniform(0.5, 1.0),
                "sharpness": random.uniform(0.5, 1.0),
                "exposure": random.uniform(0.5, 1.0),
                "face_quality": random.uniform(0.5, 1.0),
            }
            composite = sum(scores.values()) / len(scores)
            scores["composite"] = composite

            decisions = {
                "quality_label": "high_quality" if composite > 0.8 else "low_quality",
                "review_flag": "keep" if composite > 0.7 else "review_for_deletion",
                "is_best_shot": composite > 0.9,
                "is_burst": False,
                "burst_group_id": None
            }

            results.append({"scores": scores, "decisions": decisions})
        return results

# FastAPI Helper Functions

_model_instance = None

def load_model(): # Singleton pattern to ensure we only load the model once per server instance
    global _model_instance
    if _model_instance is None:
        _model_instance = Model()
    return _model_instance

def predict_batch(model, images): # This function is called by the FastAPI endpoint, it extracts image paths and calls the model's predict method
    image_paths = [img["image_path"] for img in images]
    return model.predict(image_paths)