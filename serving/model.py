import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import mlflow
import mlflow.pytorch 
import numpy as np
import onnxruntime as ort
import cv2

class Model:

    def __init__(self):
        # Pull model type and device from config to determine which inference engine to initialize
        from config import CONFIG 
        self.model_type = CONFIG.get("model_type", "pytorch")
        self.device_type = CONFIG.get("device", "gpu")
        
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
        mlflow.set_tracking_uri(
            os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://mlflow.bestshot-platform.svc.cluster.local:5000"
            )
        )

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        version = client.get_model_version_by_alias("bestshot-iqa", "production")
        #print(f"Loading model: bestshot-iqa version {latest.version} (run_id: {latest.run_id})")
        print(f"Loading model: bestshot-iqa version {version.version} (run_id: {version.run_id})")

        self.model = mlflow.pytorch.load_model("models:/bestshot-iqa@production")
        self.model.eval()

        # Respect explicit override first; otherwise use config with safe fallback.
        device_override = os.getenv("DEVICE", "").strip().lower()
        if device_override in {"cpu", "cuda"}:
            self.device = torch.device(device_override)
        elif self.device_type == "gpu" and torch.cuda.is_available():
            # ROCm also uses "cuda" as the torch device name.
            self.device = torch.device("cuda")
        else:
            print("GPU unavailable or not requested, falling back to CPU")
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"CUDA available: {torch.cuda.is_available()}")
        #print(f"ROCm version: {torch.version.hip if hasattr(torch, 'version') and hasattr(torch.version, 'hip') else 'N/A'}")
        #print(f"ROCm available: {torch.hip.is_available()}")
        print(f"Selected device: {self.device}")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Loaded real model v{version.version} on: {self.device}")

    def compute_sharpness(self, img_tensor):
        gray = 0.299 * img_tensor[:, 0] + \
            0.587 * img_tensor[:, 1] + \
            0.114 * img_tensor[:, 2]
        gray = gray.unsqueeze(1)

        laplacian_kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        edges = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)

        variance = edges.var(dim=[1, 2, 3])

        # normalize to match data team but keep 0-10 range for interpretability ASK data about why did 0-1
        score = (variance * 300).clamp(0, 10)

        return score

    def compute_exposure(self, img_tensor):
        gray = 0.299 * img_tensor[:, 0] + \
            0.587 * img_tensor[:, 1] + \
            0.114 * img_tensor[:, 2]

        # brightness (0–1)
        brightness = gray.mean(dim=[1, 2])

        # contrast (std dev)
        contrast = gray.std(dim=[1, 2])

        # ideal brightness ~0.5
        brightness_score = 1.0 - (brightness - 0.5).abs() * 2

        # encourage some contrast (not flat)
        contrast_score = (contrast * 2).clamp(0, 1)

        # combine
        score = (0.7 * brightness_score + 0.3 * contrast_score) * 10

        return score.clamp(0, 10)

    def compute_face_quality(self, img_tensor):
        B, C, H, W = img_tensor.shape

        # center crop (assume face likely here)
        h1, h2 = int(H * 0.25), int(H * 0.75)
        w1, w2 = int(W * 0.25), int(W * 0.75)

        center = img_tensor[:, :, h1:h2, w1:w2]

        # sharpness logic from above 
        gray = 0.299 * center[:, 0] + \
            0.587 * center[:, 1] + \
            0.114 * center[:, 2]
        gray = gray.unsqueeze(1)

        laplacian_kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        edges = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
        sharpness = edges.var(dim=[1, 2, 3])

        sharpness_score = (sharpness * 300).clamp(0, 10)

        # brightness
        brightness = gray.mean(dim=[1, 2, 3])
        brightness_score = (1.0 - (brightness - 0.5).abs() * 2) * 10

        # combine
        score = 0.6 * sharpness_score + 0.4 * brightness_score

        return score.clamp(0, 10)

    def _init_onnx(self):
        # AMD-optimized ONNX setup
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider'] if self.device_type == "gpu" else ['CPUExecutionProvider']
        
        # Load the frozen model file
        self.session = ort.InferenceSession("model.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX active")
    

    def load_images(self, images):
        #Accept either file paths or base64 image bytes, need to handle both for sidecar/ immich integration and testing with local files. Returns a batch tensor of shape (batch, 3, 300, 300) ready for model input.
        tensors = []
        for img in images:
            try:
                if isinstance(img, dict) and img.get("image_bytes"):
                    # base64 encoded bytes from sidecar/Immich
                    import base64
                    from io import BytesIO
                    raw = base64.b64decode(img["image_bytes"])
                    pil_img = Image.open(BytesIO(raw)).convert("RGB")
                    img_tensor = self.transform(pil_img)
                elif isinstance(img, dict) and img.get("image_path"):
                    path = img["image_path"]
                    if os.path.exists(path):
                        pil_img = Image.open(path).convert("RGB")
                        img_tensor = self.transform(pil_img)
                    else:
                        img_tensor = torch.randn(3, 300, 300)
                else:
                    img_tensor = torch.randn(3, 300, 300)
            except Exception as e:
                print(f"[model] Error loading image: {e}")
                img_tensor = torch.randn(3, 300, 300)
            
            tensors.append(img_tensor)
        return torch.stack(tensors)

    def predict(self, images):  # ← renamed from image_paths to images
        x_tensor = self.load_images(images)
        x_gpu = x_tensor.to(self.device)

        print("Batch size:", len(images))
        print("Batch shape:", x_tensor.shape[0])
        print("Input device:", x_gpu.device)

        with torch.no_grad():
            raw_output       = self.model(x_gpu)
            sharpness_scores = self.compute_sharpness(x_gpu)
            exposure_scores  = self.compute_exposure(x_gpu)

        results = []
        for i, img in enumerate(images):  # ← img not path
            koniq_score  = float(raw_output[i].item()) / 10.0
            sharpness    = float(sharpness_scores[i].item())
            exposure     = float(exposure_scores[i].item())
            
            # get path for face quality if available
            path = img.get("image_path", "") if isinstance(img, dict) else ""
            face_quality = random.uniform(5.0, 10.0)  # placeholder

            composite = (koniq_score  * 0.4 +
                        sharpness    * 0.3 +
                        exposure     * 0.2 +
                        face_quality * 0.1)

            scores = {
                "koniq_score":     float(round(koniq_score, 4)),
                "sharpness":       float(round(sharpness, 4)),
                "exposure":        float(round(exposure, 4)),
                "face_quality":    float(round(face_quality, 4)),
                "composite_score": float(round(composite, 4)),
            }

            decisions = {
                "quality_label": "high_quality" if composite > 7.0 else "low_quality",
                "review_flag":   bool(composite < 5.0),
                "is_best_shot":  bool(composite > 8.5),
                "is_burst":      False,
                "burst_group_id": None
            }
            print("Input device:", x_gpu.device) # TESTING LINE
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
    '''image_paths = [img["image_path"] for img in images]
    return model.predict(image_paths)''' 
    return model.predict(images)