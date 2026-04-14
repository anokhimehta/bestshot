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
        self.device_type = CONFIG.get("device", "gup")
        
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
            
    '''def _init_pytorch(self):
        mlflow.set_tracking_uri("http://129.114.25.172:8000")

        # get version info before loading
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        version = client.get_model_version_by_alias("bestshot-iqa", "production")
        print(f"Loading model: bestshot-iqa version {version.version} (run_id: {version.run_id})")

        self.model = mlflow.pytorch.load_model("models:/bestshot-iqa@production")
        self.model.eval()
        self.device = torch.device("cuda" if self.device_type == "gpu" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("CUDA available:", torch.cuda.is_available())
        print("Selected device:", self.device)
        print("Model device:", next(self.model.parameters()).device)
        print(f"Loaded real model v{version.version} on: {self.device}")'''

    def _init_pytorch(self):
        mlflow.set_tracking_uri("http://129.114.25.172:8000")

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        version = client.get_model_version_by_alias("bestshot-iqa", "production")
        print(f"Loading model: bestshot-iqa version {version.version} (run_id: {version.run_id})")

        self.model = mlflow.pytorch.load_model("models:/bestshot-iqa@production")
        self.model.eval()

        # ROCm AMD GPU check — torch.cuda works for ROCm too but need to check differently
        if self.device_type == "gpu":
            if torch.cuda.is_available(): # is_rocm = torch.version.hip is not None
                self.device = torch.device("cuda")
            elif hasattr(torch, 'hip') and torch.hip.is_available():
                self.device = torch.device("cuda")  # ROCm uses cuda device name
            else:
                print("WARNING: GPU requested but no GPU found, falling back to CPU")
                self.device = torch.device("cuda") # mps? 
        else:
            self.device = torch.device("cuda")

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

        laplacian = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)

        edges = torch.nn.functional.conv2d(gray, laplacian, padding=1)
        scores = edges.var(dim=[1, 2, 3])
        return (scores * 500).clamp(0, 10)


    def compute_exposure(self, img_tensor):
        """Exposure score using PyTorch — runs on GPU"""
        # brightness = mean of all channels per image
        brightness = img_tensor.mean(dim=[1, 2, 3]) * 255  # (batch,)
        # penalize deviation from ideal brightness (128)
        score = 10.0 - (brightness - 128).abs() / 12.8
        return score.clamp(0, 10)  # shape: (batch,)

    """def compute_face_quality(self, img_path):
        import time
        t0 = time.time()
        if not os.path.exists(img_path):
            return random.uniform(5.0, 10.0)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # load cascades
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            # no faces — neutral score
            return 7.0
        
        # use the largest face
        best_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = best_face
        
        # face size score — larger face = better framing
        img_area  = 224 * 224
        face_area = w * h
        coverage  = face_area / img_area
        size_score = min(coverage * 50, 10.0)
        
        # eye check — look for eyes within the face region
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
        
        if len(eyes) >= 2:
            # both eyes detected — eyes are open
            eye_score = 10.0
        elif len(eyes) == 1:
            # only one eye detected — possibly closed or turned
            eye_score = 5.0
        else:
            # no eyes detected — likely closed eyes
            eye_score = 2.0
        
        # combine size and eye scores
        face_quality = (size_score * 0.4 + eye_score * 0.6)
        print(f"face_quality took: {(time.time()-t0)*1000:.1f}ms")
        return round(max(face_quality, 2.0), 4)"""

    def _init_onnx(self):
        # AMD-optimized ONNX setup
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider'] if self.device_type == "gpu" else ['CPUExecutionProvider']
        
        # Load the frozen model file
        self.session = ort.InferenceSession("model.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX active")
    
    def load_images(self, image_paths):# Preprocesses images into a batch tensor, this is called by the predict method to prepare the input for inference
        images = []
        for path in image_paths:
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.randn(3, 300, 300)  # fallback for missing images
            images.append(img_tensor)
        return torch.stack(images)

    def predict(self, image_paths):
        x_tensor = self.load_images(image_paths)
        x_gpu = x_tensor.to(self.device)

        print("Batch size:", len(image_paths))
        print("Batch shape:", x_tensor.shape[0])

        with torch.no_grad():
            raw_output = self.model(x_gpu)          # (batch,) koniq scores
            sharpness_scores = self.compute_sharpness(x_gpu)   # (batch,)
            exposure_scores  = self.compute_exposure(x_gpu)    # (batch,)

        results = []
        for i, path in enumerate(image_paths):
            koniq_score  = float(raw_output[i].item()) / 10.0
            sharpness    = float(sharpness_scores[i].item())
            exposure     = float(exposure_scores[i].item())
            #face_quality = self.compute_face_quality(path)  # CPU, per image
            face_quality = random.uniform(5.0, 10.0)

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
    image_paths = [img["image_path"] for img in images]
    return model.predict(image_paths)