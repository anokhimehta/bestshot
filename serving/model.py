import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import random
import os

class Model:
    def __init__(self):
        # Determine device: use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device: use CUDA if available
        
        self.model = models.efficientnet_b3(weights=None)
        self.model.to(self.device)  # Move model to GPU
        self.model.eval()
        # preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),  # converts to [3, H, W]
        ])

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            img = self.transform(img).to(self.device) # Move image tensor to GPU
            images.append(img)
        return torch.stack(images)

    def predict(self, image_paths):
        x = self.load_images(image_paths) # Load and preprocess images, move to GPU
        with torch.no_grad():
            _ = self.model(x) # Math happens on AMD compute units 
            _ = self.model(x) # we ignore the actual output since this is a dummy model, we just want to simulate the latency and return random scores

        results = []
        for i in range(len(image_paths)):
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

# helper functions for FastAPI
_model_instance = None
def load_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = Model()
    return _model_instance

def predict_batch(model, images):
    image_paths = [img["image_path"] for img in images]
    return model.predict(image_paths)