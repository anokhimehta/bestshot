"""
BestShot evaluate.py
 
Loads a registered MLflow model, runs inference on the eval split,
computes PLCC and SRCC against ground truth MOS scores, and returns
pass/fail based on defined thresholds.
 
Usage (standalone):
    python evaluate.py --model-version 3 --data-dir /tmp/koniq10k
    python evaluate.py --model-uri models:/bestshot-iqa/latest --data-dir /tmp/koniq10k
 
Called from train.py after training to gate model registration.
"""

import argparse
import sys
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------------ #
# Thresholds — adjust if best run hits different numbers         #
# ------------------------------------------------------------------ #
PLCC_THRESHOLD = 0.80
SRCC_THRESHOLD = 0.78


def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_inference(model, image_paths, device, batch_size=32):
    transform = get_transform()
    model.eval()
    model.to(device)

    predictions = []    
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        tensors = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                tensors.append(transform(image))
            except Exception as e:
                print(f"Error opening image {path}: {e}")
                tensors.append(torch.zeros(3, 300, 300))  # Placeholder for failed image
        
        batch_tensor = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            
        predictions.extend(outputs.squeeze().cpu().tolist())
    
    return predictions

def evaluate(model_uri, data_dir, dataset_version=None):
    # Core evaluation function. Can be called from train.py directly.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from MLflow
    print(f"Loading model from {model_uri}...")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()

    # Load eval dataset
    data_dir = Path(data_dir)
    if dataset_version is not None:
        # From batch pipeline: labels/v{n}/eval.csv
        # Columns: image_path, quality_score, label, split, source, ...
        csv_path = data_dir / "labels" / f"v{dataset_version}" / "eval.csv"
        print(f"Loading versioned eval split from {csv_path} ...")
        df = pd.read_csv(csv_path)

        def resolve_path(swift_path):
            filename = Path(swift_path).name
            return data_dir / "512x384" / filename
        
        df["local_path"] = df["image_path"].apply(resolve_path)
        ground_truth = df["quality_score"].tolist()

    else:
        # Raw KonIQ CSV — used for local testing before batch pipeline is wired up
        csv_path = data_dir / "koniq10k_scores_and_distributions.csv"
        print(f"Loading raw KonIQ eval split from {csv_path} ...")
        df = pd.read_csv(csv_path)

        # Filter for eval split
        n = len(df)
        eval_df = df.iloc[int(n*0.8):]  # Last 20% for eval

        def resolve_path(filename):
            return data_dir / "512x384" / filename
        
        eval_df = eval_df.copy()
        eval_df["local_path"] = eval_df["image_name"].apply(resolve_path)
        #normalize MOS scores to 0-10 range
        eval_df["quality_score"] = (eval_df["MOS"] - 1) / 4 * 10
        df = eval_df 
        ground_truth = df["quality_score"].tolist()

    image_paths = df["local_path"].tolist()
    
    # Run inference
    predictions = run_inference(model, image_paths, device)
    
    # Filter out any samples where image failed to load
    # (they got zero tensors and will skew metrics — remove them)
    valid = [
        (p, g) for p, g, path in zip(predictions, ground_truth, image_paths)
        if Path(path).exists()
    ]
    if len(valid) < len(predictions):
        print(f"  Dropped {len(predictions) - len(valid)} samples with missing images")
    predictions = [p for p, _ in valid]
    ground_truth = [g for _, g in valid]
    
    # Compute PLCC and SRCC
    plcc, _ = pearsonr(predictions, ground_truth)
    srcc, _ = spearmanr(predictions, ground_truth)

    passed = (plcc >= PLCC_THRESHOLD) and (srcc >= SRCC_THRESHOLD)
    
    print()
    print("=" * 40)
    print(f"  PLCC:     {plcc:.4f}  (threshold: {PLCC_THRESHOLD})")
    print(f"  SRCC:     {srcc:.4f}  (threshold: {SRCC_THRESHOLD})")
    print(f"  Samples:  {len(predictions)}")
    print(f"  Result:   {'PASS ✅' if passed else 'FAIL ❌'}")
    print("=" * 40)
 
    return {
        "plcc": plcc,
        "srcc": srcc,
        "passed": passed,
        "n_samples": len(predictions),
    }