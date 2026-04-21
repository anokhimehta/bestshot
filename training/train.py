import argparse
import yaml
import pandas as pd
import subprocess
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import mlflow
import mlflow.pytorch
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from evaluate import run_inference, evaluate

import io
import swiftclient
from dotenv import load_dotenv
import os

BUCKET = 'ak12754-data-proj19'

"""
BestShot train.py
Usage: python train.py --config config/baseline.yaml
"""

'''
1. Dataset loading and preprocessing
'''
def get_swift_conn():
    load_dotenv('/workspace/.env')
    load_dotenv('/home/cc/bestshot/.env')
    return swiftclient.Connection(
        auth_version='3',
        authurl=os.environ['OS_AUTH_URL'],
        os_options={
            'application_credential_id': os.environ['OS_APPLICATION_CREDENTIAL_ID'],
            'application_credential_secret': os.environ['OS_APPLICATION_CREDENTIAL_SECRET'],
            'region_name': os.environ['OS_REGION_NAME'],
            'auth_type': 'v3applicationcredential'
        }
    )


def download_dataset(conn, local_dir):
    local_dir = Path(local_dir)
    img_dir = local_dir / '512x384'
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download CSV
    csv_local = local_dir / 'koniq10k_scores_and_distributions.csv'
    if not csv_local.exists():
        print("Downloading scores CSV from object storage...")
        _, content = conn.get_object(BUCKET, 'koniq10k/koniq10k_scores_and_distributions.csv')
        csv_local.write_bytes(content)
        print("CSV downloaded.")

    # Download images (skip already cached)
    df = pd.read_csv(csv_local)
    missing = [name for name in df['image_name'] if not (img_dir / name).exists()]
    print(f"Downloading {len(missing)} missing images (skipping {len(df) - len(missing)} cached)...")
    for image_name in missing:
        _, img_data = conn.get_object(BUCKET, f'koniq10k/images/{image_name}')
        (img_dir / image_name).write_bytes(img_data)

    # Return latest dataset version for lineage logging
    try:
        _, objects = conn.get_container(BUCKET, prefix='labels/')
        versions = set()
        for obj in objects:
            parts = obj['name'].split('/')
            if len(parts) >= 2 and parts[1].startswith('v'):
                versions.add(int(parts[1][1:]))
        return f"v{max(versions)}" if versions else "koniq10k-only"
    except:
        return "koniq10k-only"


class EpochTimingCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self._epoch_start
        pl_module.log("time_per_epoch_seconds", epoch_time)
        
#use for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
#1. Dataset loading and preprocessing
class KonIQDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        data_dir = Path(data_dir)
        df = pd.read_csv(data_dir / "koniq10k_scores_and_distributions.csv")
        self.image_paths = [data_dir / "512x384" / name for name in df["image_name"]]
        #normalize scores to [0, 10] from [1, 5]
        self.scores = ((df["MOS"] - 1) / 4 * 10).tolist()  # normalize to 0-10
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return image, score


#2. Model definition (e.g., ResNet, ViT)
class BestShotModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        # Freeze early stages based on config
        frozen_stages = config.get('frozen_stages', 0)
        stages = list(self.backbone.blocks.children())
        for i, stage in enumerate(stages):
            if i < frozen_stages:
                for param in stage.parameters():
                    param.requires_grad = False
        
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        score = self.head(features)
        return score.squeeze(1)  # output shape (batch,)
    
    def training_step(self, batch, _):
        images, scores = batch
        preds = self(images)
        loss = F.mse_loss(preds, scores)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, _):
        images, scores = batch
        preds = self(images)
        loss = F.mse_loss(preds, scores)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

#3. Training loop with logging and checkpointing
def main(config):

    set_seed(config.get("seed", 42))

    # Download dataset from object storage if not already cached locally
    conn = get_swift_conn()
    dataset_version = download_dataset(conn, config['data_dir'])

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        #using mean and standard deviation of the ImageNet dataset, computed across millions of images, per channel (R, G, B).
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = KonIQDataset(config['data_dir'], transform=transform)
    if config.get('limit'):
        dataset = torch.utils.data.Subset(dataset, range(config['limit']))
    n = len(dataset)
    train_ds, validation_ds = random_split(dataset, [int(n * 0.8), n - int(n * 0.8)])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config.get('num_workers', 4))
    validation_loader = DataLoader(validation_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4))

    mlflow.pytorch.autolog(log_models=False)  # we'll log manually to control registration
    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(config)
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
        else:
            mlflow.log_param("gpu_name", "none")
            mlflow.log_param("gpu_memory_gb", 0)
        mlflow.log_param("dataset_version", dataset_version)

        model = BestShotModel(config)
        accelerator = config.get('accelerator', 'gpu')
        if accelerator == 'gpu' and not has_gpu:
            accelerator = 'cpu'
        trainer = L.Trainer(
            max_epochs=config['epochs'],
            accelerator=accelerator,
            devices=1,
            callbacks=[EpochTimingCallback()]
        )

        start = time.time()
        trainer.fit(model, train_loader, validation_loader)

        #save model so it can be loaded for evaluation, even if it doesn't pass the criteria to be registered as "bestshot-iqa"
        mlflow.pytorch.log_model(model, "model")

        # Run evaluation script and log results
        eval_results = evaluate(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            data_dir=config['data_dir'],
            dataset_version=config.get('dataset_version', None)
        )
        mlflow.log_metric("plcc", eval_results["plcc"])
        mlflow.log_metric("srcc", eval_results["srcc"])
        mlflow.log_metric("eval_n_samples", eval_results["n_samples"])
        mlflow.log_metric("training_time_seconds", time.time() - start)
        if has_gpu:
            mlflow.log_param("peak_vram_gb", round(torch.cuda.max_memory_allocated(0) / 1e9, 2))
        else:
            mlflow.log_param("peak_vram_gb", 0)

        if eval_results["passed"]:
            run_id = mlflow.active_run().info.run_id
            mv = mlflow.register_model(f"runs:/{run_id}/model", "bestshot-iqa")
            client = mlflow.tracking.MlflowClient()
            # New successful models land in Staging; promotion automation
            # is responsible for moving them to Production.
            client.transition_model_version_stage(
                "bestshot-iqa",
                mv.version,
                "Staging"
            )
            print("Model passed evaluation and has been registered to Staging.")
        else:
            print(f"Did not pass quality gate (PLCC={eval_results['plcc']:.4f}, SRCC={eval_results['srcc']:.4f}) — not registered")




#4. Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BestShot model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)

