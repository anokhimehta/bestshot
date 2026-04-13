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

"""
BestShot train.py
Usage: python train.py --config config/baseline.yaml
"""

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
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_param("gpu_memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
        
        # git SHA
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except subprocess.CalledProcessError:
            git_sha = "unknown"
        mlflow.log_param("git_sha", git_sha)

        model = BestShotModel(config)
        trainer = L.Trainer(
            max_epochs=config['epochs'],
            accelerator=config.get('accelerator', 'gpu'),
            devices=1,
            callbacks=[EpochTimingCallback()]
        )

        start = time.time()
        trainer.fit(model, train_loader, validation_loader)

        eval_results = evaluate(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            data_dir=config['data_dir'],
            dataset_version=config.get('dataset_version', None)
        )
        mlflow.log_metric("plcc", eval_results["plcc"])
        mlflow.log_metric("srcc", eval_results["srcc"])
        mlflow.log_metric("eval_n_samples", eval_results["n_samples"])
        mlflow.log_metric("training_time_seconds", time.time() - start)
        mlflow.log_param("peak_vram_gb", round(torch.cuda.max_memory_allocated(0) / 1e9, 2))

        if eval_results["passed"]:
            mlflow.pytorch.log_model(
                model, 
                "model", 
                registered_model_name="bestshot-iqa")
            print("Model passed evaluation and has been registered.")
        else:
            mlflow.pytorch.log_model(model, "model")  # still saved as artifact, just not registered
            print("Model did not pass evaluation. Not registering.")



#4. Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BestShot model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)

