# TODO: main training entrypoint
import argparse
import yaml
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import mlflow
import mlflow.pytorch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

"""
BestShot train.py
Usage: python train.py --config config/baseline.yaml
"""

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
        #use efficientnet_b3 as backbone
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 1)  # regression head

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

    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run():
        mlflow.log_params(config)

        model = BestShotModel(config)
        trainer = L.Trainer(
            max_epochs=config['epochs'],
            accelerator=config.get('accelerator', 'gpu'),  # defaults to gpu, overrideable
            devices=1
        )
        trainer.fit(model, train_loader, validation_loader)
        mlflow.pytorch.log_model(model, "model")



#4. Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BestShot model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)

