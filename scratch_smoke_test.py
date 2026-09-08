import torch
import os
import sys
import yaml
import time
import json
import segmentation_models_pytorch as smp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.dataset import MAGFiLODataset
from src.data.augmentations import get_training_augmentation
from src.models.unet_baseline import get_baseline_model
from torch.utils.data import DataLoader

with open('configs/baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['dataset']['train_csv'] = 'outputs/kaggle_train_split.csv'
config['dataset']['val_csv'] = 'outputs/kaggle_val_split.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_baseline_model(config).to(device)

dataset = MAGFiLODataset(
    csv_path=config['dataset']['train_csv'],
    img_dir=config['dataset']['image_dir'],
    json_path=config['dataset']['json_path'],
    transforms=get_training_augmentation(config)
)
loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
bce_fn = torch.nn.BCEWithLogitsLoss()

model.train()
batch = next(iter(loader))
imgs, masks = batch[0].to(device), batch[1].to(device)

if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
start_time = time.time()

optimizer.zero_grad()
if scaler:
    with torch.amp.autocast('cuda'):
        preds = model(imgs)
        loss = loss_fn(preds, masks) + bce_fn(preds, masks)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    preds = model(imgs)
    loss = loss_fn(preds, masks) + bce_fn(preds, masks)
    loss.backward()
    optimizer.step()

batch_time = time.time() - start_time
mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0

report = {
    "device": device.type,
    "batch_size": config['training']['batch_size'],
    "loss": float(loss.item()),
    "batch_time_seconds": batch_time,
    "peak_memory_mb": mem_mb,
    "amp_enabled": scaler is not None
}

print(json.dumps(report, indent=2))
with open('outputs/kaggle_gpu_smoke_test.json', 'w') as f:
    json.dump(report, f, indent=2)
