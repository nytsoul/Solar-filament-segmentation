import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
import json
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLODataset
from src.data.augmentations import get_training_augmentation, get_validation_augmentation
from src.models.unet_baseline import get_baseline_model

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import argparse

def configure_kaggle_paths(config):
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    if os.path.exists(kaggle_dir):
        config['dataset']['image_dir'] = os.path.join(kaggle_dir, 'train', 'train_images')
        config['dataset']['json_path'] = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        print(f"Detected Kaggle environment. Using dataset paths from {kaggle_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--limit_batches', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    configure_kaggle_paths(config)
        
    seed_everything(config['training'].get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_aug = get_training_augmentation(config)
    val_aug = get_validation_augmentation(config)
    
    train_dataset = MAGFiLODataset(
        csv_path=os.path.join(base_dir, config['dataset']['train_csv']),
        img_dir=config['dataset']['image_dir'] if os.path.isabs(config['dataset']['image_dir']) else os.path.join(base_dir, config['dataset']['image_dir']),
        json_path=config['dataset']['json_path'] if os.path.isabs(config['dataset']['json_path']) else os.path.join(base_dir, config['dataset']['json_path']),
        transforms=train_aug
    )
    val_dataset = MAGFiLODataset(
        csv_path=os.path.join(base_dir, config['dataset']['val_csv']),
        img_dir=config['dataset']['image_dir'] if os.path.isabs(config['dataset']['image_dir']) else os.path.join(base_dir, config['dataset']['image_dir']),
        json_path=config['dataset']['json_path'] if os.path.isabs(config['dataset']['json_path']) else os.path.join(base_dir, config['dataset']['json_path']),
        transforms=val_aug
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda')
    )
    
    model = get_baseline_model(config)
    if device.type == 'cuda' and config['training'].get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    epochs = config['training']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    bce_fn = torch.nn.BCEWithLogitsLoss()
    
    def combined_loss(y_pred, y_true):
        return loss_fn(y_pred, y_true) + bce_fn(y_pred, y_true)
        
    use_amp = config['training'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_dice = 0.0
    out_dir_base = args.out_dir if args.out_dir else base_dir
    checkpoints_dir = os.path.join(out_dir_base, 'checkpoints')
    outputs_dir = os.path.join(out_dir_base, 'outputs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    history = []
    print(f"Starting training for {epochs} epochs...")
    
    import itertools
    limit_batches = args.limit_batches
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        train_iter = itertools.islice(train_loader, limit_batches) if limit_batches else train_loader
        total_train = limit_batches if limit_batches else len(train_loader)
        
        for imgs, masks in tqdm(train_iter, desc=f"Epoch {epoch}/{epochs} [Train]", total=total_train):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = combined_loss(preds, masks)
                scaler.scale(loss).backward()
                
                if config['training'].get('gradient_clip_val'):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
                    
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                loss = combined_loss(preds, masks)
                loss.backward()
                if config['training'].get('gradient_clip_val'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
                optimizer.step()
                
            train_loss += loss.item()
            
        train_loss /= total_train
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        tp, fp, fn = 0, 0, 0
        
        val_iter = itertools.islice(val_loader, limit_batches) if limit_batches else val_loader
        total_val = limit_batches if limit_batches else len(val_loader)
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_iter, desc=f"Epoch {epoch}/{epochs} [Val]", total=total_val):
                imgs, masks = imgs.to(device), masks.to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        preds = model(imgs)
                        loss = combined_loss(preds, masks)
                else:
                    preds = model(imgs)
                    loss = combined_loss(preds, masks)
                    
                val_loss += loss.item()
                
                preds_binary = (torch.sigmoid(preds) > 0.5).int()
                masks_int = masks.int()
                
                tp += (preds_binary * masks_int).sum().item()
                fp += (preds_binary * (1 - masks_int)).sum().item()
                fn += ((1 - preds_binary) * masks_int).sum().item()
                
        val_loss /= total_val
        
        val_dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        val_iou = tp / (tp + fp + fn + 1e-7)
        
        epoch_duration = time.time() - epoch_start
        
        mem_info = ""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            mem_info = f" | GPU Mem: {allocated:.1f}GB/{reserved:.1f}GB"
            
        print(f"Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Time: {epoch_duration:.1f}s{mem_info}")
        
        epoch_data = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "duration_s": epoch_duration
        }
        history.append(epoch_data)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_baseline.pth'))
            print("  --> Saved new best model (Validation Dice)")
            
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'last_baseline.pth'))
        
        pd.DataFrame(history).to_csv(os.path.join(outputs_dir, 'training_history.csv'), index=False)
        with open(os.path.join(outputs_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == '__main__':
    main()
