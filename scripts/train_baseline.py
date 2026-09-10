import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
import json
import time
import logging
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLODataset
from src.data.augmentations import get_training_augmentation, get_validation_augmentation
from src.models.unet_baseline import get_baseline_model
from src.losses.dice import DiceLoss

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

def configure_kaggle_paths(config, base_dir):
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    if os.path.exists(kaggle_dir):
        config['dataset']['image_dir'] = os.path.join(kaggle_dir, 'train', 'train_images')
        config['dataset']['json_path'] = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        print(f"Detected Kaggle environment. Using dataset paths from {kaggle_dir}")
        return True
    return False

def plot_history(history, outputs_dir):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_dice = [h['val_dice'] for h in history]
    val_iou = [h['val_iou'] for h in history]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_dice, label='Val Dice', color='green')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_iou, label='Val IoU', color='orange')
    plt.title('Intersection over Union')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--limit_batches', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    in_kaggle = configure_kaggle_paths(config, base_dir)
    
    # Determine outputs path
    if in_kaggle and not args.out_dir:
        out_dir_base = '/kaggle/working'
    else:
        out_dir_base = args.out_dir if args.out_dir else base_dir
        
    checkpoints_dir = os.path.join(out_dir_base, 'checkpoints')
    outputs_dir = os.path.join(out_dir_base, 'outputs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(outputs_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Copy splits to output
    source_train_csv = os.path.abspath(os.path.join(base_dir, config['dataset']['train_csv']))
    source_val_csv = os.path.abspath(os.path.join(base_dir, config['dataset']['val_csv']))
    dest_train = os.path.abspath(os.path.join(outputs_dir, 'kaggle_train_split.csv'))
    dest_val = os.path.abspath(os.path.join(outputs_dir, 'kaggle_val_split.csv'))
    
    if os.path.exists(source_train_csv) and source_train_csv != dest_train:
        shutil.copy(source_train_csv, dest_train)
    if os.path.exists(source_val_csv) and source_val_csv != dest_val:
        shutil.copy(source_val_csv, dest_val)
        
    seed_everything(config['training'].get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_aug = get_training_augmentation(config)
    val_aug = get_validation_augmentation(config)
    
    train_dataset = MAGFiLODataset(
        csv_path=source_train_csv,
        img_dir=config['dataset']['image_dir'] if os.path.isabs(config['dataset']['image_dir']) else os.path.join(base_dir, config['dataset']['image_dir']),
        json_path=config['dataset']['json_path'] if os.path.isabs(config['dataset']['json_path']) else os.path.join(base_dir, config['dataset']['json_path']),
        transforms=train_aug
    )
    val_dataset = MAGFiLODataset(
        csv_path=source_val_csv,
        img_dir=config['dataset']['image_dir'] if os.path.isabs(config['dataset']['image_dir']) else os.path.join(base_dir, config['dataset']['image_dir']),
        json_path=config['dataset']['json_path'] if os.path.isabs(config['dataset']['json_path']) else os.path.join(base_dir, config['dataset']['json_path']),
        transforms=val_aug
    )
    
    logger.info(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    
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
    multi_gpu = False
    if device.type == 'cuda' and config['training'].get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = torch.nn.DataParallel(model)
        multi_gpu = True
        
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = DiceLoss()
    bce_fn = torch.nn.BCEWithLogitsLoss()
    
    def combined_loss(y_pred, y_true):
        return loss_fn(y_pred, y_true) + bce_fn(y_pred, y_true)
        
    use_amp = config['training'].get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    logger.info(f"AMP Status: {use_amp}")
    
    best_dice = 0.0
    history = []
    logger.info(f"Starting training for {epochs} epochs...")
    
    import itertools
    limit_batches = args.limit_batches
    
    start_time_total = time.time()
    best_epoch = 0
    best_iou = 0.0
    final_val_loss = 0.0
    
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
        final_val_loss = val_loss
        
        val_dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        val_iou = tp / (tp + fp + fn + 1e-7)
        
        epoch_duration = time.time() - epoch_start
        
        mem_info = ""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            mem_info = f" | GPU Mem: {allocated:.1f}GB/{reserved:.1f}GB"
            
        logger.info(f"Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Time: {epoch_duration:.1f}s{mem_info}")
        
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
            best_epoch = epoch
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_baseline.pth'))
            logger.info("  --> Saved new best model (Validation Dice)")
            
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'latest_checkpoint.pth'))
        
        pd.DataFrame(history).to_csv(os.path.join(outputs_dir, 'training_history.csv'), index=False)
        with open(os.path.join(outputs_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
            
        plot_history(history, outputs_dir)

    total_time = time.time() - start_time_total
    
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_val_iou": best_iou,
        "final_val_loss": final_val_loss,
        "total_training_duration_s": total_time,
        "num_train_images": len(train_dataset),
        "num_val_images": len(val_dataset),
        "multi_gpu_used": multi_gpu,
        "amp_used": use_amp,
        "best_checkpoint_path": os.path.join(checkpoints_dir, 'best_baseline.pth')
    }
    
    with open(os.path.join(outputs_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    logger.info("Training complete.")
    logger.info(json.dumps(final_metrics, indent=4))

if __name__ == '__main__':
    main()
