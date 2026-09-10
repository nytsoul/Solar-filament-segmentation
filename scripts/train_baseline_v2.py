"""
Full-Image Baseline v2 Training Script

Key changes from v1:
- Region-aware patch sampling (filament/disk/limb/background)
- Full-image validation using sliding-window inference (no CenterCrop)
- Best checkpoint saved by full-image validation Dice
- Comprehensive diagnostics and visualizations
"""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLOPatchDataset
from src.data.dataset_fullimg import MAGFiLOFullImageDataset
from src.data.augmentations import get_training_augmentation_v2, get_inference_augmentation
from src.models.unet_baseline import get_baseline_model
from src.losses.dice import DiceLoss
from src.inference.sliding_window import predict_full_image

import argparse


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
    val_dice = [h['val_dice_fullimg'] for h in history]
    val_iou = [h['val_iou_fullimg'] for h in history]
    pct_fil = [h.get('sampling_stats', {}).get('pct_patches_with_filament', 0) for h in history]
    pct_bg = [h.get('sampling_stats', {}).get('pct_patches_without_filament', 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_loss, 'o-', label='Train Loss', color='blue')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, val_dice, 'o-', label='Val Dice (Full Image)', color='green')
    axes[0, 1].set_title('Full-Image Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, val_iou, 'o-', label='Val IoU (Full Image)', color='orange')
    axes[1, 0].set_title('Full-Image IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, pct_fil, 'o-', label='% Patches with Filament', color='purple')
    axes[1, 1].plot(epochs, pct_bg, 's--', label='% Background/Limb/Disk Patches', color='gray')
    axes[1, 1].set_title('Patch Sampling Distribution')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_curves_v2.png'))
    plt.close()


def visualize_predictions(model, val_dataset, device, outputs_dir, config, num_images=5):
    """Save full-image prediction visualizations for representative val images."""
    vis_dir = os.path.join(outputs_dir, 'val_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    patch_size = config['dataset'].get('patch_size', 768)
    overlap = config.get('validation', {}).get('overlap', 0.25)

    num_images = min(num_images, len(val_dataset))

    with torch.no_grad():
        for i in range(num_images):
            img_tensor, mask_tensor, filename = val_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            mask_np = mask_tensor.squeeze().numpy()

            prob_map = predict_full_image(model, img_tensor, patch_size=patch_size, overlap=overlap, device=device)
            pred_binary = (prob_map > 0.5).astype(np.float32)

            # Compute per-image metrics
            intersection = (pred_binary * mask_np).sum()
            dice = (2 * intersection) / (pred_binary.sum() + mask_np.sum() + 1e-7)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Original image (denormalize approximately)
            img_vis = img_tensor.squeeze().cpu().numpy()
            img_vis = (img_vis * 0.5 + 0.5).clip(0, 1)
            axes[0].imshow(img_vis, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
            axes[2].set_title('Probability Map')
            axes[2].axis('off')

            axes[3].imshow(pred_binary, cmap='gray')
            axes[3].set_title(f'Prediction (Dice: {dice:.4f})')
            axes[3].axis('off')

            plt.suptitle(f'{filename}')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'val_pred_{filename}.png'))
            plt.close()


def full_image_validation(model, val_dataset, device, config, logger, use_amp=False):
    """
    Run full-image validation using sliding-window inference.
    
    Returns: (mean_dice, mean_iou)
    """
    model.eval()
    patch_size = config['dataset'].get('patch_size', 768)
    overlap = config.get('validation', {}).get('overlap', 0.25)

    all_dice = []
    all_iou = []

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Full-Image Val", leave=False):
            img_tensor, mask_tensor, filename = val_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)
            mask_np = mask_tensor.squeeze().numpy()  # (H, W)

            prob_map = predict_full_image(
                model, img_tensor,
                patch_size=patch_size,
                overlap=overlap,
                device=device
            )

            pred_binary = (prob_map > 0.5).astype(np.float32)

            # Pixel-level Dice and IoU
            intersection = (pred_binary * mask_np).sum()
            pred_sum = pred_binary.sum()
            gt_sum = mask_np.sum()
            union = pred_sum + gt_sum - intersection

            dice = (2 * intersection) / (pred_sum + gt_sum + 1e-7)
            iou = intersection / (union + 1e-7) if union > 0 else 0.0

            all_dice.append(dice)
            all_iou.append(iou)

    mean_dice = float(np.mean(all_dice))
    mean_iou = float(np.mean(all_iou))
    return mean_dice, mean_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline_v2.yaml')
    parser.add_argument('--limit_batches', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--val_limit', type=int, default=None,
                        help='Limit number of validation images for smoke testing')
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

    experiment_dir = os.path.join(out_dir_base, 'outputs', 'full_image_baseline_v2')
    checkpoints_dir = os.path.join(out_dir_base, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(experiment_dir, 'training_v2.log')
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
    dest_train = os.path.abspath(os.path.join(experiment_dir, 'kaggle_train_split.csv'))
    dest_val = os.path.abspath(os.path.join(experiment_dir, 'kaggle_val_split.csv'))

    if os.path.exists(source_train_csv) and source_train_csv != dest_train:
        shutil.copy(source_train_csv, dest_train)
    if os.path.exists(source_val_csv) and source_val_csv != dest_val:
        shutil.copy(source_val_csv, dest_val)

    # Save config
    with open(os.path.join(experiment_dir, 'config_v2.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    seed_everything(config['training'].get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Datasets ---
    img_dir = config['dataset']['image_dir']
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(base_dir, img_dir)
    json_path = config['dataset']['json_path']
    if not os.path.isabs(json_path):
        json_path = os.path.join(base_dir, json_path)

    train_aug = get_training_augmentation_v2(config)
    inf_aug = get_inference_augmentation()

    train_dataset = MAGFiLOPatchDataset(
        csv_path=source_train_csv,
        img_dir=img_dir,
        json_path=json_path,
        config=config,
        transforms=train_aug
    )

    # Full-image validation dataset
    val_csv_for_fullimg = source_val_csv
    if args.val_limit:
        # Create a temporary limited val CSV for smoke testing
        val_df_full = pd.read_csv(source_val_csv)
        val_df_limited = val_df_full.head(args.val_limit)
        limited_csv_path = os.path.join(experiment_dir, 'val_limited.csv')
        val_df_limited.to_csv(limited_csv_path, index=False)
        val_csv_for_fullimg = limited_csv_path

    val_dataset = MAGFiLOFullImageDataset(
        csv_path=val_csv_for_fullimg,
        img_dir=img_dir,
        json_path=json_path,
        transform=inf_aug
    )

    logger.info(f"Training patches per epoch: {len(train_dataset)} ({len(train_dataset.df)} images x {train_dataset.patches_per_image} patches)")
    logger.info(f"Validation images: {len(val_dataset)} (full 2048x2048)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=(device.type == 'cuda')
    )

    # --- Model ---
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
    logger.info(f"Sampling probabilities: filament={train_dataset.p_filament:.2f}, disk={train_dataset.p_disk:.2f}, limb={train_dataset.p_limb:.2f}, background={train_dataset.p_background:.2f}")

    best_dice = 0.0
    history = []
    logger.info(f"Starting training for {epochs} epochs...")

    import itertools
    limit_batches = args.limit_batches

    start_time_total = time.time()
    best_epoch = 0
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Reset sampling stats each epoch
        train_dataset.reset_stats()

        model.train()
        train_loss = 0.0
        epoch_strategy_counts = {'filament': 0, 'disk': 0, 'limb': 0, 'background': 0, 'fallback': 0}
        epoch_patches_with_filament = 0
        epoch_patches_without_filament = 0

        train_iter = itertools.islice(train_loader, limit_batches) if limit_batches else train_loader
        total_train = limit_batches if limit_batches else len(train_loader)

        for batch in tqdm(train_iter, desc=f"Epoch {epoch}/{epochs} [Train]", total=total_train):
            if len(batch) == 3:
                imgs, masks, meta = batch
                for s in meta['strategy']:
                    epoch_strategy_counts[s] = epoch_strategy_counts.get(s, 0) + 1
                has_fil = int(meta['has_filament'].sum().item())
                epoch_patches_with_filament += has_fil
                epoch_patches_without_filament += (len(imgs) - has_fil)
            else:
                imgs, masks = batch

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

        # --- Sampling diagnostics ---
        total_strat = sum(epoch_strategy_counts.values())
        total_fil = epoch_patches_with_filament + epoch_patches_without_filament
        if total_strat > 0:
            sampling_stats = {
                'sample_counts': epoch_strategy_counts,
                'total_patches': total_strat,
                'pct_filament_strategy': epoch_strategy_counts.get('filament', 0) / max(total_strat, 1) * 100,
                'pct_disk_strategy': epoch_strategy_counts.get('disk', 0) / max(total_strat, 1) * 100,
                'pct_limb_strategy': epoch_strategy_counts.get('limb', 0) / max(total_strat, 1) * 100,
                'pct_background_strategy': epoch_strategy_counts.get('background', 0) / max(total_strat, 1) * 100,
                'pct_patches_with_filament': epoch_patches_with_filament / max(total_fil, 1) * 100,
                'pct_patches_without_filament': epoch_patches_without_filament / max(total_fil, 1) * 100,
            }
        else:
            sampling_stats = train_dataset.get_stats()

        logger.info(f"  Sampling: filament={sampling_stats['pct_filament_strategy']:.1f}%, "
                     f"disk={sampling_stats['pct_disk_strategy']:.1f}%, "
                     f"limb={sampling_stats['pct_limb_strategy']:.1f}%, "
                     f"background={sampling_stats['pct_background_strategy']:.1f}%")
        logger.info(f"  Patches with filament: {sampling_stats['pct_patches_with_filament']:.1f}%, "
                     f"without: {sampling_stats['pct_patches_without_filament']:.1f}%")

        # --- Full-image validation ---
        # Need to handle DataParallel for validation
        val_model = model.module if multi_gpu else model
        val_dice, val_iou = full_image_validation(val_model, val_dataset, device, config, logger, use_amp)

        epoch_duration = time.time() - epoch_start

        mem_info = ""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            mem_info = f" | GPU Mem: {allocated:.1f}GB/{reserved:.1f}GB"

        logger.info(f"Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | "
                     f"Val Dice (Full): {val_dice:.4f} | Val IoU (Full): {val_iou:.4f} | "
                     f"Time: {epoch_duration:.1f}s{mem_info}")

        epoch_data = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_dice_fullimg": val_dice,
            "val_iou_fullimg": val_iou,
            "duration_s": epoch_duration,
            "sampling_stats": sampling_stats
        }
        history.append(epoch_data)

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_baseline_v2.pth'))
            logger.info("  --> Saved new best model (Full-Image Validation Dice)")

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'latest_checkpoint_v2.pth'))

        # Save history (exclude non-serializable sampling_stats nested dict)
        history_serializable = []
        for h in history:
            h_copy = {k: v for k, v in h.items() if k != 'sampling_stats'}
            h_copy.update({
                'pct_filament_patches': h['sampling_stats']['pct_patches_with_filament'],
                'pct_background_patches': h['sampling_stats']['pct_patches_without_filament'],
            })
            history_serializable.append(h_copy)

        pd.DataFrame(history_serializable).to_csv(os.path.join(experiment_dir, 'training_history_v2.csv'), index=False)
        with open(os.path.join(experiment_dir, 'training_history_v2.json'), 'w') as f:
            json.dump(history, f, indent=4, default=str)

        plot_history(history, experiment_dir)

    total_time = time.time() - start_time_total

    # --- End-of-training visualizations ---
    logger.info("Generating validation visualizations...")
    val_model = model.module if multi_gpu else model
    num_vis = config.get('validation', {}).get('num_vis_images', 5)
    visualize_predictions(val_model, val_dataset, device, experiment_dir, config, num_images=num_vis)

    final_metrics = {
        "experiment": "full_image_baseline_v2",
        "best_epoch": best_epoch,
        "best_val_dice_fullimg": best_dice,
        "best_val_iou_fullimg": best_iou,
        "total_training_duration_s": total_time,
        "num_train_images": len(train_dataset.df),
        "num_val_images": len(val_dataset),
        "patches_per_epoch": len(train_dataset),
        "multi_gpu_used": multi_gpu,
        "amp_used": use_amp,
        "best_checkpoint_path": os.path.join(checkpoints_dir, 'best_baseline_v2.pth'),
        "sampling": {
            "p_filament": train_dataset.p_filament,
            "p_disk": train_dataset.p_disk,
            "p_limb": train_dataset.p_limb,
            "p_background": train_dataset.p_background,
        },
        "architecture": config['model']['architecture'],
        "encoder": config['model']['encoder_name'],
    }

    with open(os.path.join(experiment_dir, 'final_metrics_v2.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    logger.info("Training complete.")
    logger.info(json.dumps(final_metrics, indent=4))


if __name__ == '__main__':
    main()
