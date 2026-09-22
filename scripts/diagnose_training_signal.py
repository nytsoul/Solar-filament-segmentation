import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLOPatchDataset
from src.data.dataset_fullimg import MAGFiLOFullImageDataset
from src.data.augmentations import get_training_augmentation_v2, get_inference_augmentation
from src.models.unet_baseline import get_baseline_model
from src.losses.dice import DiceLoss

def get_stats(tensor, is_mask=False):
    t_np = tensor.detach().cpu().numpy()
    stats = {
        'shape': list(t_np.shape),
        'min': float(np.min(t_np)),
        'max': float(np.max(t_np)),
        'mean': float(np.mean(t_np)),
        'std': float(np.std(t_np))
    }
    if is_mask:
        stats['unique_values'] = np.unique(t_np).tolist()
        stats['fg_percentage'] = float(np.mean(t_np > 0)) * 100
        stats['fg_pixels'] = int(np.sum(t_np > 0))
        # Connected components on the first item in batch
        mask_2d = (t_np[0, 0] > 0).astype(np.uint8)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
        stats['num_components'] = num_labels - 1
    return stats

def analyze_model_output(model, imgs, masks, is_train=True):
    # Output BEFORE training (random weights)
    # We will instantiate a fresh model for this
    pass

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'baseline_v2.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Force patch size for diagnostic to be reasonable if not set
    config['dataset']['patch_size'] = config['dataset'].get('patch_size', 768)

    outputs_dir = os.path.join(base_dir, 'outputs')
    vis_dir = os.path.join(outputs_dir, 'training_signal_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    img_dir = config['dataset']['image_dir']
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(base_dir, img_dir)
    json_path = config['dataset']['json_path']
    if not os.path.isabs(json_path):
        json_path = os.path.join(base_dir, json_path)
    
    train_csv = os.path.join(base_dir, 'outputs', 'kaggle_train_split.csv')
    val_csv = os.path.join(base_dir, 'outputs', 'kaggle_val_split.csv')

    train_aug = get_training_augmentation_v2(config)
    
    # We use PatchDataset to see exactly what the model sees during training
    train_dataset = MAGFiLOPatchDataset(
        csv_path=train_csv,
        img_dir=img_dir,
        json_path=json_path,
        config=config,
        transforms=train_aug,
        return_meta=True
    )
    
    val_dataset = MAGFiLOPatchDataset(
        csv_path=val_csv,
        img_dir=img_dir,
        json_path=json_path,
        config=config,
        transforms=train_aug,
        return_meta=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model 1: Untrained
    model_untrained = get_baseline_model(config).to(device)
    
    # Model 2: Trained (best_baseline_v2)
    model_trained = get_baseline_model(config).to(device)
    ckpt_path = os.path.join(base_dir, 'checkpoints', 'best_baseline_v2.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = '/kaggle/input/datasets/nytsoul/solar-filament-segmentation-6/solar-filament-segmentation/checkpoints/best_baseline_v2.pth'
    state_dict = torch.load(ckpt_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_trained.load_state_dict(state_dict, strict=True)

    bce_fn = torch.nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    results = []
    
    # Process 10 train and 10 val
    for split_name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        # Filter to make sure we get some patches with filaments
        samples_collected = 0
        idx = 0
        while samples_collected < 10 and idx < len(dataset):
            img, mask, meta = dataset[idx]
            idx += 1
            
            # To ensure a good mix, we force the first 5 to have filaments if possible
            if samples_collected < 5 and meta['has_filament'] == 0:
                continue
                
            img_b = img.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            
            res = {
                'split': split_name,
                'sample_idx': samples_collected,
                'strategy': meta['strategy'],
                'has_filament': meta['has_filament']
            }
            
            # 1. Raw image stats
            res['image_stats'] = get_stats(img_b, is_mask=False)
            
            # 2. GT mask stats
            res['gt_stats'] = get_stats(mask_b, is_mask=True)
            
            # 3. Untrained model output
            with torch.no_grad():
                model_untrained.eval()
                logits_u = model_untrained(img_b)
                probs_u = torch.sigmoid(logits_u)
                res['untrained_stats'] = get_stats(probs_u, is_mask=False)
                res['untrained_fg_050'] = float((probs_u > 0.5).float().mean() * 100)
            
            # 4. Trained model output & Gradients
            model_trained.train() # Need train mode for gradients
            model_trained.zero_grad()
            logits_t = model_trained(img_b)
            probs_t = torch.sigmoid(logits_t)
            
            res['trained_stats'] = get_stats(probs_t, is_mask=False)
            res['trained_fg'] = {
                '0.30': float((probs_t > 0.3).float().mean() * 100),
                '0.40': float((probs_t > 0.4).float().mean() * 100),
                '0.50': float((probs_t > 0.5).float().mean() * 100),
                '0.55': float((probs_t > 0.55).float().mean() * 100),
                '0.60': float((probs_t > 0.6).float().mean() * 100),
                '0.70': float((probs_t > 0.7).float().mean() * 100)
            }
            
            # 5. Loss decomposition
            loss_bce = bce_fn(logits_t, mask_b)
            loss_dice = dice_fn(logits_t, mask_b)
            loss_total = loss_bce + loss_dice
            
            res['loss'] = {
                'bce': float(loss_bce.item()),
                'dice': float(loss_dice.item()),
                'total': float(loss_total.item())
            }
            
            # 6. Gradient diagnostic
            loss_total.backward()
            
            total_norm = 0.0
            non_zero_grads = 0
            nan_inf_grads = 0
            
            for p in model_trained.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    non_zero_grads += (p.grad.data != 0).sum().item()
                    if not torch.isfinite(p.grad.data).all():
                        nan_inf_grads += 1
                        
            total_norm = total_norm ** 0.5
            
            res['gradients'] = {
                'total_norm': float(total_norm),
                'non_zero_params_count': non_zero_grads,
                'nan_inf_tensors': nan_inf_grads
            }
            
            # 7. Compare
            pred_fg_pct = res['trained_fg']['0.50']
            target_fg_pct = res['gt_stats']['fg_percentage']
            intersection = float(((probs_t > 0.5) & (mask_b > 0)).float().sum())
            union = float(((probs_t > 0.5) | (mask_b > 0)).float().sum())
            res['compare'] = {
                'target_fg_pct': target_fg_pct,
                'pred_fg_pct': pred_fg_pct,
                'overlap_iou': intersection / (union + 1e-7)
            }
            
            results.append(res)
            
            # 8. Save visualization (only 5 training patches)
            if split_name == 'train' and samples_collected < 5:
                img_vis = img_b[0, 0].detach().cpu().numpy()
                img_vis = (img_vis * 0.5 + 0.5).clip(0, 1)
                mask_vis = mask_b[0, 0].detach().cpu().numpy()
                prob_vis = probs_t[0, 0].detach().cpu().numpy()
                pred_vis = (prob_vis > 0.5).astype(np.float32)
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img_vis, cmap='gray'); axes[0].set_title('Input')
                axes[1].imshow(mask_vis, cmap='gray'); axes[1].set_title('GT Mask')
                axes[2].imshow(prob_vis, cmap='magma', vmin=0, vmax=1); axes[2].set_title('Prob Map')
                axes[3].imshow(pred_vis, cmap='gray'); axes[3].set_title('Prediction (>0.5)')
                
                for ax in axes: ax.axis('off')
                plt.suptitle(f"Train Patch {samples_collected} (Strategy: {meta['strategy']})")
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'train_patch_{samples_collected}.png'))
                plt.close()
                
            samples_collected += 1

    # 9. Save JSON and CSV
    with open(os.path.join(outputs_dir, 'training_signal_diagnostic.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Flatten for CSV
    flat_results = []
    for r in results:
        flat = {
            'split': r['split'],
            'sample_idx': r['sample_idx'],
            'strategy': r['strategy'],
            'has_filament': r['has_filament'],
            'img_mean': r['image_stats']['mean'],
            'gt_fg_pct': r['gt_stats']['fg_percentage'],
            'untrained_prob_mean': r['untrained_stats']['mean'],
            'trained_prob_mean': r['trained_stats']['mean'],
            'trained_fg_050': r['trained_fg']['0.50'],
            'loss_total': r['loss']['total'],
            'grad_norm': r['gradients']['total_norm']
        }
        flat_results.append(flat)
    
    df = pd.DataFrame(flat_results)
    df.to_csv(os.path.join(outputs_dir, 'training_signal_diagnostic.csv'), index=False)
    
    # Generate summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    
    axes[0].plot(df_train['sample_idx'], df_train['loss_total'], 'bo-', label='Train Loss')
    axes[0].plot(df_val['sample_idx'], df_val['loss_total'], 'ro-', label='Val Loss')
    axes[0].set_title('Total Loss per Sample')
    axes[0].legend()
    
    axes[1].plot(df_train['sample_idx'], df_train['trained_prob_mean'], 'bo-', label='Train Prob Mean')
    axes[1].plot(df_val['sample_idx'], df_val['trained_prob_mean'], 'ro-', label='Val Prob Mean')
    axes[1].set_title('Mean Probability per Sample')
    axes[1].legend()
    
    axes[2].plot(df_train['sample_idx'], df_train['grad_norm'], 'go-', label='Train Grad Norm')
    axes[2].set_title('Gradient Norm (Train only)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_signal_diagnostic.png'))
    plt.close()
    
    print("\n--- Diagnostic Complete ---")
    print("Investigating specific failure modes:")
    
    def analyze(results):
        issues = []
        
        # 1. Mask normalization (are masks 0-1 or 0-255?)
        max_gt = max([r['gt_stats']['max'] for r in results])
        if max_gt > 1.0:
            issues.append(f"CRITICAL: GT masks are not normalized (max value = {max_gt}). BCE expects 0-1.")
            
        # 2. Sigmoid applied twice
        # UnetPlusPlus outputs logits. DiceLoss applies sigmoid. BCEWithLogitsLoss expects logits.
        # This is correct.
        
        # 3. Image normalization
        img_mins = [r['image_stats']['min'] for r in results]
        img_maxs = [r['image_stats']['max'] for r in results]
        if min(img_mins) < -2 or max(img_maxs) > 2:
            issues.append(f"WARNING: Images might not be properly normalized. Ranges: [{min(img_mins):.2f}, {max(img_maxs):.2f}]")
            
        # 4. Logits treated as probabilities
        # Verified they are treated as logits in BCEWithLogitsLoss.
        
        # 5. Gradients
        grad_norms = [r['gradients']['total_norm'] for r in results if r['split'] == 'train']
        if all(g < 1e-5 for g in grad_norms):
            issues.append("CRITICAL: Vanishing gradients. Model is barely learning.")
        elif any(g > 100 for g in grad_norms):
            issues.append("WARNING: Exploding gradients detected.")
            
        # 6. Class imbalance
        fg_pcts = [r['gt_stats']['fg_percentage'] for r in results]
        mean_fg = np.mean(fg_pcts)
        issues.append(f"INFO: Average foreground percentage in sampled patches: {mean_fg:.4f}%")
        
        # 7. Model Collapse
        prob_means = [r['trained_stats']['mean'] for r in results]
        if np.std(prob_means) < 0.05 and 0.4 < np.mean(prob_means) < 0.6:
            issues.append("CRITICAL: Model has collapsed to predicting ~0.5 everywhere.")
            
        return issues
        
    issues = analyze(results)
    for issue in issues:
        print(issue)
        
    print("\nOutputs saved to outputs/training_signal_diagnostic.*")

if __name__ == '__main__':
    main()
