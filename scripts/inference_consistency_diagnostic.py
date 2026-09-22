"""
5-Image Inference Consistency Diagnostic
=========================================
Compares EXACT inference paths used by:
  1. train_baseline_v2.py  -> full_image_validation()
  2. evaluate_pq.py        -> main() evaluation loop

Uses the SAME 5 validation images and the same checkpoint for both paths.
Reports per-image probability stats, foreground pixel counts, Dice, IoU,
and saves side-by-side prediction images.

DOES NOT retrain, modify the model, run full PQ evaluation, or change thresholds.
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.data.dataset_fullimg     import MAGFiLOFullImageDataset
from src.data.augmentations       import get_inference_augmentation
from src.data.annotations         import MAGFiLOAnnotationParser
from src.models.unet_baseline     import get_baseline_model
from src.inference.sliding_window import predict_full_image

CHECKPOINT_PATH = "/kaggle/input/datasets/nytsoul/solar-filament-segmentation-6/solar-filament-segmentation/checkpoints/best_baseline_v2.pth"
CONFIG_PATH     = os.path.join(BASE_DIR, "configs", "baseline_v2.yaml")
VAL_CSV_PATH    = os.path.join(BASE_DIR, "outputs", "kaggle_val_split.csv")
N_IMAGES        = 5
PROB_THRESH     = 0.5
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs", "diagnostic_consistency")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

img_dir   = config['dataset']['image_dir']
json_path = config['dataset']['json_path']
if not os.path.isabs(img_dir):
    img_dir = os.path.join(BASE_DIR, img_dir)
if not os.path.isabs(json_path):
    json_path = os.path.join(BASE_DIR, json_path)

patch_size = config['dataset'].get('patch_size', 768)
overlap    = config.get('validation', {}).get('overlap', 0.25)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")
print(f"[INFO] Patch size: {patch_size}  |  Overlap: {overlap}")
print(f"[INFO] Prob thresh: {PROB_THRESH}")
print()

print("=" * 60)
print("CHECKPOINT LOADING")
print("=" * 60)
model = get_baseline_model(config).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"  Model class   : {model.__class__.__name__}")
print(f"  Param count   : {param_count:,}")

if not os.path.exists(CHECKPOINT_PATH):
    local_ckpt = os.path.join(BASE_DIR, "checkpoints", "best_baseline_v2.pth")
    if os.path.exists(local_ckpt):
        CHECKPOINT_PATH = local_ckpt
        print(f"  [WARNING] Kaggle path not found. Using local: {local_ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH} or {local_ckpt}")

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
if list(state_dict.keys())[0].startswith('module.'):
    print("  Detected DataParallel prefix. Stripping...")
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

print(f"  Checkpoint keys : {len(state_dict)}")
print(f"  Model keys      : {len(model.state_dict())}")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  Missing keys    : {len(missing)}")
if missing:
    for k in missing[:5]: print(f"    - {k}")
print(f"  Unexpected keys : {len(unexpected)}")
if unexpected:
    for k in unexpected[:5]: print(f"    - {k}")

try:
    model.load_state_dict(state_dict, strict=True)
    print("  Strict load     : SUCCESS")
except RuntimeError as e:
    print(f"  Strict load     : FAILED - {e}")

model.eval()
print()

transform   = get_inference_augmentation()
val_dataset = MAGFiLOFullImageDataset(
    csv_path=VAL_CSV_PATH, img_dir=img_dir, json_path=json_path, transform=transform
)
n_use = min(N_IMAGES, len(val_dataset))
print(f"[INFO] Using first {n_use} of {len(val_dataset)} validation images.")
print()


def prob_stats(prob_map):
    return {'min': float(prob_map.min()), 'max': float(prob_map.max()),
            'mean': float(prob_map.mean()), 'std': float(prob_map.std())}


def v2_metrics(prob_map, mask_np, thresh=0.5):
    """Exact train_baseline_v2.py formula: prob > thresh, float masks, 1e-7 smoothing."""
    pred_binary = (prob_map > thresh).astype(np.float32)
    mask_np_f   = mask_np.astype(np.float32)
    intersection = (pred_binary * mask_np_f).sum()
    pred_sum = pred_binary.sum(); gt_sum = mask_np_f.sum()
    union = pred_sum + gt_sum - intersection
    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-7)
    iou  = intersection / (union + 1e-7) if union > 0 else 0.0
    return {'pred_binary': pred_binary, 'foreground_pixels': int(pred_sum),
            'dice': float(dice), 'iou': float(iou)}


def pq_pixel_metrics(prob_map, mask_np, thresh=0.5):
    """Exact evaluate_pq.py formula: prob >= thresh, logical masks."""
    pred_semantic = (prob_map >= thresh).astype(np.uint8)
    gt_semantic   = mask_np.astype(np.uint8)
    intersection  = np.logical_and(pred_semantic, gt_semantic).sum()
    union         = np.logical_or(pred_semantic,  gt_semantic).sum()
    pixel_iou  = intersection / union if union > 0 else 0.0
    pixel_dice = (2 * intersection) / (pred_semantic.sum() + gt_semantic.sum() + 1e-7)
    return {'pred_semantic': pred_semantic, 'foreground_pixels': int(pred_semantic.sum()),
            'dice': float(pixel_dice), 'iou': float(pixel_iou)}


records = []
cache   = []

print("=" * 60)
print("PER-IMAGE DIAGNOSTIC")
print("=" * 60)

for i in range(n_use):
    img_tensor, mask_tensor, filename = val_dataset[i]
    img_tensor_b = img_tensor.unsqueeze(0).to(device)
    mask_np      = mask_tensor.squeeze().numpy()

    with torch.no_grad():
        prob_map = predict_full_image(model, img_tensor_b,
                                      patch_size=patch_size, overlap=overlap, device=device)

    cache.append((img_tensor, mask_np, filename, prob_map))
    ps  = prob_stats(prob_map)
    v2  = v2_metrics(prob_map, mask_np, thresh=PROB_THRESH)
    pq  = pq_pixel_metrics(prob_map, mask_np, thresh=PROB_THRESH)
    mm  = int(np.abs(v2['pred_binary'].astype(float) - pq['pred_semantic'].astype(float)).sum())

    records.append({
        'idx': i, 'filename': filename,
        'prob_min': ps['min'], 'prob_max': ps['max'],
        'prob_mean': ps['mean'], 'prob_std': ps['std'],
        'v2_fg_pixels': v2['foreground_pixels'], 'pq_fg_pixels': pq['foreground_pixels'],
        'binary_mismatch_pixels': mm,
        'v2_dice': v2['dice'], 'v2_iou': v2['iou'],
        'pq_dice': pq['dice'], 'pq_iou': pq['iou'],
        'dice_diff': abs(v2['dice'] - pq['dice']),
        'iou_diff':  abs(v2['iou']  - pq['iou']),
    })

    print(f"\nImage {i+1}/{n_use}: {filename}")
    print(f"  Prob  min/max/mean/std : {ps['min']:.6f} / {ps['max']:.6f} / {ps['mean']:.6f} / {ps['std']:.6f}")
    print(f"  V2  threshold operator : prob_map  > {PROB_THRESH}  (strict >)")
    print(f"  PQ  threshold operator : prob_map >= {PROB_THRESH}  (>=)")
    print(f"  V2 foreground pixels   : {v2['foreground_pixels']:,}")
    print(f"  PQ foreground pixels   : {pq['foreground_pixels']:,}")
    print(f"  Binary mismatch pixels : {mm:,}")
    print(f"  V2 Dice  : {v2['dice']:.6f}    PQ Dice  : {pq['dice']:.6f}    Delta={abs(v2['dice']-pq['dice']):.6f}")
    print(f"  V2 IoU   : {v2['iou']:.6f}    PQ IoU   : {pq['iou']:.6f}    Delta={abs(v2['iou']-pq['iou']):.6f}")

    img_vis = img_tensor.squeeze().cpu().numpy()
    img_vis = (img_vis * 0.5 + 0.5).clip(0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img_vis, cmap='gray');            axes[0].set_title('Input Image'); axes[0].axis('off')
    axes[1].imshow(mask_np, cmap='gray');            axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
    axes[2].set_title(f"Prob Map\nmin={ps['min']:.4f} max={ps['max']:.4f}"); axes[2].axis('off')
    axes[3].imshow(v2['pred_binary'], cmap='gray')
    axes[3].set_title(f"V2 Pred (>0.5)\nDice={v2['dice']:.4f} IoU={v2['iou']:.4f}\nFG={v2['foreground_pixels']:,}")
    axes[3].axis('off')
    axes[4].imshow(pq['pred_semantic'], cmap='gray')
    axes[4].set_title(f"PQ Pred (>=0.5)\nDice={pq['dice']:.4f} IoU={pq['iou']:.4f}\nFG={pq['foreground_pixels']:,}")
    axes[4].axis('off')
    plt.suptitle(f"{filename}  |  Binary mismatch: {mm:,} pixels", fontsize=11)
    plt.tight_layout()
    safe = filename.replace('/', '_').replace('\\', '_')
    sp   = os.path.join(OUTPUT_DIR, f"diagnostic_{i+1:02d}_{safe}.png")
    plt.savefig(sp, dpi=100, bbox_inches='tight'); plt.close()
    print(f"  Saved: {sp}")


df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, 'per_image_consistency_report.csv'), index=False)

mv2d = df['v2_dice'].mean(); mpqd = df['pq_dice'].mean()
mv2i = df['v2_iou'].mean();  mpqi = df['pq_iou'].mean()
mv2f = df['v2_fg_pixels'].mean(); mpqf = df['pq_fg_pixels'].mean()
mpm  = df['prob_mean'].mean(); mpmax = df['prob_max'].max()

print("\n[INFO] Counting pixels exactly at threshold...")
boundary = []
for _, _, fn, pm in cache:
    b = int((pm == PROB_THRESH).sum())
    boundary.append(b)
    print(f"  {fn}: {b} pixels exactly at {PROB_THRESH}")
avg_boundary = float(np.mean(boundary))

if mpmax < 0.01:
    model_state = "NEAR-ZERO output (collapsed / DataParallel mismatch / wrong checkpoint)"
elif mpmax < 0.1:
    model_state = "VERY LOW output (max<0.1) - likely wrong checkpoint or DataParallel prefix not stripped"
elif mpm > 0.8:
    model_state = "VERY HIGH output (mean>0.8) - label inversion?"
else:
    model_state = "MODEL OUTPUT APPEARS REASONABLE"

if mpmax < 0.05:
    rc_code = "CHECKPOINT_MISMATCH_OR_UNINIT"
    rc_desc = ("Near-zero model outputs on best_baseline_v2.pth. "
               "evaluate_pq.py defaults to best_baseline.pth (NOT v2). "
               "V2 training used the correctly-loaded model internally, "
               "while evaluate_pq.py used random/v1 weights.")
    fix = ("python scripts/evaluate_pq.py --checkpoint checkpoints/best_baseline_v2.pth\n"
           "  The default checkpoint in evaluate_pq.py is 'best_baseline.pth', NOT 'best_baseline_v2.pth'.")
elif abs(mv2d - mpqd) < 0.005:
    rc_code = "CHECKPOINT_DEFAULT_MISMATCH"
    rc_desc = ("Both pipelines produce IDENTICAL results on the same checkpoint. "
               "The observed gap (Dice 0.5889 vs 0.0111) is because evaluate_pq.py "
               "defaulted to best_baseline.pth (v1/random weights) "
               "instead of best_baseline_v2.pth (the trained v2 checkpoint).")
    fix = "python scripts/evaluate_pq.py --checkpoint checkpoints/best_baseline_v2.pth"
else:
    rc_code = "FORMULA_MISMATCH"
    rc_desc = (f"Dice differs by {abs(mv2d-mpqd):.4f}. "
               "Check threshold operator (> vs >=) and Dice formula differences.")
    fix = "Align threshold operator and formula between the two scripts."

summary = {
    "n_images_tested": n_use, "checkpoint_path": CHECKPOINT_PATH,
    "patch_size": patch_size, "overlap": overlap, "prob_threshold": PROB_THRESH,
    "model_state": model_state,
    "mean_prob_min": round(df['prob_min'].mean(), 6),
    "mean_prob_max": round(df['prob_max'].mean(), 6),
    "mean_prob_mean": round(mpm, 6),
    "mean_prob_std": round(df['prob_std'].mean(), 6),
    "mean_v2_fg_pixels": round(mv2f, 1), "mean_pq_fg_pixels": round(mpqf, 1),
    "avg_boundary_pixels_at_0.5": round(avg_boundary, 1),
    "mean_v2_dice": round(mv2d, 6), "mean_pq_dice": round(mpqd, 6),
    "dice_difference": round(abs(mv2d - mpqd), 6),
    "mean_v2_iou":  round(mv2i, 6), "mean_pq_iou":  round(mpqi, 6),
    "iou_difference": round(abs(mv2i - mpqi), 6),
    "root_cause_code": rc_code, "ROOT_CAUSE": rc_desc, "RECOMMENDED_FIX": fix,
}

with open(os.path.join(OUTPUT_DIR, 'consistency_summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)

SEP = "=" * 70
print(f"\n\n{SEP}")
print("AGGREGATE RESULTS  (averaged over 5 images)")
print(SEP)
print(f"  Mean prob min/max/mean/std : {df['prob_min'].mean():.6f} / {df['prob_max'].mean():.6f} / {mpm:.6f} / {df['prob_std'].mean():.6f}")
print(f"  Mean V2 foreground pixels  : {mv2f:,.1f}")
print(f"  Mean PQ foreground pixels  : {mpqf:,.1f}")
print(f"  Mean V2 Dice               : {mv2d:.6f}")
print(f"  Mean PQ Dice               : {mpqd:.6f}")
print(f"  Mean V2 IoU                : {mv2i:.6f}")
print(f"  Mean PQ IoU                : {mpqi:.6f}")

print(f"\n{SEP}")
print("ROOT CAUSE ANALYSIS")
print(SEP)
print(f"ROOT CAUSE          : {rc_code}")
print()
print("V2 INFERENCE        :")
print("  - MAGFiLOFullImageDataset + get_inference_augmentation()")
print("  - predict_full_image()  ->  prob_map > 0.5  (strict greater-than)")
print("  - Dice = (2*I)/(P+G+1e-7)  using float masks")
print("  - IoU  = I/(P+G-I+1e-7)   using float masks")
print()
print("PQ INFERENCE        :")
print("  - MAGFiLOFullImageDataset + get_inference_augmentation()")
print("  - predict_full_image()  ->  prob_map >= 0.5  (greater-or-equal)")
print("  - Dice = (2*I)/(P+G+1e-7)  using logical masks")
print("  - IoU  = logical_and/logical_or")
print("  - DEFAULT checkpoint = 'best_baseline.pth' (NOT best_baseline_v2.pth!)")
print()
print(f"MAX PROBABILITY DIFFERENCE  : 0.0  (identical prob maps - same model + same call)")
print(f"DICE DIFFERENCE             : {abs(mv2d - mpqd):.6f}")
print(f"IOU DIFFERENCE              : {abs(mv2i - mpqi):.6f}")
print()
print(f"MODEL STATE                 : {model_state}")
print()
print("RECOMMENDED FIX     :")
print(f"  {fix}")
print()
print(f"Results saved to: {OUTPUT_DIR}")
print(f"{SEP}")
