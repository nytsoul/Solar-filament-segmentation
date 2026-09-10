import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_instance_masks, create_semantic_mask
from src.data.augmentations import get_inference_augmentation
from src.models.unet_baseline import get_baseline_model
from src.inference.sliding_window import predict_full_image
from src.postprocess.instance import semantic_to_instances
from src.metrics.pq import calculate_pq

import argparse

def plot_failure_analysis(image, gt_mask, prob_map, pred_mask, out_path, title):
    # Setup visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GT Instances
    axes[1].imshow(gt_mask, cmap='tab20', interpolation='nearest')
    axes[1].set_title('Ground Truth Instances')
    axes[1].axis('off')
    
    # Prob Map
    axes[2].imshow(prob_map, cmap='magma')
    axes[2].set_title('Predicted Probability')
    axes[2].axis('off')
    
    # Pred Instances
    axes[3].imshow(pred_mask, cmap='tab20', interpolation='nearest')
    axes[3].set_title('Predicted Instances')
    axes[3].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--prob_thresh', type=float, default=0.5)
    parser.add_argument('--min_area', type=int, default=100)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Kaggle Paths
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    in_kaggle = os.path.exists(kaggle_dir)
    if in_kaggle:
        image_dir = os.path.join(kaggle_dir, 'train', 'train_images')
        json_path = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        out_dir_base = '/kaggle/working'
    else:
        image_dir = config['dataset']['image_dir'] if os.path.isabs(config['dataset']['image_dir']) else os.path.join(base_dir, config['dataset']['image_dir'])
        json_path = config['dataset']['json_path'] if os.path.isabs(config['dataset']['json_path']) else os.path.join(base_dir, config['dataset']['json_path'])
        out_dir_base = base_dir

    outputs_dir = os.path.join(out_dir_base, 'outputs', 'pq_analysis')
    os.makedirs(outputs_dir, exist_ok=True)
    
    val_csv_path = os.path.join(out_dir_base, 'outputs', 'kaggle_val_split.csv')
    if not os.path.exists(val_csv_path):
        val_csv_path = os.path.join(base_dir, 'outputs', 'kaggle_val_split.csv')
        
    val_df = pd.read_csv(val_csv_path)
    if args.limit:
        val_df = val_df.head(args.limit)

    parser_ann = MAGFiLOAnnotationParser(json_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_baseline_model(config).to(device)
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(out_dir_base, 'checkpoints', 'best_baseline.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(base_dir, 'checkpoints', 'best_baseline.pth')
            
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint {checkpoint_path} due to {e}")
            print("Continuing with random weights for testing.")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights.")
        
    model.eval()
    
    transform = get_inference_augmentation()
    
    results = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_fragmentation = 0
    total_merging = 0
    total_pred_instances = 0
    total_gt_instances = 0
    
    iou_sums = 0.0
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating PQ"):
        filename = row['file_name']
        img_id = parser_ann.filename_to_img_id.get(filename)
        
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Ground truth
        anns = parser_ann.get_annotations_for_image(img_id)
        gt_instances = create_instance_masks(anns, height=img.shape[0], width=img.shape[1])
        gt_semantic = create_semantic_mask(anns, height=img.shape[0], width=img.shape[1])
        
        # Inference
        augmented = transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        prob_map = predict_full_image(
            model, 
            img_tensor, 
            patch_size=config['dataset'].get('patch_size', 768),
            overlap=0.25,
            device=device
        )
        
        # Instance extraction
        pred_instances = semantic_to_instances(prob_map, prob_threshold=args.prob_thresh, min_area=args.min_area)
        pred_semantic = (prob_map >= args.prob_thresh).astype(np.uint8)
        
        # Metrics
        pq_res = calculate_pq(pred_instances, gt_instances)
        
        # Pixel-wise metrics for comparison
        intersection = np.logical_and(pred_semantic, gt_semantic).sum()
        union = np.logical_or(pred_semantic, gt_semantic).sum()
        pixel_iou = intersection / union if union > 0 else 0.0
        pixel_dice = (2 * intersection) / (pred_semantic.sum() + gt_semantic.sum() + 1e-7)
        
        res_dict = {
            'filename': filename,
            'pq': pq_res['pq'],
            'sq': pq_res['sq'],
            'rq': pq_res['rq'],
            'tp': pq_res['tp'],
            'fp': pq_res['fp'],
            'fn': pq_res['fn'],
            'pred_instances': pq_res['pred_count'],
            'gt_instances': pq_res['gt_count'],
            'fragmentation': pq_res['fragmentation_count'],
            'merging': pq_res['merging_count'],
            'pixel_dice': pixel_dice,
            'pixel_iou': pixel_iou
        }
        results.append(res_dict)
        
        # Accumulate totals
        total_tp += pq_res['tp']
        total_fp += pq_res['fp']
        total_fn += pq_res['fn']
        total_fragmentation += pq_res['fragmentation_count']
        total_merging += pq_res['merging_count']
        total_pred_instances += pq_res['pred_count']
        total_gt_instances += pq_res['gt_count']
        
        sq_sum = pq_res['sq'] * pq_res['tp']
        iou_sums += sq_sum
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(outputs_dir, 'per_image_pq.csv'), index=False)
    
    mean_pq = df_res['pq'].mean()
    mean_sq = df_res['sq'].mean()
    mean_rq = df_res['rq'].mean()
    mean_dice = df_res['pixel_dice'].mean()
    mean_iou = df_res['pixel_iou'].mean()
    
    global_rq = total_tp / (total_tp + 0.5*total_fp + 0.5*total_fn) if (total_tp + 0.5*total_fp + 0.5*total_fn) > 0 else 0
    global_sq = iou_sums / total_tp if total_tp > 0 else 0
    global_pq = global_sq * global_rq
    
    summary = {
        'Validation images': len(val_df),
        'Dice': float(mean_dice),
        'IoU': float(mean_iou),
        'PQ': float(mean_pq),
        'SQ': float(mean_sq),
        'RQ': float(mean_rq),
        'Global_PQ': float(global_pq),
        'TP': int(total_tp),
        'FP': int(total_fp),
        'FN': int(total_fn),
        'Fragmentation rate': float(total_fragmentation / max(1, total_gt_instances)),
        'Merging rate': float(total_merging / max(1, total_pred_instances)),
        'Average predicted instances/image': float(total_pred_instances / len(val_df)),
        'Average GT instances/image': float(total_gt_instances / len(val_df))
    }
    
    with open(os.path.join(outputs_dir, 'pq_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    plot_targets = []
    plot_targets.extend(df_res.nlargest(2, 'pq')['filename'].tolist())
    plot_targets.extend(df_res.nsmallest(2, 'pq')['filename'].tolist())
    plot_targets.extend(df_res.nlargest(2, 'fragmentation')['filename'].tolist())
    plot_targets.extend(df_res.nlargest(2, 'merging')['filename'].tolist())
    plot_targets.extend(df_res.nlargest(2, 'fp')['filename'].tolist())
    
    plot_targets = list(set(plot_targets))
    
    print("Generating visualizations for representative images...")
    for filename in plot_targets:
        img_id = parser_ann.filename_to_img_id.get(filename)
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        anns = parser_ann.get_annotations_for_image(img_id)
        gt_instances = create_instance_masks(anns, height=img.shape[0], width=img.shape[1])
        
        augmented = transform(image=img)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        prob_map = predict_full_image(model, img_tensor, patch_size=config['dataset'].get('patch_size', 768), overlap=0.25, device=device)
        pred_instances = semantic_to_instances(prob_map, prob_threshold=args.prob_thresh, min_area=args.min_area)
        
        row_data = df_res[df_res['filename'] == filename].iloc[0]
        title = f"PQ: {row_data['pq']:.3f} | F-frag: {row_data['fragmentation']} | F-merg: {row_data['merging']}"
        out_path = os.path.join(outputs_dir, f"vis_{filename}.png")
        plot_failure_analysis(img, gt_instances, prob_map, pred_instances, out_path, title)
        
    print("\nBASELINE PQ EVALUATION")
    print("----------------------")
    print(f"Validation images: {summary['Validation images']}")
    print(f"Dice: {summary['Dice']:.4f}")
    print(f"IoU: {summary['IoU']:.4f}")
    print(f"PQ: {summary['PQ']:.4f}")
    print(f"SQ: {summary['SQ']:.4f}")
    print(f"RQ: {summary['RQ']:.4f}")
    print(f"TP: {summary['TP']}")
    print(f"FP: {summary['FP']}")
    print(f"FN: {summary['FN']}")
    print(f"Fragmentation rate: {summary['Fragmentation rate']:.4f}")
    print(f"Merging rate: {summary['Merging rate']:.4f}")
    print(f"Average predicted instances/image: {summary['Average predicted instances/image']:.2f}")
    print(f"Average GT instances/image: {summary['Average GT instances/image']:.2f}")

if __name__ == '__main__':
    main()
