import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset_fullimg import MAGFiLOFullImageDataset
from src.data.augmentations import get_inference_augmentation
from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_instance_masks
from src.models.unet_baseline import get_baseline_model
from src.inference.sliding_window import predict_full_image
from src.postprocess.instance import semantic_to_instances
from src.metrics.pq import calculate_pq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10, help='Limit number of validation images for smoke testing')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'baseline_v2.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    image_dir = config['dataset']['image_dir']
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(base_dir, image_dir)
    json_path = config['dataset']['json_path']
    if not os.path.isabs(json_path):
        json_path = os.path.join(base_dir, json_path)
    val_csv_path = os.path.join(base_dir, 'outputs', 'kaggle_val_split.csv')
    
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    val_dataset = MAGFiLOFullImageDataset(
        csv_path=val_csv_path,
        img_dir=image_dir,
        json_path=json_path,
        transform=get_inference_augmentation()
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_baseline_model(config).to(device)
    
    checkpoint_path = os.path.join(base_dir, 'checkpoints', 'best_baseline_v2.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Warning: local checkpoint {checkpoint_path} not found. Trying Kaggle path.")
        checkpoint_path = '/kaggle/input/datasets/nytsoul/solar-filament-segmentation-6/solar-filament-segmentation/checkpoints/best_baseline_v2.pth'
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    parser_ann = MAGFiLOAnnotationParser(json_path)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95]
    
    threshold_results = {t: [] for t in thresholds}
    image_stats = []

    limit = args.limit if args.limit else len(val_dataset)
    
    for i in tqdm(range(limit), desc="Processing images"):
        img_tensor, mask_tensor, filename = val_dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        gt_semantic = mask_tensor.squeeze().numpy().astype(np.uint8)
        
        img_id = parser_ann.filename_to_img_id.get(filename)
        anns = parser_ann.get_annotations_for_image(img_id)
        gt_instances = create_instance_masks(anns, height=gt_semantic.shape[0], width=gt_semantic.shape[1])
        
        with torch.no_grad():
            prob_map = predict_full_image(
                model, 
                img_tensor, 
                patch_size=config['dataset'].get('patch_size', 768),
                overlap=config.get('validation', {}).get('overlap', 0.25),
                device=device
            )
            
        prob_min = float(np.min(prob_map))
        prob_max = float(np.max(prob_map))
        prob_mean = float(np.mean(prob_map))
        prob_median = float(np.median(prob_map))
        p90 = float(np.percentile(prob_map, 90))
        p95 = float(np.percentile(prob_map, 95))
        p99 = float(np.percentile(prob_map, 99))
        
        image_stats.append({
            'filename': filename,
            'prob_min': prob_min,
            'prob_max': prob_max,
            'prob_mean': prob_mean,
            'prob_median': prob_median,
            'p90': p90,
            'p95': p95,
            'p99': p99
        })

        for thresh in thresholds:
            pred_semantic = (prob_map >= thresh).astype(np.uint8)
            fg_pct = float(np.sum(pred_semantic)) / pred_semantic.size * 100.0
            
            intersection = np.logical_and(pred_semantic, gt_semantic).sum()
            union = np.logical_or(pred_semantic, gt_semantic).sum()
            pixel_iou = intersection / union if union > 0 else 0.0
            pixel_dice = (2 * intersection) / (pred_semantic.sum() + gt_semantic.sum() + 1e-7)
            
            # Using min_area=100 as in evaluate_pq.py
            pred_instances = semantic_to_instances(prob_map, prob_threshold=thresh, min_area=100)
            
            pq_res = calculate_pq(pred_instances, gt_instances)
            
            threshold_results[thresh].append({
                'filename': filename,
                'dice': pixel_dice,
                'iou': pixel_iou,
                'fg_pct': fg_pct,
                'pred_cc': pq_res['pred_count'],
                'gt_cc': pq_res['gt_count'],
                'tp': pq_res['tp'],
                'fp': pq_res['fp'],
                'fn': pq_res['fn']
            })

    # Aggregate results
    agg_results = []
    for thresh in thresholds:
        df_thresh = pd.DataFrame(threshold_results[thresh])
        agg_results.append({
            'threshold': thresh,
            'mean_dice': df_thresh['dice'].mean(),
            'mean_iou': df_thresh['iou'].mean(),
            'mean_fg_pct': df_thresh['fg_pct'].mean(),
            'mean_pred_cc': df_thresh['pred_cc'].mean(),
            'mean_gt_cc': df_thresh['gt_cc'].mean(),
            'mean_tp': df_thresh['tp'].mean(),
            'mean_fp': df_thresh['fp'].mean(),
            'mean_fn': df_thresh['fn'].mean()
        })
        
    df_agg = pd.DataFrame(agg_results)
    df_agg.to_csv(os.path.join(outputs_dir, 'threshold_diagnostic.csv'), index=False)
    
    with open(os.path.join(outputs_dir, 'threshold_diagnostic.json'), 'w') as f:
        json.dump({
            'threshold_metrics': agg_results,
            'image_stats': image_stats
        }, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_agg['threshold'], df_agg['mean_dice'], marker='o', label='Mean Dice')
    plt.plot(df_agg['threshold'], df_agg['mean_iou'], marker='s', label='Mean IoU')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Pixel Dice and IoU vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outputs_dir, 'threshold_diagnostic.png'))
    plt.close()
    
    print("\n--- Threshold Diagnostic Summary ---")
    print(df_agg.to_string(index=False))
    
    print("\n--- Average Image Stats ---")
    df_stats = pd.DataFrame(image_stats)
    print(df_stats.drop('filename', axis=1).mean().to_string())

if __name__ == '__main__':
    main()
