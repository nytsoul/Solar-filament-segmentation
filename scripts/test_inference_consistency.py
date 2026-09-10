import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLODataset
from src.data.augmentations import get_validation_augmentation, get_inference_augmentation
from src.models.unet_baseline import get_baseline_model
from src.inference.sliding_window import predict_full_image

def calculate_dice(probs, mask, thresh=0.5):
    preds = (probs > thresh).astype(np.float32)
    mask = mask.astype(np.float32)
    intersection = (preds * mask).sum()
    union = preds.sum() + mask.sum()
    return (2 * intersection) / (union + 1e-7)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'baseline.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_baseline_model(config).to(device)
    
    checkpoint_path = os.path.join(base_dir, 'checkpoints', 'best_baseline.pth')
    state_dict = torch.load(checkpoint_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    val_csv_path = os.path.join(base_dir, 'outputs', 'kaggle_val_split.csv')
    val_df = pd.read_csv(val_csv_path).head(5)
    
    img_dir = config['dataset']['image_dir']
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(base_dir, img_dir)
        
    json_path = config['dataset']['json_path']
    if not os.path.isabs(json_path):
        json_path = os.path.join(base_dir, json_path)
        
    val_aug = get_validation_augmentation(config)
    inf_aug = get_inference_augmentation()
    
    val_dataset = MAGFiLODataset(
        csv_path=val_csv_path,
        img_dir=img_dir,
        json_path=json_path,
        transforms=val_aug
    )
    
    out_dir = os.path.join(base_dir, 'outputs', 'pq_analysis', 'inference_consistency')
    os.makedirs(out_dir, exist_ok=True)
    
    report = []
    
    for idx, row in val_df.iterrows():
        filename = row['file_name']
        print(f"Processing {filename}...")
        
        # Method A: Training Validation Pipeline
        img_tensor_A, mask_tensor_A = val_dataset[idx] # img is (1, 768, 768), mask is (1, 768, 768)
        img_tensor_A = img_tensor_A.unsqueeze(0).to(device) # (1, 1, 768, 768)
        mask_A = mask_tensor_A.squeeze().numpy() # (768, 768)
        
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits_A = model(img_tensor_A)
            else:
                logits_A = model(img_tensor_A)
            probs_A = torch.sigmoid(logits_A).squeeze().cpu().numpy()
            
        dice_A = calculate_dice(probs_A, mask_A)
        
        # Method B: Full Inference Pipeline
        img_path = os.path.join(img_dir, filename)
        img_full = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # (2048, 2048)
        
        augmented_B = inf_aug(image=img_full)
        img_tensor_B = augmented_B['image'].unsqueeze(0).to(device) # (1, 1, 2048, 2048)
        
        probs_B_full = predict_full_image(model, img_tensor_B, patch_size=768, overlap=0.25, device=device)
        
        # Extract center 768x768
        h, w = probs_B_full.shape
        y_min = (h - 768) // 2
        x_min = (w - 768) // 2
        probs_B = probs_B_full[y_min:y_min+768, x_min:x_min+768]
        
        dice_B = calculate_dice(probs_B, mask_A)
        
        diff = np.abs(probs_A - probs_B)
        
        # Method C: What if we pass (H, W, 1) to inf_aug?
        img_full_expanded = img_full[..., np.newaxis]
        augmented_C = inf_aug(image=img_full_expanded)
        img_tensor_C = augmented_C['image'].unsqueeze(0).to(device)
        probs_C_full = predict_full_image(model, img_tensor_C, patch_size=768, overlap=0.25, device=device)
        probs_C = probs_C_full[y_min:y_min+768, x_min:x_min+768]
        dice_C = calculate_dice(probs_C, mask_A)
        diff_C = np.abs(probs_A - probs_C)
        
        # Diagnostics
        stat = {
            "filename": filename,
            "dice_A_val_pipeline": float(dice_A),
            "dice_B_full_pipeline": float(dice_B),
            "dice_C_full_pipeline_HW1": float(dice_C),
            "probs_A_mean": float(probs_A.mean()),
            "probs_B_mean": float(probs_B.mean()),
            "probs_C_mean": float(probs_C.mean()),
            "max_abs_diff_A_vs_B": float(diff.max()),
            "mean_abs_diff_A_vs_B": float(diff.mean()),
            "max_abs_diff_A_vs_C": float(diff_C.max()),
            "mean_abs_diff_A_vs_C": float(diff_C.mean())
        }
        report.append(stat)
        
        # Visualization
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(mask_A, cmap='gray')
        axes[0].set_title('Ground Truth (Center)')
        axes[0].axis('off')
        
        axes[1].imshow(probs_A, vmin=0, vmax=1, cmap='magma')
        axes[1].set_title(f'Method A (Val Pipeline)\nDice: {dice_A:.4f}')
        axes[1].axis('off')
        
        axes[2].imshow(probs_B, vmin=0, vmax=1, cmap='magma')
        axes[2].set_title(f'Method B (Full Infer)\nDice: {dice_B:.4f}')
        axes[2].axis('off')
        
        axes[3].imshow(diff, vmin=0, vmax=1, cmap='hot')
        axes[3].set_title(f'Diff A vs B\nMax: {diff.max():.4f}')
        axes[3].axis('off')
        
        axes[4].imshow(probs_B_full, vmin=0, vmax=1, cmap='magma')
        axes[4].set_title('Method B (Full Image)')
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{filename}_comparison.png"))
        plt.close()
        
    with open(os.path.join(os.path.dirname(out_dir), 'inference_consistency_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print("Done. Report saved.")

if __name__ == '__main__':
    main()
