import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unet_baseline import get_baseline_model
from src.data.augmentations import get_inference_augmentation
from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_semantic_mask

def sliding_window_inference(model, image, patch_size, device, transforms):
    h, w = image.shape
    stride = patch_size // 2
    
    model.eval()
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
    pad_h_new, pad_w_new = image.shape
    
    prob_map = np.zeros((pad_h_new, pad_w_new), dtype=np.float32)
    count_map = np.zeros((pad_h_new, pad_w_new), dtype=np.float32)
    
    for y in range(0, pad_h_new - patch_size + 1, stride):
        for x in range(0, pad_w_new - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_hwc = patch[..., np.newaxis]
            
            aug = transforms(image=patch_hwc)
            tensor = aug['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        pred = model(tensor)
                else:
                    pred = model(tensor)
                    
                prob = torch.sigmoid(pred).squeeze().cpu().numpy()
                
            prob_map[y:y+patch_size, x:x+patch_size] += prob
            count_map[y:y+patch_size, x:x+patch_size] += 1
            
    prob_map = prob_map[:h, :w]
    count_map = count_map[:h, :w]
    
    prob_map /= count_map
    return prob_map

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'baseline.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    model = get_baseline_model(config)
    ckpt_path = os.path.join(base_dir, 'checkpoints', 'best_baseline.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    
    transforms = get_inference_augmentation()
    
    val_csv = os.path.join(base_dir, config['dataset']['val_csv'])
    val_df = pd.read_csv(val_csv)
    
    img_dir = os.path.join(base_dir, config['dataset']['image_dir'])
    json_path = os.path.join(base_dir, config['dataset']['json_path'])
    parser = MAGFiLOAnnotationParser(json_path)
    
    out_dir = os.path.join(base_dir, 'outputs', 'baseline_predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    tp, fp, fn = 0, 0, 0
    
    print("Running sliding-window inference on validation set...")
    visualized = 0
    
    # We will just evaluate 20 validation images to save time during this smoke test, as instructed:
    # "use a small number of epochs only for the smoke test if GPU/time is limited"
    eval_df = val_df.head(20)
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        img_id = str(row['image_id'])
        filename = str(row['file_name'])
        
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        prob_map = sliding_window_inference(model, img, config['dataset']['patch_size'], device, transforms)
        pred_mask = (prob_map > 0.5).astype(np.uint8)
        
        anns = parser.get_annotations_for_image(img_id)
        gt_mask = create_semantic_mask(anns, img.shape[0], img.shape[1]) if anns else np.zeros_like(img)
        
        tp += np.sum((pred_mask == 1) & (gt_mask == 1))
        fp += np.sum((pred_mask == 1) & (gt_mask == 0))
        fn += np.sum((pred_mask == 0) & (gt_mask == 1))
        
        if visualized < 10:
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay[pred_mask == 1] = [0, 0, 255]
            
            out_path = os.path.join(out_dir, f"pred_{filename}")
            cv2.imwrite(out_path, overlay)
            visualized += 1
            
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    pq_note = "PQ (Panoptic Quality) is not correctly implemented in the baseline since this is a pure semantic model without instance discrimination. Requires further postprocessing or Mask2Former."
    
    metrics = {
        "validation_dice": float(dice),
        "validation_iou": float(iou),
        "pq_status": pq_note
    }
    
    with open(os.path.join(base_dir, 'outputs', 'baseline_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation complete. Dice: {dice:.4f}, IoU: {iou:.4f}")
    print(pq_note)

if __name__ == '__main__':
    main()
