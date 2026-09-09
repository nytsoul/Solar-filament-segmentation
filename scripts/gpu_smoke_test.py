import torch
import os
import sys
import yaml
import time
import json
import numpy as np
import segmentation_models_pytorch as smp

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.data.dataset import MAGFiLODataset
from src.data.augmentations import get_training_augmentation
from src.models.unet_baseline import get_baseline_model
from torch.utils.data import DataLoader

def run_smoke_test():
    report = {}
    
    # 1. Dataset Loading
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    if not os.path.exists(kaggle_dir):
        kaggle_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
        
    config_path = os.path.join(base_dir, 'configs', 'baseline.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['dataset']['train_csv'] = 'outputs/kaggle_train_split.csv'
    config['dataset']['val_csv'] = 'outputs/kaggle_val_split.csv'
    config['dataset']['image_dir'] = os.path.join(kaggle_dir, 'train', 'train_images')
    config['dataset']['json_path'] = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    config['training']['batch_size'] = 4
    
    try:
        dataset = MAGFiLODataset(
            csv_path=os.path.join(base_dir, config['dataset']['train_csv']),
            img_dir=config['dataset']['image_dir'],
            json_path=config['dataset']['json_path'],
            transforms=get_training_augmentation(config)
        )
        
        # Test 1 raw image
        row = dataset.df.iloc[0]
        img_id = str(row['image_id'])
        filename = str(row['file_name'])
        import cv2
        from src.data.masks import create_semantic_mask
        img_path = os.path.join(dataset.img_dir, filename)
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        anns = dataset.parser.get_annotations_for_image(img_id)
        raw_mask = create_semantic_mask(anns, raw_img.shape[0], raw_img.shape[1]) if anns else np.zeros_like(raw_img)
        
        report['image_shape'] = list(raw_img.shape)
        report['raw_mask_shape'] = list(raw_mask.shape)
        
        # 2. Patch Pipeline
        patch_img, patch_mask = dataset[0]
        report['patch_image_shape'] = list(patch_img.shape)
        report['patch_mask_shape'] = list(patch_mask.shape)
        report['mask_statistics'] = {
            "min": float(patch_mask.min()),
            "max": float(patch_mask.max()),
            "unique_values": [float(x) for x in torch.unique(patch_mask)]
        }
        
        # 3. Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_baseline_model(config)
        model = model.to(device)
        report['model_parameter_count'] = sum(p.numel() for p in model.parameters())
        report['GPU_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        report['GPU_count'] = torch.cuda.device_count()
        report['device_used'] = device.type
        
        # 4. One Real GPU Training Step
        loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        batch = next(iter(loader))
        imgs, masks = batch[0].to(device), batch[1].to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        bce_fn = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                dice_loss = loss_fn(preds, masks)
                bce_loss = bce_fn(preds, masks)
                loss = dice_loss + bce_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            dice_loss = loss_fn(preds, masks)
            bce_loss = bce_fn(preds, masks)
            loss = dice_loss + bce_loss
            loss.backward()
            optimizer.step()
            
        batch_time = time.time() - start_time
        
        # Verify gradients
        has_nans = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_nans = True
                    break
                    
        report['batch_size'] = config['training']['batch_size']
        report['loss'] = float(loss.item())
        report['BCE_component'] = float(bce_loss.item())
        report['Dice_component'] = float(dice_loss.item())
        report['batch_time_seconds'] = batch_time
        report['peak_GPU_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0
        report['AMP_status'] = "Enabled" if scaler else "Disabled"
        report['gradient_status'] = "Finite" if not has_nans else "Contains NaNs/Infs"
        report['status'] = "PASS" if device.type == 'cuda' else "FAIL_NO_CUDA"
        
    except Exception as e:
        report['status'] = "FAIL"
        report['error'] = str(e)
        import traceback
        report['traceback'] = traceback.format_exc()

    out_dir = os.path.join(base_dir, 'outputs')
    if not os.access(base_dir, os.W_OK) and os.path.exists('/kaggle/working'):
        out_dir = '/kaggle/working/outputs'
        
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kaggle_gpu_smoke_test.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(json.dumps(report, indent=2))
    
    if report.get('status') == 'PASS':
        print("\nGPU SMOKE TEST: PASS")
    else:
        print("\nGPU SMOKE TEST: FAIL")
    
if __name__ == '__main__':
    run_smoke_test()
