import os
import json
import cv2
import pandas as pd
import numpy as np
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_semantic_mask

def main():
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    if not os.path.exists(kaggle_dir):
        kaggle_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
        
    train_images_dir = os.path.join(kaggle_dir, 'train', 'train_images')
    json_path = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    
    images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
    num_images = len(images)
    
    parser = MAGFiLOAnnotationParser(json_path)
    
    file_to_img_id = parser.filename_to_img_id
    
    successful_masks = 0
    checked_files = 0
    total_annotations = 0
    
    for filename in images:
        if filename in file_to_img_id:
            img_id = str(file_to_img_id[filename])
            anns = parser.get_annotations_for_image(img_id)
        else:
            anns = []
            
        total_annotations += len(anns) if anns else 0
        
        if checked_files < 10:
            img_path = os.path.join(train_images_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                mask = create_semantic_mask(anns, img.shape[0], img.shape[1]) if anns else np.zeros(img.shape)
                if mask.shape == (2048, 2048):
                    successful_masks += 1
            checked_files += 1

    report = {
        "dataset_path": kaggle_dir,
        "num_images_found": num_images,
        "num_annotations_mapped": total_annotations,
        "masks_reconstructed": successful_masks,
        "target_resolution": [2048, 2048],
        "status": "Success" if num_images == 707 and successful_masks == 10 else "Failed"
    }
    
    os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)
    with open(os.path.join(base_dir, 'outputs', 'kaggle_integration_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Integration verify completed: {num_images} images, {total_annotations} annotations mapped.")
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
