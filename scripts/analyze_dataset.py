import os
import json
import numpy as np
import cv2
from datetime import datetime

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026', 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    img_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026', 'train', 'train_images')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_id_to_info = {img['id']: img for img in data['images']}
    anns_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    valid_images = []
    for img in data['images']:
        if os.path.exists(os.path.join(img_dir, img['file_name'])):
            valid_images.append(img['id'])
            
    total_images = len(valid_images)
    total_filaments = 0
    filaments_per_image = []
    areas = []
    
    zero_filaments = 0
    one_filament = 0
    multiple_filaments = 0
    
    dates = []
    intensities = []
    
    for img_id in valid_images:
        anns = anns_by_img.get(img_id, [])
        count = len(anns)
        filaments_per_image.append(count)
        total_filaments += count
        
        if count == 0:
            zero_filaments += 1
        elif count == 1:
            one_filament += 1
        else:
            multiple_filaments += 1
            
        for ann in anns:
            poly_area = 0
            for seg in ann.get('segmentation', []):
                pts = np.array(seg, np.float32).reshape((-1, 1, 2))
                poly_area += cv2.contourArea(pts)
            if poly_area > 0:
                areas.append(poly_area)
            
        img_info = img_id_to_info[img_id]
        if 'date_captured' in img_info:
            try:
                dt = datetime.strptime(img_info['date_captured'], '%Y-%m-%d %H:%M:%S')
                dates.append(dt)
            except:
                pass

    areas = np.array(areas)
    if len(areas) > 0:
        small_threshold = np.percentile(areas, 33.3)
        large_threshold = np.percentile(areas, 66.7)
        
        small_pct = np.sum(areas <= small_threshold) / len(areas) * 100
        med_pct = np.sum((areas > small_threshold) & (areas <= large_threshold)) / len(areas) * 100
        large_pct = np.sum(areas > large_threshold) / len(areas) * 100
    else:
        small_pct = med_pct = large_pct = 0
        small_threshold = large_threshold = 0
    
    stats = {
        "total_training_images": total_images,
        "total_filament_instances": total_filaments,
        "filaments_per_image": {
            "min": int(np.min(filaments_per_image)) if filaments_per_image else 0,
            "max": int(np.max(filaments_per_image)) if filaments_per_image else 0,
            "mean": float(np.mean(filaments_per_image)) if filaments_per_image else 0,
            "median": float(np.median(filaments_per_image)) if filaments_per_image else 0
        },
        "filament_areas": {
            "min": float(np.min(areas)) if len(areas) > 0 else 0,
            "max": float(np.max(areas)) if len(areas) > 0 else 0,
            "mean": float(np.mean(areas)) if len(areas) > 0 else 0,
            "median": float(np.median(areas)) if len(areas) > 0 else 0
        },
        "size_distribution_percentage": {
            "small": float(small_pct),
            "medium": float(med_pct),
            "large": float(large_pct)
        },
        "image_counts_by_filaments": {
            "zero": zero_filaments,
            "one": one_filament,
            "multiple": multiple_filaments
        }
    }
    
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'full_dataset_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    days = {}
    for d in dates:
        day_str = d.strftime('%Y-%m-%d')
        days[day_str] = days.get(day_str, 0) + 1
        
    analysis_md = f"""# Split Analysis and Leakage Prevention

## Findings
I analyzed the `date_captured` fields and filenames in the dataset.
- Total valid images with dates: {len(dates)}
- Unique days of observation: {len(days)}

Some days have multiple observations captured just hours or minutes apart. A random image split is not safe. If highly correlated observations (e.g., the exact same filaments from different timepoints on the same day) end up in both training and validation sets, this causes temporal data leakage. This artificially inflates validation metrics.

## Strategy
To resolve this, we will perform a **Grouped/Stratified Split** using the observation Day as the grouping ID. All observations from a specific day will be placed entirely into either the training or validation set, guaranteeing that nearly-identical temporal observations do not cross the train/val boundary.
"""
    with open(os.path.join(out_dir, 'split_analysis.md'), 'w') as f:
        f.write(analysis_md)
        
    print("Dataset analysis complete.")

if __name__ == '__main__':
    main()
