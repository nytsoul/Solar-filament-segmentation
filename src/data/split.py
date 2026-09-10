import os
import json
import random
import csv
from collections import defaultdict
from datetime import datetime

def generate_splits(json_path: str, img_dir: str, train_out: str, val_out: str, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    valid_images = []
    for img in data['images']:
        if os.path.exists(os.path.join(img_dir, img['file_name'])):
            valid_images.append(img)
            
    # Group by Year-Month-Day to avoid temporal leakage
    grouped_images = defaultdict(list)
    for img in valid_images:
        date_str = img.get('date_captured', '')
        if date_str:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                group_key = dt.strftime('%Y-%m-%d')
            except ValueError:
                group_key = 'unknown'
        else:
            group_key = 'unknown'
            
        grouped_images[group_key].append(img)
        
    groups = list(grouped_images.keys())
    groups.sort() # Ensure determinism
    random.shuffle(groups)
    
    train_images = []
    val_images = []
    
    target_val_count = int(len(valid_images) * val_ratio)
    
    for group in groups:
        if len(val_images) < target_val_count:
            val_images.extend(grouped_images[group])
        else:
            train_images.extend(grouped_images[group])
            
    # Verify no overlap
    train_ids = set(img['id'] for img in train_images)
    val_ids = set(img['id'] for img in val_images)
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, "Data Leakage Detected: Image IDs overlap between train and val."
    
    # Save splits
    for split_data, out_path in [(train_images, train_out), (val_images, val_out)]:
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'file_name', 'date_captured'])
            for img in split_data:
                writer.writerow([img['id'], img['file_name'], img.get('date_captured', '')])
                
    # Calculate instances
    anns_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    train_filaments = sum(len(anns_by_img.get(img_id, [])) for img_id in train_ids)
    val_filaments = sum(len(anns_by_img.get(img_id, [])) for img_id in val_ids)
    
    print("--- Split Results ---")
    print(f"Total training images: {len(valid_images)}")
    print(f"Total filament instances: {train_filaments + val_filaments}")
    print(f"Train image count: {len(train_images)}")
    print(f"Validation image count: {len(val_images)}")
    print(f"Train filament count: {train_filaments}")
    print(f"Validation filament count: {val_filaments}")
    print(f"Overlap detected: {len(overlap) > 0}")
    print(f"Temporal/group leakage handled: Yes, grouped by observation Day.")
    print(f"Splits saved to {train_out} and {val_out}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026', 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    img_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026', 'train', 'train_images')
    train_out = os.path.join(base_dir, 'outputs', 'train_split.csv')
    val_out = os.path.join(base_dir, 'outputs', 'val_split.csv')
    
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    generate_splits(json_path, img_dir, train_out, val_out)
