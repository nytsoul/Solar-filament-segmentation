import os
import json
import random
import pandas as pd
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.data.annotations import MAGFiLOAnnotationParser

def generate_kaggle_splits():
    kaggle_dir = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    if not os.path.exists(kaggle_dir):
        kaggle_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
        
    train_images_dir = os.path.join(kaggle_dir, 'train', 'train_images')
    json_path = os.path.join(kaggle_dir, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    
    parser = MAGFiLOAnnotationParser(json_path)
    images_on_disk = set(f for f in os.listdir(train_images_dir) if f.endswith('.jpeg') or f.endswith('.jpg'))
    
    records = []
    for img_id, img_info in parser.img_id_to_info.items():
        filename = img_info['file_name']
        if filename in images_on_disk:
            img_id = str(img_info['id'])
            date_captured = img_info.get('date_captured', 'unknown')
            day = date_captured.split('T')[0] if 'T' in date_captured else date_captured.split(' ')[0]
            
            anns = parser.get_annotations_for_image(img_id)
            num_filaments = len(anns) if anns else 0
            
            records.append({
                'image_id': img_id,
                'file_name': filename,
                'day': day,
                'num_filaments': num_filaments
            })
            
    df = pd.DataFrame(records)
    
    days = df['day'].unique()
    random.seed(42)
    random.shuffle(days)
    
    target_train_images = int(len(df) * 0.8)
    train_days = []
    val_days = []
    train_count = 0
    
    for day in days:
        day_count = len(df[df['day'] == day])
        if train_count + day_count <= target_train_images or train_count < target_train_images * 0.9:
            train_days.append(day)
            train_count += day_count
        else:
            val_days.append(day)
            
    train_df = df[df['day'].isin(train_days)]
    val_df = df[df['day'].isin(val_days)]
    
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(out_dir, 'kaggle_train_split.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'kaggle_val_split.csv'), index=False)
    
    analysis = f"""# Kaggle Split Analysis
Total Images: {len(df)}
Total Filaments: {df['num_filaments'].sum()}
Positive Images: {len(df[df['num_filaments'] > 0])}

## Train Split
Images: {len(train_df)}
Filaments: {train_df['num_filaments'].sum()}
Positive Images: {len(train_df[train_df['num_filaments'] > 0])}

## Validation Split
Images: {len(val_df)}
Filaments: {val_df['num_filaments'].sum()}
Positive Images: {len(val_df[val_df['num_filaments'] > 0])}

## Leakage Check
Days in both train and val: {len(set(train_df['day']).intersection(set(val_df['day'])))}
"""
    with open(os.path.join(out_dir, 'kaggle_split_analysis.md'), 'w') as f:
        f.write(analysis)
        
    print(analysis)

if __name__ == '__main__':
    generate_kaggle_splits()
