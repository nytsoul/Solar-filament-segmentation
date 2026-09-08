import os
import sys
import json
import random
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_individual_mask, create_instance_masks, create_semantic_mask, calculate_mask_area, calculate_bounding_box, calculate_connected_components

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
    train_dir = os.path.join(dataset_dir, 'train')
    train_images_dir = os.path.join(train_dir, 'train_images')
    json_path = os.path.join(train_dir, 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    
    out_dir = os.path.join(base_dir, 'outputs', 'mask_verification')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading annotations...")
    parser = MAGFiLOAnnotationParser(json_path)
    
    # Filter for valid images in the directory
    valid_images = []
    for img_id, filename in parser.img_id_to_filename.items():
        if os.path.exists(os.path.join(train_images_dir, filename)):
            # only check images with annotations
            if parser.get_annotations_for_image(img_id):
                valid_images.append((img_id, filename))
            
    print(f"Found {len(valid_images)} images with annotations.")
    
    random.seed(42)
    selected_images = random.sample(valid_images, min(20, len(valid_images)))
    
    report = {
        "verified_images_count": 0,
        "total_instances_reconstructed": 0,
        "empty_or_invalid_masks": 0,
        "filament_areas": [],
        "all_dimensions_match": True,
        "problems": []
    }
    
    for img_id, filename in selected_images:
        img_path = os.path.join(train_images_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Sanity check: dimensions match json
        img_info = parser.get_image_info(img_id)
        if img_info.get('height') != h or img_info.get('width') != w:
            report['all_dimensions_match'] = False
            report['problems'].append(f"Image {filename} dimensions {w}x{h} don't match JSON {img_info.get('width')}x{img_info.get('height')}")
        
        annotations = parser.get_annotations_for_image(img_id)
        if not annotations:
            continue
            
        instance_mask = create_instance_masks(annotations, h, w)
        semantic_mask = create_semantic_mask(annotations, h, w)
        
        # Verify individual masks
        seen_ids = set()
        for ann in annotations:
            ann_id = ann.get('id')
            if ann_id in seen_ids:
                report['problems'].append(f"Duplicate annotation ID {ann_id} in image {filename}")
            seen_ids.add(ann_id)
            
            # create single mask
            single_mask = create_individual_mask(ann, h, w)
            area = calculate_mask_area(single_mask)
            
            if area == 0:
                report['empty_or_invalid_masks'] += 1
                report['problems'].append(f"Empty mask for annotation {ann_id} in {filename}")
            else:
                report['filament_areas'].append(area)
                
            if area < 10:
                report['problems'].append(f"Tiny mask (area={area}) for annotation {ann_id} in {filename}")
            if area > (h * w * 0.5):
                report['problems'].append(f"Huge mask (area={area}) for annotation {ann_id} in {filename}")
                
            report['total_instances_reconstructed'] += 1
            
        # Create visualizations
        # 1. Instance mask visualization (colormap)
        # normalize instance mask to 0-255 for visualization
        if np.max(instance_mask) > 0:
            norm_inst = (instance_mask * (255 / np.max(instance_mask))).astype(np.uint8)
            inst_color = cv2.applyColorMap(norm_inst, cv2.COLORMAP_JET)
            # mask out background
            inst_color[instance_mask == 0] = 0
        else:
            inst_color = np.zeros_like(img)
            
        # 2. Semantic mask visualization
        sem_vis = (semantic_mask * 255).astype(np.uint8)
        
        # 3. Overlay on original
        overlay = img.copy()
        # Overlay semantic mask in red
        overlay[semantic_mask == 1] = [0, 0, 255]
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, overlay)
        
        # Draw boundaries (contours) on another overlay
        boundaries = img.copy()
        contours, _ = cv2.findContours(semantic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundaries, contours, -1, (0, 255, 0), 2)
        
        # Save
        cv2.imwrite(os.path.join(out_dir, f"{filename}_instance.png"), inst_color)
        cv2.imwrite(os.path.join(out_dir, f"{filename}_semantic.png"), sem_vis)
        cv2.imwrite(os.path.join(out_dir, f"{filename}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(out_dir, f"{filename}_boundaries.png"), boundaries)
        
        report['verified_images_count'] += 1
        
    # Summarize areas
    if report['filament_areas']:
        areas = report['filament_areas']
        report['area_stats'] = {
            "min": int(np.min(areas)),
            "max": int(np.max(areas)),
            "median": float(np.median(areas)),
            "mean": float(np.mean(areas))
        }
    
    report_path = os.path.join(base_dir, 'outputs', 'mask_verification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Verified {report['verified_images_count']} images.")
    print(f"Reconstructed {report['total_instances_reconstructed']} filaments.")
    print(f"Found {report['empty_or_invalid_masks']} empty masks.")
    if 'area_stats' in report:
        print(f"Area stats: Min {report['area_stats']['min']}, Median {report['area_stats']['median']:.1f}, Max {report['area_stats']['max']}")
    print(f"Problems found: {len(report['problems'])}")
    for p in report['problems'][:5]:
        print(" -", p)
    if len(report['problems']) > 5:
        print(f"   ... and {len(report['problems']) - 5} more.")

if __name__ == '__main__':
    main()
