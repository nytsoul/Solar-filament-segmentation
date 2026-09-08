import os
import json
import cv2
import glob
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
    train_dir = os.path.join(dataset_dir, 'train')
    train_images_dir = os.path.join(train_dir, 'train_images')
    json_path = os.path.join(train_dir, 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    vis_dir = os.path.join(base_dir, 'outputs', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Loading JSON for visualization...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    anns_by_filename = {}
    img_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        filename = img_id_to_filename.get(img_id)
        if not filename: continue
        if filename not in anns_by_filename:
            anns_by_filename[filename] = []
        anns_by_filename[filename].append(ann)
        
    visualized_filenames = set()
    for filename, annotations in anns_by_filename.items():
        if len(visualized_filenames) >= 10:
            break
            
        img_path = os.path.join(train_images_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        overlay = img.copy()
        for ann in annotations:
            for seg in ann.get('segmentation', []):
                pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
            if 'spine' in ann:
                spine = ann['spine']
                spine_pts = np.array(spine, np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [spine_pts], isClosed=False, color=(0, 255, 0), thickness=2)
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        out_path = os.path.join(vis_dir, f"vis_{filename}")
        cv2.imwrite(out_path, img)
        visualized_filenames.add(filename)
        print(f"Saved visualization for {filename}")

if __name__ == '__main__':
    main()
