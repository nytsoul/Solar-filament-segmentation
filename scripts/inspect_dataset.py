import os
import json
import cv2
import glob

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'MAGFiLO_1.0_Kaggle_2026')
    
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    train_images_dir = os.path.join(train_dir, 'train_images')
    test_images_dir = os.path.join(test_dir, 'test_images')
    
    json_path = os.path.join(train_dir, 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    
    report = {
        "dataset_dir": dataset_dir,
        "train_images_dir": train_images_dir,
        "test_images_dir": test_images_dir,
        "annotation_json": json_path,
        "annotation_info": {},
        "train_images_info": {},
        "test_images_info": {}
    }
    
    print("Inspecting annotation JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    report["annotation_info"] = {
        "top_level_keys": list(data.keys()),
        "categories": data.get('categories', []),
        "number_of_images_in_json": len(data.get('images', [])),
        "number_of_annotations": len(data.get('annotations', [])),
        "segmentation_format": "COCO Polygon (list of lists of floats)",
        "bounding_box_format": "[x, y, width, height]",
        "additional_features": ["spine format detected"] if 'spine' in data.get('annotations', [{}])[0] else []
    }
    
    # Analyze annotations per image
    ann_by_img = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        ann_by_img[img_id] = ann_by_img.get(img_id, 0) + 1
        
    if ann_by_img:
        counts = list(ann_by_img.values())
        report["annotation_info"]["filaments_per_image"] = {
            "min": min(counts),
            "max": max(counts),
            "avg": sum(counts) / len(counts)
        }
        
    print("Inspecting training images...")
    train_imgs = glob.glob(os.path.join(train_images_dir, '*'))
    report["train_images_info"]["count"] = len(train_imgs)
    
    if train_imgs:
        sample_img = cv2.imread(train_imgs[0], cv2.IMREAD_UNCHANGED)
        report["train_images_info"]["sample_shape"] = sample_img.shape
        report["train_images_info"]["dtype"] = str(sample_img.dtype)
        report["train_images_info"]["min_val"] = int(sample_img.min())
        report["train_images_info"]["max_val"] = int(sample_img.max())
        report["train_images_info"]["channels"] = 1 if len(sample_img.shape) == 2 else sample_img.shape[2]
        
    print("Inspecting test images...")
    test_imgs = glob.glob(os.path.join(test_images_dir, '*'))
    report["test_images_info"]["count"] = len(test_imgs)
    
    report_out_path = os.path.join(base_dir, 'outputs', 'dataset_report.json')
    os.makedirs(os.path.dirname(report_out_path), exist_ok=True)
    with open(report_out_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n--- Dataset Inspection Summary ---")
    print(f"Dataset Path: {dataset_dir}")
    print(f"Train Images (in dir): {len(train_imgs)}")
    print(f"Test Images (in dir): {len(test_imgs)}")
    if train_imgs:
        print(f"Image Dimensions: {sample_img.shape}, Channels: {report['train_images_info']['channels']}, Dtype: {sample_img.dtype}")
    print(f"Annotations JSON: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations.")
    if ann_by_img:
        print(f"Filaments per image: Min {min(counts)}, Max {max(counts)}, Avg {sum(counts)/len(counts):.2f}")
    print(f"Format: COCO Polygon with Bounding Box.")
    print(f"Report saved to {report_out_path}")

if __name__ == '__main__':
    main()
