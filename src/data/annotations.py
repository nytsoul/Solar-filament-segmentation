import json
import numpy as np
from typing import Dict, List, Any

class MAGFiLOAnnotationParser:
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.img_id_to_filename = {img['id']: img['file_name'] for img in self.data['images']}
        self.filename_to_img_id = {v: k for k, v in self.img_id_to_filename.items()}
        self.img_id_to_info = {img['id']: img for img in self.data['images']}
        
        self.anns_by_img_id = {}
        for ann in self.data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.anns_by_img_id:
                self.anns_by_img_id[img_id] = []
            self.anns_by_img_id[img_id].append(ann)
            
    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        return self.img_id_to_info.get(image_id, {})
        
    def get_annotations_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """Returns the list of raw annotation dictionaries for the given image."""
        return self.anns_by_img_id.get(image_id, [])

    def parse_segmentation_to_polygon(self, segmentation: List[List[float]]) -> List[np.ndarray]:
        """Convert COCO segmentation list to list of numpy polygons."""
        polygons = []
        for seg in segmentation:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            polygons.append(pts)
        return polygons
