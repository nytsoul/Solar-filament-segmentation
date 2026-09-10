import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

def create_individual_mask(annotation: Dict[str, Any], height: int = 2048, width: int = 2048) -> np.ndarray:
    """Create a binary mask for a single filament annotation."""
    mask = np.zeros((height, width), dtype=np.uint8)
    segmentation = annotation.get('segmentation', [])
    for seg in segmentation:
        pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask

def create_instance_masks(annotations: List[Dict[str, Any]], height: int = 2048, width: int = 2048) -> np.ndarray:
    """Create an instance mask where each filament has a unique integer ID."""
    instance_mask = np.zeros((height, width), dtype=np.int32)
    for i, ann in enumerate(annotations, start=1):
        segmentation = ann.get('segmentation', [])
        for seg in segmentation:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            # Create a temporary binary mask for this poly to handle overlaps
            temp = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp, [pts], 1)
            instance_mask[temp == 1] = i
    return instance_mask

def create_semantic_mask(annotations: List[Dict[str, Any]], height: int = 2048, width: int = 2048) -> np.ndarray:
    """Create a combined semantic mask for all filaments (binary)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        for seg in segmentation:
            pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
    return mask

def calculate_mask_area(mask: np.ndarray) -> int:
    """Calculate the area of a binary mask (number of non-zero pixels)."""
    return int(np.count_nonzero(mask))

def calculate_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate bounding box [x, y, w, h] from a binary mask. Returns (0,0,0,0) if empty."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return (0, 0, 0, 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

def calculate_connected_components(mask: np.ndarray) -> int:
    """Calculate the number of connected components in a binary mask."""
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return max(0, num_labels - 1)
