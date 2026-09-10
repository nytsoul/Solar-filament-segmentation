import numpy as np
import cv2

def semantic_to_instances(prob_map: np.ndarray, prob_threshold: float = 0.5, min_area: int = 100) -> np.ndarray:
    """
    Convert a semantic probability map to an instance mask using connected components.
    
    Args:
        prob_map: 2D numpy array with probabilities [0, 1].
        prob_threshold: Threshold to binarize the probability map.
        min_area: Minimum area (pixels) for a component to be kept.
        
    Returns:
        instance_mask: 2D numpy array where 0 is background and 1..N are unique instance IDs.
    """
    binary_mask = (prob_map >= prob_threshold).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    instance_mask = np.zeros_like(labels, dtype=np.int32)
    current_instance_id = 1
    
    # stats[0] is the background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            instance_mask[labels == i] = current_instance_id
            current_instance_id += 1
            
    return instance_mask
