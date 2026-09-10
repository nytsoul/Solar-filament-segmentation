import numpy as np

def calculate_pq(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.5):
    """
    Calculate Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
    Also returns matching statistics and failure modes (fragmentation/merging).
    
    Args:
        pred_mask: 2D numpy array of predicted instance IDs (0 is background).
        gt_mask: 2D numpy array of ground-truth instance IDs (0 is background).
        iou_threshold: Threshold for a TP match.
        
    Returns:
        dict: containing pq, sq, rq, tp, fp, fn, and failure analysis counts.
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids != 0]
    
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]
    
    num_pred = len(pred_ids)
    num_gt = len(gt_ids)
    
    if num_pred == 0 and num_gt == 0:
        return {
            'pq': 1.0, 'sq': 1.0, 'rq': 1.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'fragmentation_count': 0,
            'merging_count': 0,
            'pred_count': 0,
            'gt_count': 0
        }
        
    if num_pred == 0:
        return {
            'pq': 0.0, 'sq': 0.0, 'rq': 0.0,
            'tp': 0, 'fp': 0, 'fn': num_gt,
            'fragmentation_count': 0,
            'merging_count': 0,
            'pred_count': 0,
            'gt_count': num_gt
        }
        
    if num_gt == 0:
        return {
            'pq': 0.0, 'sq': 0.0, 'rq': 0.0,
            'tp': 0, 'fp': num_pred, 'fn': 0,
            'fragmentation_count': 0,
            'merging_count': 0,
            'pred_count': num_pred,
            'gt_count': 0
        }

    # Map to contiguous IDs [1..N]
    pred_map = {id: i+1 for i, id in enumerate(pred_ids)}
    gt_map = {id: i+1 for i, id in enumerate(gt_ids)}
    
    mapped_pred = np.zeros_like(pred_mask)
    for k, v in pred_map.items(): mapped_pred[pred_mask == k] = v
    
    mapped_gt = np.zeros_like(gt_mask)
    for k, v in gt_map.items(): mapped_gt[gt_mask == k] = v
        
    # Fast 2D intersection using bincount
    max_gt = num_gt + 1
    combined_idx = mapped_pred.astype(np.int64) * max_gt + mapped_gt.astype(np.int64)
    counts = np.bincount(combined_idx.ravel(), minlength=(num_pred+1)*max_gt)
    intersection_matrix = counts.reshape(num_pred+1, max_gt)
    
    pred_areas = intersection_matrix.sum(axis=1) # Sum over GTs
    gt_areas = intersection_matrix.sum(axis=0)   # Sum over Preds
    
    tp = 0
    iou_sum = 0.0
    
    fragmentation_count = 0
    merging_count = 0
    
    for p in range(1, num_pred + 1):
        for g in range(1, num_gt + 1):
            intersection = intersection_matrix[p, g]
            if intersection == 0:
                continue
                
            union = pred_areas[p] + gt_areas[g] - intersection
            iou = intersection / union if union > 0 else 0.0
            
            if iou >= iou_threshold:
                tp += 1
                iou_sum += iou
                
    # Detect fragmentation and merging
    # Fragmentation: A single GT intersects significantly with multiple predictions (> 10% of prediction area)
    for g in range(1, num_gt + 1):
        intersecting_preds = np.where(intersection_matrix[1:, g] > 0.1 * pred_areas[1:])[0] + 1
        if len(intersecting_preds) > 1:
            fragmentation_count += 1
            
    # Merging: A single Prediction intersects significantly with multiple GTs (> 10% of GT area)
    for p in range(1, num_pred + 1):
        intersecting_gts = np.where(intersection_matrix[p, 1:] > 0.1 * gt_areas[1:])[0] + 1
        if len(intersecting_gts) > 1:
            merging_count += 1
            
    fp = num_pred - tp
    fn = num_gt - tp
    
    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = sq * rq
    
    return {
        'pq': pq,
        'sq': sq,
        'rq': rq,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fragmentation_count': fragmentation_count,
        'merging_count': merging_count,
        'pred_count': num_pred,
        'gt_count': num_gt
    }
