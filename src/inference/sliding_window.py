import torch
import torch.nn.functional as F
import numpy as np

def predict_full_image(model, image_tensor, patch_size=768, overlap=0.25, device=None):
    """
    Run sliding-window inference on a full-resolution image.
    
    Args:
        model: PyTorch model.
        image_tensor: Tensor of shape (1, C, H, W).
        patch_size: Size of the patches (default: 768).
        overlap: Fraction of patch_size to overlap (default: 0.25).
        device: Device to run inference on.
        
    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    _, C, H, W = image_tensor.shape
    stride = int(patch_size * (1 - overlap))
    
    prob_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Calculate padded dimensions if necessary
    pad_h = (patch_size - H % stride) % patch_size if H % stride != 0 else 0
    pad_w = (patch_size - W % stride) % patch_size if W % stride != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    _, _, padded_H, padded_W = image_tensor.shape
    padded_prob_map = torch.zeros((padded_H, padded_W), dtype=torch.float32, device=device)
    padded_count_map = torch.zeros((padded_H, padded_W), dtype=torch.float32, device=device)
    
    # Create a 2D Gaussian or triangular weight map to reduce boundary artifacts
    # For simplicity, we use a constant weight of 1.0 (mean averaging)
    weight = torch.ones((patch_size, patch_size), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for y in range(0, padded_H - patch_size + 1, stride):
            for x in range(0, padded_W - patch_size + 1, stride):
                patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size].to(device)
                
                # Forward pass
                with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.autocast('cpu', enabled=False):
                    logits = model(patch)
                    
                probs = torch.sigmoid(logits).squeeze()
                
                padded_prob_map[y:y+patch_size, x:x+patch_size] += probs * weight
                padded_count_map[y:y+patch_size, x:x+patch_size] += weight

    # Normalize by count map
    padded_prob_map = padded_prob_map / (padded_count_map + 1e-7)
    
    # Crop back to original size
    final_prob_map = padded_prob_map[:H, :W].cpu().numpy()
    
    return final_prob_map
