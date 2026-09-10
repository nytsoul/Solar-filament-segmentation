"""
Full-image validation dataset for 2048x2048 solar images.

Returns full-resolution images and masks for use with sliding-window inference
during validation. Batch size must be 1.
"""
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_semantic_mask


class MAGFiLOFullImageDataset(Dataset):
    """
    Dataset that returns full 2048x2048 images with their semantic masks.
    
    For use during validation with sliding-window inference.
    Images are normalized using get_inference_augmentation() externally.
    """

    def __init__(self, csv_path, img_dir, json_path, transform=None):
        """
        Args:
            csv_path: Path to CSV with columns [image_id, file_name, ...].
            img_dir: Directory containing the images.
            json_path: Path to the COCO-format annotation JSON.
            transform: Albumentations transform (should be get_inference_augmentation()).
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.parser = MAGFiLOAnnotationParser(json_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['image_id'])
        filename = str(row['file_name'])

        img_path = os.path.join(self.img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found.")

        h, w = img.shape

        # Reconstruct full-resolution semantic mask
        annotations = self.parser.get_annotations_for_image(img_id)
        if len(annotations) > 0:
            mask = create_semantic_mask(annotations, height=h, width=w)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        # Expand to HWC for albumentations
        img = img[..., np.newaxis]  # (H, W, 1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented['image']    # (1, H, W)
            mask = augmented['mask']           # (H, W)

        # mask -> (1, H, W) float
        import torch
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() if not isinstance(mask, np.ndarray) else torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor, filename
