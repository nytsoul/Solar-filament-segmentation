import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.annotations import MAGFiLOAnnotationParser
from src.data.masks import create_semantic_mask

class MAGFiLODataset(Dataset):
    """Original v1 dataset. Kept for backward compatibility."""
    def __init__(self, csv_path, img_dir, json_path, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.parser = MAGFiLOAnnotationParser(json_path)
        self.transforms = transforms
        
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
            
        # reconstruct mask
        annotations = self.parser.get_annotations_for_image(img_id)
        if len(annotations) > 0:
            h, w = img.shape
            mask = create_semantic_mask(annotations, height=h, width=w)
        else:
            mask = np.zeros(img.shape, dtype=np.uint8)
            
        # img must be HWC for albumentations
        img = img[..., np.newaxis] # (H, W, 1)
        
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # mask should be (1, H, W) float
        mask = mask.unsqueeze(0).float()
        
        return img, mask


class MAGFiLOPatchDataset(Dataset):
    """
    V2 dataset with region-aware patch sampling.
    
    On each __getitem__, selects a patch sampling strategy based on
    configurable probabilities:
      - filament: crop around a random filament annotation
      - disk: random crop from on-disk region
      - limb: crop centered on the solar limb ring
      - background: crop from off-disk background
    
    Records sampling statistics for diagnostics.
    """
    
    def __init__(self, csv_path, img_dir, json_path, config, transforms=None, return_meta=True):
        """
        Args:
            csv_path: Path to split CSV.
            img_dir: Image directory.
            json_path: Annotation JSON path.
            config: Full config dict (needs 'sampling', 'dataset' sections).
            transforms: Albumentations transform (v2, no crop).
            return_meta: If True, returns (img_patch, mask_patch, meta_dict).
        """
        from src.data.solar_disk import detect_solar_disk, get_limb_mask, get_background_mask, sample_patch_location
        
        self.return_meta = return_meta
        
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.parser = MAGFiLOAnnotationParser(json_path)
        self.transforms = transforms
        
        self.patch_size = config['dataset'].get('patch_size', 768)
        self.patches_per_image = config['training'].get('patches_per_image', 4)
        
        sampling = config.get('sampling', {})
        self.p_filament = sampling.get('p_filament', 0.50)
        self.p_disk = sampling.get('p_disk', 0.20)
        self.p_limb = sampling.get('p_limb', 0.15)
        self.p_background = sampling.get('p_background', 0.15)
        self.limb_width = sampling.get('limb_width', 50)
        
        # Normalize probabilities
        total = self.p_filament + self.p_disk + self.p_limb + self.p_background
        self.p_filament /= total
        self.p_disk /= total
        self.p_limb /= total
        self.p_background /= total
        
        self._sample_patch_location = sample_patch_location
        self._detect_solar_disk = detect_solar_disk
        self._get_limb_mask = get_limb_mask
        self._get_background_mask = get_background_mask
        
        # Sampling statistics (reset each epoch)
        self.sample_counts = {'filament': 0, 'disk': 0, 'limb': 0, 'background': 0, 'fallback': 0}
        self.patches_with_filament = 0
        self.patches_without_filament = 0
        
        # Precompute annotation bounding boxes per image for filament sampling
        self._precompute_filament_locations()

    def _precompute_filament_locations(self):
        """Precompute filament center locations for each image."""
        self.filament_centers = {}
        for _, row in self.df.iterrows():
            img_id = str(row['image_id'])
            anns = self.parser.get_annotations_for_image(img_id)
            centers = []
            for ann in anns:
                bbox = ann.get('bbox', None)
                if bbox is not None:
                    x, y, bw, bh = bbox
                    cy = int(y + bh / 2)
                    cx = int(x + bw / 2)
                    centers.append((cy, cx))
                else:
                    # Try to compute from segmentation
                    segs = ann.get('segmentation', [])
                    for seg in segs:
                        pts = np.array(seg).reshape(-1, 2)
                        if len(pts) > 0:
                            cx = int(pts[:, 0].mean())
                            cy = int(pts[:, 1].mean())
                            centers.append((cy, cx))
                            break
            self.filament_centers[img_id] = centers

    def __len__(self):
        return len(self.df) * self.patches_per_image

    def reset_stats(self):
        """Reset sampling statistics (call at the start of each epoch)."""
        self.sample_counts = {'filament': 0, 'disk': 0, 'limb': 0, 'background': 0, 'fallback': 0}
        self.patches_with_filament = 0
        self.patches_without_filament = 0

    def get_stats(self):
        """Return current sampling statistics."""
        total = sum(self.sample_counts.values())
        total_patches = self.patches_with_filament + self.patches_without_filament
        return {
            'sample_counts': dict(self.sample_counts),
            'total_patches': total,
            'pct_filament_strategy': self.sample_counts['filament'] / max(total, 1) * 100,
            'pct_disk_strategy': self.sample_counts['disk'] / max(total, 1) * 100,
            'pct_limb_strategy': self.sample_counts['limb'] / max(total, 1) * 100,
            'pct_background_strategy': self.sample_counts['background'] / max(total, 1) * 100,
            'pct_patches_with_filament': self.patches_with_filament / max(total_patches, 1) * 100,
            'pct_patches_without_filament': self.patches_without_filament / max(total_patches, 1) * 100,
        }

    def __getitem__(self, idx):
        # Map idx to image index
        img_idx = idx // self.patches_per_image
        row = self.df.iloc[img_idx]
        img_id = str(row['image_id'])
        filename = str(row['file_name'])
        
        img_path = os.path.join(self.img_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        
        h, w = img.shape
        
        # Reconstruct full mask
        annotations = self.parser.get_annotations_for_image(img_id)
        if len(annotations) > 0:
            mask = create_semantic_mask(annotations, height=h, width=w)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        # Detect solar disk regions
        disk_info = self._detect_solar_disk(img)
        disk_mask = disk_info['disk_mask']
        limb_mask = self._get_limb_mask(disk_mask, disk_info['center'], disk_info['radius'], self.limb_width)
        bg_mask = self._get_background_mask(disk_mask)
        
        # Choose sampling strategy
        strategy = np.random.choice(
            ['filament', 'disk', 'limb', 'background'],
            p=[self.p_filament, self.p_disk, self.p_limb, self.p_background]
        )
        
        y, x = None, None
        
        if strategy == 'filament':
            centers = self.filament_centers.get(img_id, [])
            if len(centers) > 0:
                ci = np.random.randint(len(centers))
                cy, cx = centers[ci]
                # Center patch on the filament with some random jitter
                jitter = self.patch_size // 4
                y = cy - self.patch_size // 2 + np.random.randint(-jitter, jitter + 1)
                x = cx - self.patch_size // 2 + np.random.randint(-jitter, jitter + 1)
                y = max(0, min(y, h - self.patch_size))
                x = max(0, min(x, w - self.patch_size))
            else:
                strategy = 'disk'  # fallback if no filaments

        if strategy == 'disk':
            loc = self._sample_patch_location(disk_mask, self.patch_size)
            if loc is not None:
                y, x = loc
            
        elif strategy == 'limb':
            loc = self._sample_patch_location(limb_mask, self.patch_size)
            if loc is not None:
                y, x = loc
                
        elif strategy == 'background':
            loc = self._sample_patch_location(bg_mask, self.patch_size)
            if loc is not None:
                y, x = loc
        
        # Final fallback: random crop
        if y is None or x is None:
            y = np.random.randint(0, max(1, h - self.patch_size))
            x = np.random.randint(0, max(1, w - self.patch_size))
            self.sample_counts['fallback'] += 1
        else:
            self.sample_counts[strategy] += 1
        
        # Extract patch
        img_patch = img[y:y + self.patch_size, x:x + self.patch_size]
        mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]
        
        # Track filament presence
        if mask_patch.sum() > 0:
            self.patches_with_filament += 1
        else:
            self.patches_without_filament += 1
        
        # Expand to HWC for albumentations
        img_patch = img_patch[..., np.newaxis]  # (patch_size, patch_size, 1)
        
        if self.transforms:
            augmented = self.transforms(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']
        
        # mask -> (1, H, W) float
        mask_patch = mask_patch.unsqueeze(0).float()
        
        if self.return_meta:
            has_fil = 1 if (mask_patch.sum() > 0) else 0
            return img_patch, mask_patch, {'strategy': strategy, 'has_filament': has_fil}
        return img_patch, mask_patch
