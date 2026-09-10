"""
Unit tests for Full-Image Baseline v2 components:
- Solar disk detection and region masks
- Region-aware patch sampling
- Full-image dataset and shape consistency
"""
import pytest
import numpy as np
import cv2
import os
import torch
import yaml

from src.data.solar_disk import detect_solar_disk, get_limb_mask, get_background_mask, get_region_masks
from src.data.augmentations import get_training_augmentation_v2, get_inference_augmentation


def test_synthetic_solar_disk_detection():
    # Create a synthetic 512x512 image with a bright disk in center
    h, w = 512, 512
    img = np.zeros((h, w), dtype=np.uint8)
    cy, cx, r = 256, 256, 180
    cv2.circle(img, (cx, cy), r, 200, -1)
    # Add a bit of noise
    img = np.clip(img + np.random.randint(0, 15, (h, w), dtype=np.uint8), 0, 255)

    info = detect_solar_disk(img)
    assert abs(info['center'][0] - cy) < 15
    assert abs(info['center'][1] - cx) < 15
    assert abs(info['radius'] - r) < 20

    masks = get_region_masks(img, limb_width=20)
    disk_mask = masks['disk_mask']
    limb_mask = masks['limb_mask']
    bg_mask = masks['background_mask']
    assert disk_mask.shape == (h, w)
    assert limb_mask.shape == (h, w)
    assert bg_mask.shape == (h, w)

    # Disk mask should cover center
    assert disk_mask[cy, cx] == 1
    # Background mask should cover corners
    assert bg_mask[10, 10] == 1
    assert bg_mask[cy, cx] == 0


def test_v2_augmentations_shape():
    aug = get_training_augmentation_v2({'augmentations': {'p_hflip': 0.5}})
    # Dummy 768x768 patch
    img = np.random.randint(0, 255, (768, 768, 1), dtype=np.uint8)
    mask = np.zeros((768, 768), dtype=np.uint8)
    mask[300:350, 300:350] = 1

    res = aug(image=img, mask=mask)
    assert res['image'].shape == (1, 768, 768)  # ToTensorV2 produces (C, H, W)
    assert res['mask'].shape == (768, 768)
