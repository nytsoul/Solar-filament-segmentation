"""
Solar disk detection and region classification for Hα images.

Provides utilities to:
- Detect the solar disk boundary using Otsu thresholding + morphology.
- Classify image regions as disk, limb, or background.
- Generate region masks for patch sampling.
"""
import numpy as np
import cv2


def detect_solar_disk(image, min_radius_frac=0.3, max_radius_frac=0.6):
    """
    Detect the solar disk in an Hα grayscale image.

    Uses Otsu thresholding followed by morphological closing to get a clean
    disk mask, then fits a minimum enclosing circle.

    Args:
        image: np.ndarray of shape (H, W), uint8 grayscale.
        min_radius_frac: Minimum expected radius as fraction of image size.
        max_radius_frac: Maximum expected radius as fraction of image size.

    Returns:
        dict with keys:
            'disk_mask': binary mask (H, W), uint8, 1=on-disk, 0=off-disk
            'center': (cy, cx) of disk center
            'radius': estimated radius in pixels
            'valid': bool, whether detection succeeded
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)

    # Otsu threshold to separate disk from background
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill small holes inside the disk
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes: find contours, keep the largest one
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # Fallback: assume disk is centered with radius = 0.45 * min_dim
        return _fallback_disk(h, w)

    # Find the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Sanity check: the disk should be a significant portion of the image
    min_area = np.pi * (min_dim * min_radius_frac) ** 2
    if area < min_area:
        return _fallback_disk(h, w)

    # Fit minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    # Sanity check radius
    if radius < min_dim * min_radius_frac or radius > min_dim * max_radius_frac:
        return _fallback_disk(h, w)

    # Create clean circular mask
    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (cx, cy), radius, 1, -1)

    return {
        'disk_mask': disk_mask,
        'center': (cy, cx),
        'radius': radius,
        'valid': True
    }


def _fallback_disk(h, w):
    """Fallback: assume disk is centered with radius = 45% of min dimension."""
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.45)
    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (cx, cy), radius, 1, -1)
    return {
        'disk_mask': disk_mask,
        'center': (cy, cx),
        'radius': radius,
        'valid': False  # flagged as fallback
    }


def get_limb_mask(disk_mask, center, radius, limb_width=50):
    """
    Create an annular mask for the solar limb region.

    Args:
        disk_mask: binary disk mask (H, W).
        center: (cy, cx) of disk center.
        radius: disk radius in pixels.
        limb_width: width of the limb annulus in pixels.

    Returns:
        np.ndarray: binary mask (H, W), 1=limb region.
    """
    h, w = disk_mask.shape
    cy, cx = center
    limb_mask = np.zeros((h, w), dtype=np.uint8)

    # Outer ring
    cv2.circle(limb_mask, (cx, cy), radius + limb_width // 2, 1, -1)

    # Subtract inner ring
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(inner, (cx, cy), max(1, radius - limb_width // 2), 1, -1)

    limb_mask = limb_mask - inner
    limb_mask = np.clip(limb_mask, 0, 1)
    return limb_mask


def get_background_mask(disk_mask):
    """
    Create a mask for the off-disk background region.

    Args:
        disk_mask: binary disk mask (H, W).

    Returns:
        np.ndarray: binary mask (H, W), 1=background.
    """
    return (1 - disk_mask).astype(np.uint8)


def get_region_masks(image, limb_width=50):
    """
    Compute all region masks for a given image.

    Returns:
        dict with keys: 'disk_info', 'disk_mask', 'limb_mask', 'background_mask'
    """
    disk_info = detect_solar_disk(image)
    disk_mask = disk_info['disk_mask']
    limb_mask = get_limb_mask(disk_mask, disk_info['center'], disk_info['radius'], limb_width)
    background_mask = get_background_mask(disk_mask)

    return {
        'disk_info': disk_info,
        'disk_mask': disk_mask,
        'limb_mask': limb_mask,
        'background_mask': background_mask
    }


def sample_patch_location(region_mask, patch_size, max_attempts=100):
    """
    Sample a random (y, x) top-left coordinate for a patch within a region.

    Args:
        region_mask: binary mask (H, W) of the target region.
        patch_size: size of the square patch.
        max_attempts: maximum random attempts before falling back to random crop.

    Returns:
        (y, x) top-left coordinate, or None if no valid location found.
    """
    h, w = region_mask.shape

    if h < patch_size or w < patch_size:
        return None

    # Find valid region pixels
    ys, xs = np.where(region_mask > 0)
    if len(ys) == 0:
        return None

    for _ in range(max_attempts):
        idx = np.random.randint(len(ys))
        # Center the patch on the sampled point
        cy, cx = ys[idx], xs[idx]
        y = cy - patch_size // 2
        x = cx - patch_size // 2

        # Clamp to image bounds
        y = max(0, min(y, h - patch_size))
        x = max(0, min(x, w - patch_size))

        # Check that at least some of the patch overlaps the region
        patch_region = region_mask[y:y + patch_size, x:x + patch_size]
        if patch_region.sum() > 0:
            return (y, x)

    return None
