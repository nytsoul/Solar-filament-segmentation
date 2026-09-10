import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(config):
    """Original v1 training augmentation with CropNonEmptyMaskIfExists."""
    aug_cfg = config.get('augmentations', {})
    patch_size = config['dataset'].get('patch_size', 768)
    
    return A.Compose([
        A.OneOf([
            A.CropNonEmptyMaskIfExists(height=patch_size, width=patch_size, p=1.0),
            A.RandomCrop(height=patch_size, width=patch_size, p=1.0)
        ], p=1.0),
        A.HorizontalFlip(p=aug_cfg.get('p_hflip', 0.5)),
        A.VerticalFlip(p=aug_cfg.get('p_vflip', 0.5)),
        A.RandomRotate90(p=aug_cfg.get('p_rot90', 0.5)),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('brightness_limit', 0.1),
            contrast_limit=aug_cfg.get('contrast_limit', 0.1),
            p=aug_cfg.get('p_brightness_contrast', 0.3)
        ),
        A.RandomGamma(
            gamma_limit=aug_cfg.get('gamma_limit', [90, 110]),
            p=aug_cfg.get('p_gamma', 0.3)
        ),
        A.GaussNoise(p=aug_cfg.get('p_gauss_noise', 0.2)),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_training_augmentation_v2(config):
    """
    V2 training augmentation: geometric + photometric transforms WITHOUT cropping.
    
    Cropping is handled externally by the MAGFiLOPatchDataset sampler.
    This transform expects a pre-cropped (patch_size, patch_size) image+mask.
    """
    aug_cfg = config.get('augmentations', {})
    
    return A.Compose([
        A.HorizontalFlip(p=aug_cfg.get('p_hflip', 0.5)),
        A.VerticalFlip(p=aug_cfg.get('p_vflip', 0.5)),
        A.RandomRotate90(p=aug_cfg.get('p_rot90', 0.5)),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('brightness_limit', 0.1),
            contrast_limit=aug_cfg.get('contrast_limit', 0.1),
            p=aug_cfg.get('p_brightness_contrast', 0.3)
        ),
        A.RandomGamma(
            gamma_limit=aug_cfg.get('gamma_limit', [90, 110]),
            p=aug_cfg.get('p_gamma', 0.3)
        ),
        A.GaussNoise(p=aug_cfg.get('p_gauss_noise', 0.2)),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_validation_augmentation(config):
    """Original v1 validation augmentation with CenterCrop (kept for backward compat)."""
    patch_size = config['dataset'].get('patch_size', 768)
    return A.Compose([
        A.CenterCrop(height=patch_size, width=patch_size),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])

def get_inference_augmentation():
    """Normalize + ToTensor for full-image inference. No cropping."""
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])
