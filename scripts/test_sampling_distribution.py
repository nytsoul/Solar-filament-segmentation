"""
Script to test and verify the patch sampling distribution of MAGFiLOPatchDataset.
Extracts patches and verifies the distribution across filament, disk, limb, and background regions.
"""
import os
import sys
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import MAGFiLOPatchDataset
from src.data.augmentations import get_training_augmentation_v2


def test_sampling():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'baseline_v2.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_csv = os.path.join(base_dir, config['dataset']['train_csv'])
    img_dir = os.path.join(base_dir, config['dataset']['image_dir'])
    json_path = os.path.join(base_dir, config['dataset']['json_path'])

    aug = get_training_augmentation_v2(config)

    dataset = MAGFiLOPatchDataset(
        csv_path=train_csv,
        img_dir=img_dir,
        json_path=json_path,
        config=config,
        transforms=aug
    )

    print(f"Total dataset length (patches per epoch): {len(dataset)}")
    print(f"Sampling targets: filament={dataset.p_filament:.2f}, disk={dataset.p_disk:.2f}, "
          f"limb={dataset.p_limb:.2f}, background={dataset.p_background:.2f}")

    num_samples = 40
    print(f"Testing sampling of {num_samples} patches...")

    dataset.reset_stats()
    for i in range(num_samples):
        img_patch, mask_patch = dataset[i]
        assert img_patch.shape == (1, 768, 768), f"Unexpected img shape: {img_patch.shape}"
        assert mask_patch.shape == (1, 768, 768), f"Unexpected mask shape: {mask_patch.shape}"

    stats = dataset.get_stats()
    print("\n--- Sampling Statistics Over 40 Patches ---")
    print(f"Counts: {stats['sample_counts']}")
    print(f"Strategy: filament={stats['pct_filament_strategy']:.1f}%, disk={stats['pct_disk_strategy']:.1f}%, "
          f"limb={stats['pct_limb_strategy']:.1f}%, background={stats['pct_background_strategy']:.1f}%")
    print(f"Patches with filament: {stats['pct_patches_with_filament']:.1f}%")
    print(f"Patches without filament (background/limb/disk): {stats['pct_patches_without_filament']:.1f}%")

    assert stats['sample_counts']['filament'] > 0, "No filament patches were sampled!"
    assert stats['sample_counts']['disk'] + stats['sample_counts']['limb'] + stats['sample_counts']['background'] > 0, "No background/disk/limb patches sampled!"
    print("\n[SUCCESS] Sampling distribution verification passed!")


if __name__ == '__main__':
    test_sampling()
