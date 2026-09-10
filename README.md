# Solar Filament Segmentation Challenge 2026

## Environment Setup
To set up your local environment, install the exact requirements:
```bash
pip install -r requirements.txt
```
Note: Ensure you have an appropriate PyTorch version installed for your hardware (e.g., CUDA 11.8 or 12.1 if running locally with a GPU).

## Kaggle GPU Setup
To train the model on Kaggle:
1. Create a new Kaggle notebook with GPU P100 or T4x2 enabled.
2. Add the Kaggle dataset (`MAGFiLO 1.0 Kaggle 2026`) to the notebook.
3. Upload this repository code into the notebook or clone it.
4. Open and run `notebooks/03_baseline_gpu_training.ipynb`.

## Dataset Location
The project expects the dataset to be in `MAGFiLO_1.0_Kaggle_2026/train/` for local development. For Kaggle, the notebook automatically locates the dataset under `/kaggle/input/`. 

## Configuration
Configuration is managed via `configs/baseline.yaml`. You can configure:
- `patch_size`, `overlap`, `batch_size`
- `learning_rate`, `weight_decay`, `epochs`, `gradient_clip_val`
- Model architecture and ResNet encoder

## Training Command
To train the baseline locally:
```bash
python scripts/train_baseline.py
```
This script automatically detects CUDA, uses mixed precision (AMP), and records history to `outputs/training_history.csv`.

## Checkpoint Location
During training, the best checkpoint (based on validation Dice) is saved to `checkpoints/best_baseline.pth`. The final epoch is saved to `checkpoints/last_baseline.pth`. On Kaggle, these are saved directly to `/kaggle/working/checkpoints/`.

## Validation Procedure
Validation during training relies on deterministic sampling. To evaluate a trained checkpoint fully via sliding window over full 2048x2048 images:
```bash
python scripts/evaluate_baseline.py
```

## Reproducing the Baseline
1. Ensure the dataset splits are generated using `python src/data/split.py` (grouping by date to prevent temporal leakage).
2. Train the model using `python scripts/train_baseline.py`.
3. Evaluate metrics and generate mask overlays using `python scripts/evaluate_baseline.py`.

---

## Full-Image Baseline v2 (Controlled Experiment)

### Motivation & Findings
- **Audit Discovery**: Baseline v1 achieved validation Dice ~0.6947 because validation evaluated only `CenterCrop(768, 768)` (center 14% of each image). When evaluated on complete 2048×2048 images with sliding-window inference, true performance was Dice 0.0948, IoU 0.0530, PQ 0.0005 due to massive false positive rates along the solar limb and outer space background.
- **Controlled Experiment**: Keeps the exact same SMP `UnetPlusPlus` + `resnet34` architecture to isolate the impact of training distribution and validation protocol.

### Key Enhancements
1. **Region-Aware Patch Sampling (`MAGFiLOPatchDataset`)**:
   - Instead of solely sampling filament-containing crops (`CropNonEmptyMaskIfExists`), patches are drawn from 4 configurable regions:
     - `filament`: 50% probability (centered on ground-truth filament instances with jitter)
     - `disk`: 20% probability (within the solar disk, including quiet Sun / plages)
     - `limb`: 15% probability (solar limb transition zone)
     - `background`: 15% probability (off-disk space background)
   - Automatically detects solar disk center and radius using Otsu thresholding and morphological closing.
2. **Full-Image Sliding-Window Validation (`MAGFiLOFullImageDataset`)**:
   - Validation reconstructs and evaluates full 2048×2048 images using the exact sliding-window inference pipeline (patch size 768, overlap 0.25).
   - Reports full-image Dice and IoU as primary validation metrics (no CenterCrop).
3. **Diagnostics & Visualizations**:
   - Tracks sampling distribution statistics across epochs.
   - Plots 4-panel training curves (`training_curves_v2.png`): Training Loss, Full-Image Dice, Full-Image IoU, and Patch Sampling Distribution.
   - Automatically generates side-by-side full-image prediction visualizations under `outputs/full_image_baseline_v2/val_visualizations/`.

### Configuration
Managed via `configs/baseline_v2.yaml`:
- **Patch size**: 768 × 768
- **Patches per image**: 4 (2,260 patches/epoch)
- **Batch size**: 4
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`, CosineAnnealingLR)
- **Loss**: BCEWithLogitsLoss + DiceLoss

### Commands
```bash
# Run 2-epoch smoke test locally (with batch and validation limits)
python scripts/train_baseline_v2.py --config configs/baseline_v2.yaml --epochs 2 --limit_batches 5 --val_limit 3

# Verify patch sampling distribution
python scripts/test_sampling_distribution.py

# Run unit tests
python -m pytest tests/test_pipeline_v2.py

# Full training on Kaggle GPU (multi-GPU / AMP enabled)
python scripts/train_baseline_v2.py --config configs/baseline_v2.yaml
```
Outputs and checkpoints are saved to:
- Checkpoints: `checkpoints/best_baseline_v2.pth`
- Visualizations: `outputs/full_image_baseline_v2/val_visualizations/`
- Curves: `outputs/full_image_baseline_v2/training_curves_v2.png`
- Metrics: `outputs/full_image_baseline_v2/final_metrics_v2.json`
