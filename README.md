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
