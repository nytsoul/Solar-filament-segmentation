# Solar Filament Segmentation: Training Signal Diagnosis

## Objective
Following the threshold diagnostic which revealed the V2 model was predicting a uniform ~0.5 probability everywhere, this diagnostic script (`scripts/diagnose_training_signal.py`) was created to analyze the internal training signal (losses and gradients) using the exact same dataset, patches, and configurations as the training loop.

## Findings

The diagnostic successfully captured 10 training patches and evaluated both the raw inputs and the model's internal state.

### 1. Data Integrity (PASS)
- **Masks are correctly normalized:** GT mask min is 0.0, max is 1.0, with exactly two unique values `[0.0, 1.0]`. BCE targets are completely valid.
- **Images are correctly normalized:** Image values range from roughly -1.0 to +1.0.
- **Class Imbalance Confirmed:** The average foreground percentage in the sampled training patches is **~1.1%**.

### 2. Loss & Gradient Computations (PASS)
- **No double-sigmoid bug:** `UnetPlusPlus` correctly outputs raw logits. `DiceLoss` applies `torch.sigmoid()` internally, and `BCEWithLogitsLoss` expects logits. The math is correctly implemented.
- **Gradients are flowing:** The gradient norm during `.backward()` is very stable (ranging from 1.9 to 2.8). There are no NaN or Inf tensors, and all 26,072,337 parameters are receiving non-zero gradients.

### 3. The Root Cause: Model Collapse (FAIL)
The diagnostic definitively proves the model has collapsed into a degenerate state:
- **Untrained Initialization:** A freshly initialized model outputs probabilities heavily skewed towards 1.0 (mean probability between 0.81 and 0.99).
- **Trained Model State:** Across *all* 10 diverse training patches (whether centered on a filament or empty space), the trained `best_baseline_v2.pth` model predicts a nearly identical mean probability of exactly **0.458 to 0.468**.

### Why did the model collapse?
This is a classic failure mode caused by **extreme class imbalance (99% background) combined with conflicting, unweighted losses**:

1. **The Washout Effect:** Because the untrained model starts by predicting `p ≈ 0.9` everywhere, the initial `BCEWithLogitsLoss` against a 99% empty mask generates a massive, uniform gradient pushing every single pixel towards 0. This huge early gradient washes out the spatial convolutional weights, reducing the network to a giant bias term.
2. **The Loss Equilibrium:** 
   - `BCEWithLogitsLoss` (which is averaged over all pixels) strongly penalizes the model for predicting foreground on the 99% background pixels, pushing the global probability down.
   - `DiceLoss` penalizes the model if it predicts 0 everywhere (which would make the intersection 0). It pushes the global probability up to maintain some overlap with the 1.1% target pixels.
   - The model finds a "lazy" local minimum exactly at `p ≈ 0.46`, where the downward pressure of the BCE loss perfectly balances the upward pressure of the Dice loss. The model sits in this equilibrium for the rest of training, completely ignoring the input image.

## Conclusion & Recommended Fixes
The `train_baseline_v2.py` pipeline is providing a destructive training signal because it fails to account for the 99-to-1 background-to-foreground imbalance.

To fix this, the training code must be updated with one or more of the following imbalance-handling techniques:
1. **Focal Loss / Weighted BCE:** Replace standard `BCEWithLogitsLoss` with Focal Loss, or add a `pos_weight` to the BCE to heavily down-weight the massive background gradients.
2. **Prior Initialization (Bias Tuning):** Initialize the final convolutional layer's bias to `-np.log((1 - 0.01) / 0.01) ≈ -4.6` so the untrained model starts by predicting `p ≈ 0.01` instead of `p ≈ 0.9`. This prevents the initial gradient shock.
3. **Tversky Loss:** Replace standard Dice loss with Tversky loss to independently control the penalties for false positives and false negatives.

These architectural changes will allow the network to learn spatial features rather than collapsing into a uniform constant.
