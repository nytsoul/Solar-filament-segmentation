# MAGFiLO Dataset Details

## Overview
The MAGFiLO dataset is structured for a Solar Filament Segmentation challenge. It provides H-alpha images and their corresponding filament annotations in a COCO-like format.

## Directory Structure
- `MAGFiLO_1.0_Kaggle_2026/`
  - `train/`
    - `train_images/` - Directory containing the training images.
    - `MAGFiLO_1.0_Annotations_kaggle2026_train.json` - COCO-format JSON with ground truth annotations.
  - `test/`
    - `test_images/` - Directory containing the test images.

## Image Details
- **Format:** `.jpeg` images.
- **Dimensions:** 2048 x 2048.
- **Color Channels:** Grayscale (1 channel), read as `uint8`.
- **Pixel Value Range:** Approximately `[0, 230]`.

## Annotation Format
The annotation JSON follows the MS-COCO format with an additional custom `spine` field specifically designed for solar filaments.

### Top Level Keys
- `info`: Dataset metadata.
- `licenses`: Licensing information.
- `categories`: Contains 4 filament categories (e.g. "Left").
- `images`: Image records matching files in `train_images`.
- `annotations`: Individual filament instance annotations.

### Individual Filament Instance Representation
Multiple filaments can exist within the same image. Each instance is an element inside the `annotations` list with the following structure:
- **`id`**: Unique string ID for the annotation.
- **`image_id`**: String matching an image's `id`.
- **`category_id`**: Integer identifying the filament's category.
- **`segmentation`**: A list of lists of floats representing the polygon vertices `[[x1, y1, x2, y2, ...]]`.
- **`bbox`**: A list `[x, y, width, height]` for the bounding box.
- **`area`**: Area of the polygon in pixels.
- **`iscrowd`**: Binary flag indicating if the segmentation is a single object or crowd.
- **`spine`**: (MAGFiLO specific) A list of floats `[x1, y1, x2, y2, ...]` representing the medial axis/spine of the filament.

## Key Statistics (Train)
- Images in directory: 707
- Images in JSON metadata: 1154
- Total Annotated Instances: 8199
- Filaments per image vary significantly (multiple filaments are very common).
