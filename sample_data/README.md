# Sample Data

This directory contains sample images for testing MuSc Industrial GUI.

## Quick Start

### Option 1: Download Sample Images (Recommended)

Download a few images from the MVTec AD dataset:

1. Visit https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Download one or more categories (bottle, screw, metal_nut recommended)
3. Extract and copy images to the appropriate folders:

```
sample_data/
├── bottle/
│   ├── normal/     <- Copy 5-10 good images here
│   └── anomaly/    <- Copy 5-10 defective images here
├── screw/
│   ├── normal/
│   └── anomaly/
└── metal_nut/
    ├── normal/
    └── anomaly/
```

### Option 2: Use Your Own Images

You can use any images for testing:

1. Create a folder for your category (e.g., `sample_data/my_product/`)
2. Add subfolders `normal/` and `anomaly/`
3. Place reference (good) images in `normal/`
4. Place test images in `anomaly/` (or any images you want to inspect)

## Running the Demo

Once you have images in place:

```bash
# CLI demo
python demo.py --input sample_data/bottle/anomaly/000.png --reference sample_data/bottle/normal/

# Or launch the GUI
musc-gui
```

## Directory Structure

```
sample_data/
├── README.md           <- This file
├── LICENSE.md          <- MVTec AD license information
├── bottle/
│   ├── normal/         <- Good/reference images
│   └── anomaly/        <- Defective/test images
├── screw/
│   ├── normal/
│   └── anomaly/
└── metal_nut/
    ├── normal/
    └── anomaly/
```

## Image Requirements

- **Format**: PNG, JPG, JPEG, BMP
- **Size**: Any size (will be resized to model input size)
- **Color**: RGB preferred, grayscale supported
- **Recommended**: At least 5 reference (normal) images per category

## Categories

| Category | Description | Common Defects |
|----------|-------------|----------------|
| bottle | Glass bottles | Contamination, broken parts |
| screw | Metal screws | Thread damage, scratches |
| metal_nut | Hexagonal nuts | Bent, scratched, flipped |

## Notes

- Reference images should represent "normal" or "good" products
- The more diverse your reference images, the better the detection
- For best results, ensure consistent lighting and camera angle
- See LICENSE.md for MVTec AD dataset usage terms
