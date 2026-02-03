# MuSc Industrial GUI Architecture

This document describes the codebase structure, data flow, and architectural decisions for the MuSc Industrial GUI.

## Overview

MuSc Industrial GUI is a zero-shot anomaly detection system for industrial quality control. It combines:
- **MuSc Algorithm**: LNAMD + MSM + RsCIN from ICLR 2024
- **Vision Transformer Backbones**: DINO, DINOv2, CLIP, TIMM models
- **PyQt5 GUI**: Real-time camera, video, and batch processing

## Directory Structure

```
MuSc/
├── industrial_gui.py      # Main GUI application (entry point)
├── demo.py                # CLI interface for headless inference
├── config.yaml            # Runtime configuration
├── config.example.yaml    # Configuration template with documentation
│
├── models/                # Core ML models
│   ├── musc.py           # MuSc class - main anomaly detection logic
│   ├── backbone/         # Vision transformer implementations
│   │   ├── _backbones.py        # Backbone loader (DINO, DINOv2)
│   │   ├── vision_transformer.py # Base ViT implementation
│   │   ├── dino_vision_transformer.py
│   │   ├── dinov2/              # DINOv2 components
│   │   └── open_clip/           # OpenCLIP implementation
│   ├── modules/          # MuSc algorithm components
│   │   ├── _LNAMD.py    # Local Neighborhood Aggregation
│   │   ├── _MSM.py      # Mutual Scoring Module
│   │   └── _RsCIN.py    # Reference-based Classification
│   └── RsCIN_features/   # RsCIN support utilities
│
├── datasets/             # Dataset loaders
│   ├── mvtec.py         # MVTec AD dataset
│   ├── visa.py          # VisA dataset
│   └── btad.py          # BTAD dataset
│
├── utils/                # Utility functions
│   ├── load_config.py   # YAML configuration loader
│   └── metrics.py       # Evaluation metrics (AUROC, F1, AuPRO)
│
├── soft_memory/          # Reference images (runtime)
├── sample_data/          # Example images for testing
├── output_gui/           # Inference output directory
├── tests/                # Test suite
│   ├── conftest.py      # Pytest fixtures
│   ├── test_smoke.py    # Basic functionality tests
│   └── test_modules.py  # Unit tests for LNAMD/MSM
│
├── assets/               # Documentation images
├── .github/              # GitHub templates and CI
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/ci.yml
│   └── pull_request_template.md
│
├── pyproject.toml        # Package metadata and dependencies
├── requirements.txt      # Alternative dependency list
├── README.md             # User documentation
├── CONTRIBUTING.md       # Contribution guidelines
├── CHANGELOG.md          # Release history
└── ARCHITECTURE.md       # This file
```

## Core Components

### 1. MuSc Class (`models/musc.py`)

The central class orchestrating anomaly detection.

```
MuSc
├── __init__(cfg)         # Initialize with config, load backbone
├── load_backbone()       # Load DINO/DINOv2/CLIP/TIMM model
├── load_datasets()       # Create dataset/dataloader for benchmarks
├── infer_on_images()     # Real-time inference (GUI/CLI)
├── make_category_data()  # Full dataset evaluation
├── visualization()       # Generate heatmap outputs
└── main()                # Run full benchmark evaluation
```

**Key Methods:**
- `infer_on_images(images)`: Takes list of tensors, returns anomaly maps
- `make_category_data(category)`: Evaluates entire dataset category

### 2. LNAMD Module (`models/modules/_LNAMD.py`)

**Local Neighborhood Aggregation with Multi-scale Distance**

Aggregates patch features from vision transformers to create robust representations.

```python
LNAMD(device, r, feature_dim, feature_layer)
├── _embed(patch_tokens)  # Aggregate features with radius r
└── Returns: [B, num_patches, num_layers, feature_dim]
```

**Parameters:**
- `r`: Aggregation radius (typically 1)
- `feature_layer`: Which transformer layers to use

### 3. MSM Module (`models/modules/_MSM.py`)

**Mutual Scoring Module**

Computes anomaly scores by comparing each image against all others in the batch.

```python
MSM(Z, device, topmin_min, topmin_max)
├── Input: Z = [N, L, C] (N images, L patches, C features)
└── Output: [N, L] anomaly scores per patch
```

**Algorithm:**
1. Compute pairwise cosine similarity between all patches
2. For each patch, find minimum distance to other images
3. Average across a range of "top-min" values for robustness

### 4. RsCIN Module (`models/modules/_RsCIN.py`)

**Reference-based Score Calibration for Image-level Normalization**

Refines image-level anomaly scores using feature-space clustering.

```python
RsCIN(ac_score, class_tokens, k_list)
├── Input: per-image anomaly scores and global features
└── Output: calibrated anomaly scores
```

## Data Flow

### GUI Inference Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Camera/   │────▶│ InferenceThread │────▶│   Results   │
│   Video     │     │  (QThread)    │     │   Sidebar   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MuSc.infer_ │
                    │  on_images() │
                    └──────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Backbone  │  │   LNAMD    │  │    MSM     │
    │  Features  │  │ Aggregation│  │  Scoring   │
    └────────────┘  └────────────┘  └────────────┘
```

### Detailed Processing Pipeline

1. **Input Preprocessing**
   - Resize image to model input size (e.g., 224x224)
   - Normalize to [0, 1] range
   - Convert to tensor [B, 3, H, W]

2. **Feature Extraction** (Backbone)
   - Extract patch tokens from transformer layers
   - Get [B, num_patches+1, feature_dim] per layer
   - CLS token for global features, patch tokens for localization

3. **LNAMD Aggregation**
   - Aggregate neighboring patch features
   - Normalize feature vectors
   - Output: [B, num_patches, num_layers, feature_dim]

4. **MSM Scoring**
   - Compute mutual scoring between all images
   - For each patch, find anomaly score
   - Output: [B, num_patches] scores

5. **Post-processing**
   - Reshape to spatial dimensions [B, H', W']
   - Interpolate to original image size
   - Normalize to [0, 1] range

6. **RsCIN** (optional, for image-level scores)
   - Calibrate scores using global features
   - Improves image-level classification accuracy

## Configuration System

### config.yaml Structure

```yaml
datasets:
  img_resize: 224           # Input resolution
  dataset_name: mvtec_ad    # For benchmark evaluation
  class_name: ALL           # Category filter
  divide_num: 1             # Dataset splitting
  data_path: ./data/        # Dataset root

device: 0                   # GPU index or 'cpu'

models:
  backbone_name: dinov2_vitb14  # Vision model
  batch_size: 1
  feature_layers: [11]      # Which transformer layers
  pretrained: openai        # Pretrained weights source
  r_list: [1]               # LNAMD aggregation radii

testing:
  output_dir: output_gui
  vis: false                # Save visualizations
  vis_type: single_norm
  save_excel: false

thresholds:
  image_threshold: 9.0      # Detection sensitivity (1-10)
  overlay_threshold: 3.5    # Heatmap intensity (1-10)
```

### Threshold Interpretation

Thresholds are displayed as 1.0-10.0 in the GUI but used as 0.1-1.0 internally:
- `image_threshold: 9.0` → 0.9 internal threshold
- Score > threshold → flagged as anomaly

## Backbone Models

### Supported Architectures

| Model | Type | Patch Size | Input Size | Speed | Accuracy |
|-------|------|------------|------------|-------|----------|
| dinov2_vitb14 | DINOv2 | 14 | 224/518 | Medium | High |
| dinov2_vitl14 | DINOv2 | 14 | 224/518 | Slow | Highest |
| dino_vitbase16 | DINO | 16 | 224 | Medium | High |
| ViT-B-16 | CLIP | 16 | 224 | Medium | High |
| ViT-L-14 | CLIP | 14 | 224 | Slow | Highest |
| vit_tiny_* | TIMM | 16 | 224 | Fast | Medium |

### Backbone Loading Logic

```python
if model_name.startswith("dino_") or model_name.startswith("dinov2_"):
    # Load via _backbones.load()
elif model_name.startswith("vit_"):
    # Load via timm.create_model()
else:
    # Load via open_clip.create_model_and_transforms()
```

## GUI Architecture

### Main Window Components

```
MainWindow
├── CentralWidget (Tabs)
│   ├── Tab 1: Live Camera
│   │   ├── Camera preview
│   │   ├── Duration/FPS controls
│   │   └── Start/Stop buttons
│   ├── Tab 2: Load Video
│   │   ├── Video file browser
│   │   ├── Folder browser
│   │   └── Processing controls
│   └── Tab 3: (reserved)
│
├── Left Dock: Configuration
│   └── Config button
│
└── Right Dock: Results
    ├── Anomalous items
    ├── Saved items
    └── Soft Memory
```

### Threading Model

- **Main Thread**: GUI updates, user interaction
- **InferenceThread (QThread)**: ML inference
  - Emits `progressChanged(int)` during processing
  - Emits `inferenceFinished(frames, maps, score, time)` on completion

### Signal/Slot Connections

```python
InferenceThread.inferenceFinished.connect(MainWindow.handle_inference_result)
InferenceThread.progressChanged.connect(ProgressBar.setValue)
```

## Testing Strategy

### Test Categories

1. **Smoke Tests** (`test_smoke.py`)
   - Import verification
   - Configuration loading
   - Basic tensor operations
   - No GPU required

2. **Module Tests** (`test_modules.py`)
   - LNAMD with synthetic data
   - MSM with synthetic data
   - Shape/dimension validation

3. **Integration Tests** (GPU required)
   - Full inference pipeline
   - Model loading
   - End-to-end processing

### Running Tests

```bash
# All tests (CPU only)
pytest tests/ -v -k "not slow"

# With GPU tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=models --cov-report=html
```

## Performance Considerations

### Memory Management

- Inference uses `torch.no_grad()` and `torch.cuda.amp.autocast()`
- `torch.cuda.empty_cache()` called after large operations
- Soft memory limited to ~50 images to prevent OOM

### Optimization Opportunities

1. **Batch Processing**: Increase batch_size for throughput
2. **Model Selection**: Smaller models for real-time
3. **Image Resolution**: Lower resolution = faster inference
4. **xFormers**: Optional accelerated attention

## Extension Points

### Adding New Backbones

1. Add loading logic to `models/backbone/_backbones.py`
2. Handle feature extraction in `MuSc.infer_on_images()`
3. Add to model list in `ConfigDialog`
4. Update image size options

### Adding Export Formats

1. Add export function in `industrial_gui.py`
2. Connect to UI button/menu
3. Handle in results sidebar

### Custom Datasets

1. Create dataset class in `datasets/`
2. Inherit from appropriate base class
3. Register in `MuSc.load_datasets()`

## Design Decisions

### Why Zero-Shot?

- No labeled defect data required
- Works immediately on new products
- Handles unseen defect types
- Lower barrier to adoption

### Why Multiple Backbones?

- Different speed/accuracy tradeoffs
- Some work better on specific domains
- Allows hardware flexibility

### Why PyQt5?

- Cross-platform support
- Native look and feel
- Good threading support for responsive UI
- Mature ecosystem

## References

- **MuSc Paper**: [arXiv:2401.16753](https://arxiv.org/abs/2401.16753)
- **DINOv2**: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- **CLIP**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
