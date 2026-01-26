# Universal Industrial Anomaly Detection GUI

A powerful, zero-shot anomaly detection system for industrial and manufacturing quality control. Detect defects, anomalies, and irregularities in real-time without training on defect examples.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## What is This?

This is a graphical interface for **universal anomaly detection** in any industrial or manufacturing setting. Simply show the system what "normal" looks like, and it will automatically flag anything unusual - scratches, dents, missing components, color variations, alignment issues, and more.

**No training required. No labeled defect examples needed. Just point and detect.**

## Key Features

- **Zero-Shot Detection**: Detects anomalies without training on defect examples
- **Universal Application**: Works across any industry - electronics, textiles, metal fabrication, pharmaceuticals, food production, etc.
- **Real-Time Inspection**: Live camera feed processing for production line monitoring
- **Video & Batch Processing**: Analyze recorded video or folders of images
- **Visual Anomaly Maps**: See exactly where defects are located with heatmap overlays
- **Customizable Sensitivity**: Adjust detection thresholds for your specific use case
- **Reference Memory System**: Store "golden samples" to improve accuracy
- **Multiple AI Backbones**: Choose from various pre-trained vision models for optimal performance

## Use Cases

- **Quality Control**: Manufacturing defect detection on production lines
- **Assembly Verification**: Ensure all components are present and correctly positioned
- **Surface Inspection**: Detect scratches, dents, cracks, or discoloration
- **PCB Inspection**: Identify soldering defects, missing components, or trace issues
- **Textile Inspection**: Find tears, stains, or weaving defects in fabrics
- **Food Safety**: Spot contamination or packaging defects
- **Pharmaceutical QC**: Detect pill defects, packaging issues, or label problems
- **Metal Fabrication**: Identify welding defects, surface irregularities, or dimensional issues

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time performance)
- Webcam (optional, for live inspection)

### Installation

1. **Install PyTorch with CUDA support** (for GPU acceleration):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower)
pip install torch torchvision torchaudio
```

Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for other CUDA versions.

2. **Install the GUI package + dependencies** (after installing PyTorch/TorchVision):

```bash
pip install -e .
# or, if you prefer the classic requirements workflow:
pip install -r requirements.txt
```

> Optional extras (install only if you need them):
> - `pip install -e .[accelerated]` -- pulls in xFormers for faster attention layers.
> - `pip install -e .[distributed]` -- installs Horovod for large multi-node runs.

3. **Copy the configuration template**:

```bash
cp config.example.yaml config.yaml
```

4. **Run the application**:

```bash
musc-gui
# or
python industrial_gui.py
```

## Dependency Management

- The repository ships a `pyproject.toml`, so `pip install -e .` (or `pip install .`) resolves the runtime dependencies automatically.
- Install GPU-specific `torch` and `torchvision` wheels from https://pytorch.org/get-started/locally/ **before** running `pip install -e .` if you need a custom CUDA build.
- `pip install -r requirements.txt` is still available for simple/air-gapped environments.
- Optional extras:
  - `accelerated` &rarr; pulls in `xformers` for memory-efficient attention.
  - `distributed` &rarr; installs `horovod[pytorch]` for large multi-node training.
  - `dev` &rarr; formatter + lint/test helpers (`black`, `ruff`, `pytest`).

## How It Works

The system uses state-of-the-art vision transformers (CLIP, DINO, DINOv2) to learn what "normal" looks like from your reference images. When processing new images, it compares them against this learned normal appearance and flags anything that deviates significantly.

**The Technology**: Built on the MuSc (Mutual Scoring) algorithm from ICLR 2024, which achieves 97.8% accuracy on standard industrial anomaly benchmarks without seeing a single defect example during training.

## Basic Usage

### Live Camera Inspection

1. Launch the GUI: `python industrial_gui.py`
2. Click **Configuration** to select your AI model and set detection sensitivity
3. Position your camera to view the inspection area
4. Set **Duration** (how many seconds to collect frames) and **Target FPS**
5. Click **Start** to begin real-time detection
6. Detected anomalies appear automatically in the sidebar with scores and visual overlays

### Processing Videos

1. Switch to the **Load Video** tab
2. Click **Browse Video** and select your video file
3. Set your collection parameters (Duration, Target FPS)
4. Click **Start Recording** to process frames
5. Review detected anomalies in the **Anomalous** sidebar

### Batch Processing Images

1. Switch to the **Load Video** tab
2. Click **Browse Folder** and select a folder of images
3. Click **Start Folder Inference** to process all images
4. Review results in the **Anomalous** sidebar

## Configuration

Edit `config.yaml` to customize the system:

```yaml
datasets:
  img_resize: 224  # Image size (224, 256, 384, 512)

device: 0  # GPU device (0, 1, etc.) or 'cpu'

models:
  backbone_name: dinov2_vitb14  # AI model to use
  batch_size: 1

thresholds:
  image_threshold: 9.0   # Detection sensitivity (1.0-10.0)
  overlay_threshold: 3.5  # Visualization intensity
```

### Available AI Models

Choose the backbone that best fits your speed/accuracy needs:

**Fast (for real-time on modest hardware)**:
- `vit_tiny_patch16_224.augreg_in21k`
- `vit_small_patch32_224.augreg_in21k`
- `dino_deitsmall16`

**Balanced (recommended)**:
- `dinov2_vitb14` ← Default, good balance
- `ViT-B-16`
- `dino_vitbase16`

**High Accuracy (slower, needs powerful GPU)**:
- `dinov2_vitl14`
- `ViT-L-14`
- `google/siglip-so400m-patch14-384`

### Threshold Settings

**Image Threshold** (1.0-10.0): Controls what gets flagged as anomalous
- **High (9.0-10.0)**: Fewer false alarms, only obvious defects
- **Medium (7.0-9.0)**: Balanced for general use
- **Low (1.0-7.0)**: Very sensitive, catches subtle issues but may have false positives

**Overlay Threshold** (1.0-10.0): Controls visualization intensity
- Only affects what you see in the heatmap overlay
- Does not change what gets detected
- Recommended: 3.0-5.0

## Advanced Features

### Soft Memory (Reference Images)

The "Soft Memory" system lets you store 10-50 reference images of known good products. These are included in every inference batch to improve accuracy and reduce false positives.

**Best Practices**:
- Use 10-50 high-quality reference images
- Include normal variations (lighting, angle, positioning)
- All images should be from the same product/setup
- Update periodically as your product or process changes

### Sidebar Panels

**Anomalous**: View all detected defects with:
- Anomaly scores
- Visual overlay toggle
- Save to disk option
- Add to soft memory
- Delete false positives

**Saved**: Manually captured frames for later review

**Soft Memory**: Your reference library of "normal" examples

## Performance Tips

- **GPU is critical**: 10-50x faster than CPU for real-time use
- **Image size**: Smaller (224x224) is faster; larger (512x512) catches finer details
- **Model selection**: Tiny models process 3-5x faster than large models
- **Soft memory**: Keep under 50 images for best performance
- **Batch size**: Can increase for faster batch processing (if GPU memory allows)

## Troubleshooting

### "CUDA Out of Memory"
- Reduce image size in config (512→384→256→224)
- Use a smaller AI model (try `vit_tiny_patch16_224.augreg_in21k`)
- Switch to CPU mode: `device: 'cpu'` in config.yaml

### Slow Performance
- Ensure you have a CUDA-enabled GPU and installed PyTorch with CUDA support
- Use a smaller model (`vit_tiny` instead of `vit_large`)
- Reduce image resolution
- Clear soft memory if it has too many images

### Too Many False Positives
- Increase image threshold (try 9.5 or 10.0)
- Add more reference images to soft memory showing normal variations
- Ensure lighting and positioning are consistent

### Missing Real Defects
- Lower image threshold (try 7.0 or 8.0)
- Use a larger, more accurate model
- Increase image resolution
- Ensure reference images truly represent "normal"

## Technical Details

Built on the MuSc (Mutual Scoring of Unlabeled Images) algorithm published at ICLR 2024:

**Paper**: ["MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images"](https://arxiv.org/pdf/2401.16753.pdf)

**Performance on Standard Benchmarks**:
- MVTec AD: 97.8% classification accuracy, 97.3% segmentation AUROC
- VisA: 92.8% classification accuracy, 98.8% segmentation AUROC

**How it works**: Uses vision transformer features to compute mutual scoring between test images, identifying outliers without requiring defect examples during training.

## Citation

If you use this tool in research or commercial applications, please cite the original MuSc paper:

```bibtex
@inproceedings{Li2024MuSc,
  title={MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images},
  author={Li, Xurui and Huang, Ziming and Xue, Feng and Zhou, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

Original MuSc algorithm by Xurui Li, Ziming Huang, Feng Xue, and Yu Zhou.
GUI implementation and industrial application focus by Dylan Martin.

## System Requirements

**Minimum**:
- Python 3.8+
- 8GB RAM
- CPU with AVX support

**Recommended**:
- Python 3.8+
- 16GB+ RAM
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- CUDA 11.8 or 12.1

**For Production Line Real-Time**:
- NVIDIA RTX 3080/4070 or better
- 16GB+ system RAM
- SSD storage
- Industrial camera (recommended: GigE Vision or USB3 with global shutter)

## Support & Contributing

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: See `README_GUI.md` for detailed GUI documentation
- **Original Research**: See the [original MuSc repository](https://github.com/xrli-U/MuSc) for research code

## Acknowledgments

This GUI builds upon the groundbreaking work of the MuSc team at Huazhong University of Science and Technology. The core anomaly detection algorithm is from their ICLR 2024 publication.


