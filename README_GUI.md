# MuSc Industrial GUI

A graphical user interface for the MuSc (Multi-Scale) anomaly detection model, designed for industrial inspection applications.

## Features

- **Live Camera Feed**: Real-time anomaly detection from webcam or camera input
- **Video Processing**: Load and analyze video files or image folders
- **Anomaly Detection**: Automatic detection and visualization of anomalies with customizable thresholds
- **Soft Memory**: Store reference images for improved detection accuracy
- **Batch Inference**: Process multiple frames efficiently
- **Interactive Controls**: Configure model parameters, thresholds, and processing settings on the fly

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- Webcam (optional, for live camera features)

## Installation

### 1. Install PyTorch

Install PyTorch with CUDA support (recommended for GPU acceleration):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio
```

Visit [PyTorch's website](https://pytorch.org/get-started/locally/) for specific installation commands for your system.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

The GUI will automatically search for a configuration file in the following locations:
1. `config.yaml` in the current directory
2. `configs/musc.yaml` in the configs subdirectory
3. Relative to the script location

You can create a `config.yaml` file based on the template below:

```yaml
datasets:
  class_name: ALL
  data_path: ./data/  # Path to your dataset (relative to project root)
  dataset_name: mvtec_ad
  divide_num: 1
  img_resize: 224

device: 0  # GPU device index (0, 1, etc.) or 'cpu'

models:
  backbone_name: vit_tiny_patch16_224.augreg_in21k
  batch_size: 1
  feature_layers:
  - 11
  pretrained: openai
  r_list:
  - 1

testing:
  output_dir: output_gui
  save_excel: false
  vis: false
  vis_type: single_norm

thresholds:
  image_threshold: 0.9  # Threshold for flagging anomalous images (0.0-1.0)
  overlay_threshold: 0.5  # Threshold for overlay visualization (0.0-1.0)
```

## Usage

### Running the GUI

```bash
python industrial_gui.py
```

### Interface Overview

The application has two main tabs:

#### Live Camera Tab
- **Start**: Begin collecting frames for inference
- **Stop**: Stop the current recording/inference
- **Capture**: Save a single frame to the "Saved" sidebar
- **Configuration**: Open settings dialog to change model and thresholds
- **Duration**: Set the time window for frame collection (seconds)
- **Live Target FPS**: Set the target frame rate for processing
- **Continuous**: Enable continuous inference mode

#### Load Video Tab
- **Browse Video**: Load a video file (.mp4, .avi, .mov)
- **Browse Folder**: Load a folder of images (.png, .jpg, .jpeg, .bmp)
- **Start Folder Inference**: Run inference on loaded folder images
- **Play/Pause**: Control video playback
- **Start/Stop Recording**: Begin/end frame collection from video
- **Slider**: Seek through video or browse loaded images

### Sidebar Panels

The right sidebar contains three tabs:

1. **Anomalous**: View detected anomalies with scores
   - Toggle overlay mask display
   - Save selected images
   - Add to soft memory
   - Delete unwanted detections
   - Load More: View all anomalies grouped by time

2. **Saved**: Manual captures and saved images
   - Export to disk
   - Add to soft memory for reference

3. **Soft Memory**: Reference images for improved detection
   - Acts as a "known good" reference set
   - Included in all inference batches
   - Helps reduce false positives

## Configuration Options

### Available Backbone Models

The following pretrained models are supported:
- `dino_deitsmall16`
- `ViT-B-32`, `ViT-B-16`, `ViT-L-14`
- `dino_vitbase16`, `dino_vitbase8`
- `dinov2_vitb14`, `dinov2_vitl14`
- `google/siglip-so400m-patch14-384`
- `vit_small_patch32_224.augreg_in21k`
- `vit_tiny_patch16_224.augreg_in21k`
- `vit_small_patch16_224.dino`

### Image Resize Options

Each model supports specific image sizes based on their patch size. The GUI automatically shows compatible sizes when you select a model. Common sizes include:
- 224x224
- 256x256
- 384x384
- 512x512

### Threshold Settings

Access via the **Configuration** button:

- **Image Threshold** (1.0-10.0): Controls which frames are flagged as anomalous
  - Higher values = fewer false positives, may miss subtle anomalies
  - Lower values = more sensitive, may have more false positives
  - Default: 9.0

- **Overlay Threshold** (1.0-10.0): Controls the visualization overlay
  - Determines which regions in the anomaly map are highlighted
  - Does not affect detection, only visualization
  - Default: 3.5

## Workflow Examples

### Example 1: Live Camera Inspection

1. Launch the GUI: `python industrial_gui.py`
2. Click **Configuration** to select your model and thresholds
3. Set **Duration** (e.g., 3 seconds) and **Live Target FPS** (e.g., 15)
4. Click **Start** to begin collecting frames
5. Anomalies will appear in the **Anomalous** sidebar tab
6. Review and save anomalies as needed

### Example 2: Batch Processing Video

1. Switch to the **Load Video** tab
2. Click **Browse Video** and select your video file
3. Use the slider to preview the video
4. Set **Duration** and **Target FPS** for batch processing
5. Click **Start Recording** to begin collecting frames
6. Review detected anomalies in the sidebar

### Example 3: Processing Image Folders

1. Switch to the **Load Video** tab
2. Click **Browse Folder** and select a folder containing images
3. Preview images using the slider
4. Click **Start Folder Inference** to process all images
5. Review results in the **Anomalous** sidebar

## Troubleshooting

### Config File Not Found
If you see "Error: Could not find config.yaml file":
- Create a `config.yaml` in the current directory, or
- Create it in a `configs/` subdirectory
- Use the template provided above

### CUDA Out of Memory
- Reduce `img_resize` in config (e.g., 224 → 192)
- Decrease duration or target FPS
- Use a smaller backbone model
- Switch to CPU mode (set `device: 'cpu'` in config)

### Model Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Camera Not Working
- Check camera permissions
- Verify camera index (default is 0)
- Test camera with other applications first
- Try different camera indices in the code if needed

### Slow Performance
- Use a GPU for significantly faster inference
- Reduce image size (smaller sizes process faster)
- Use lighter backbone models (e.g., `vit_tiny_patch16_224.augreg_in21k`)
- Reduce the number of images in soft memory

## Performance Tips

- **GPU Usage**: Models run significantly faster on CUDA-enabled GPUs (10-50x speedup)
- **Image Size**: Smaller sizes (224x224) are faster but may miss fine details
- **Batch Processing**: Larger durations process more frames per inference batch
- **Soft Memory**: Keep soft memory size reasonable (10-50 images) for best performance
- **Model Selection**: Smaller models like `vit_tiny_patch16_224.augreg_in21k` are faster

## File Organization

The application creates the following directories:
- `soft_memory/`: Stores soft memory reference images
- `output_gui/`: Default output directory for results

## Advanced Usage

### Customizing Thresholds Per Use Case

Different industrial inspection scenarios may require different threshold settings:

- **High-precision inspection** (minimal false negatives): Lower image threshold (7.0-8.0)
- **Quality control** (minimal false positives): Higher image threshold (9.0-10.0)
- **General inspection**: Medium threshold (8.0-9.0)

### Using Soft Memory Effectively

Soft memory works best when:
1. You have 10-50 known good reference images
2. Reference images represent normal variations
3. Images are from the same product/setup
4. Images are captured under similar conditions

## Known Limitations

- Live camera FPS is limited by camera hardware and model inference speed
- Very large image sizes (>512) may cause memory issues on GPUs with limited VRAM
- Inference time increases with batch size and image resolution

## License

This GUI is built on top of the MuSc anomaly detection model.
See the main README.md for license information.

## Citation

If you use this work, please cite the original MuSc paper:

```bibtex
@inproceedings{Li2024MuSc,
  title={MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images},
  author={Li, Xurui and Huang, Ziming and Xue, Feng and Zhou, Yu},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## Support

For issues specific to the GUI, please check:
1. This README for common problems
2. The main MuSc README.md for model-related questions
3. GitHub issues for known problems and solutions
