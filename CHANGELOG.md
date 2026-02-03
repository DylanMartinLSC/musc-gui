# Changelog

All notable changes to MuSc Industrial GUI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub issue and PR templates
- CONTRIBUTING.md with development guidelines
- CHANGELOG.md (this file)
- Sample data directory structure with MVTec AD licensing info
- CLI demo script (`demo.py`) for non-GUI inference
- Smoke tests for model loading and inference
- GitHub Actions CI workflow
- FAQ section in README
- Type hints for core MuSc class
- Docstrings for public methods
- ARCHITECTURE.md documenting codebase structure
- Pre-commit hooks configuration (black, ruff)
- Unit tests for LNAMD and MSM modules
- Soft memory persistence (loads existing images on startup)
- JSON/CSV export for detection results
- Jupyter notebook example

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of MuSc Industrial GUI
- PyQt5-based graphical interface
- Zero-shot anomaly detection using MuSc algorithm (LNAMD + MSM + RsCIN)
- Support for 14+ backbone models:
  - DINO (ViT-S/8, ViT-S/16, ViT-B/8, ViT-B/16)
  - DINOv2 (ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14)
  - CLIP (ViT-B/16, ViT-L/14)
  - SigLIP variants
  - TIMM models
- Live camera feed processing
- Video file processing
- Batch image processing
- Soft memory system for reference images
- Configurable detection thresholds
- Heatmap overlay visualization
- Excel export for results
- Support for MVTec AD, VisA, and BTAD datasets
- Comprehensive documentation (README, README_GUI)
- MIT License

### Technical Details
- Based on "MuSc: Zero-Shot Anomaly Classification and Segmentation by Mutual Scoring" (ICLR 2024)
- GPU acceleration via CUDA
- CPU fallback support
- Multi-threaded inference for responsive UI

[Unreleased]: https://github.com/OWNER/musc-gui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/musc-gui/releases/tag/v0.1.0
