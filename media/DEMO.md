# Real MuSc GUI capture

Selected actual Windows application frames: load an image folder, Analyze Folder Images, inspect model results, then double-click a detected image to open the real overlay. Frames are paced into nine seconds; this is not a continuous speed recording. No UI or model output was fabricated.

The source app ran with CUDA PyTorch 2.5.1+cu124 on an RTX 3090, cached `dinov2_vitl14_reg`, 224px input, layer 11, r=1 and a 12-image pool (eight good and four broken_large bottle test images). Image threshold 7.0 (internally 0.7), overlay threshold 3.5 (internally 0.35). The app reported 0.78 s / 64.95 ms per image for this warm run. These are UI-reported values for this small pool, not a benchmark or accuracy qualification. Good images are also flagged at this illustrative threshold; overlays are coarse. The portable release uses CPU PyTorch and does not promise this GPU timing. The recording configuration was restored afterward.

Start with `python industrial_gui.py --no-camera` for folder-only use. This skips automatic camera connection while still loading the real model. The inference worker no longer calls OpenCV waitKey: Qt displays results on its main thread, and pumping a second GUI event loop from the worker blocked this run.

Images: MVTec AD, by Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger / MVTec Software GmbH. [Dataset and license terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). This GIF includes adapted dataset images (resizing and model heatmap overlays) and is shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This media license is separate from the application code license.
