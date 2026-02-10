import sys
import os
import yaml
import cv2
import torch
import numpy as np
from datetime import datetime
import time as pytime
from collections import deque
from functools import partial

from PyQt5.QtCore import (
    Qt,
    QTimer,
    QThread,
    pyqtSignal,
    pyqtSlot
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QDialogButtonBox,
    QTabWidget,
    QScrollArea,
    QGridLayout,
    QCheckBox,
    QFileDialog,
    QSlider,
    QStyle,
    QGroupBox,
    QProgressBar,
    QDockWidget,
    QLineEdit,
    QMessageBox,
    QStatusBar
)
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QFont,
    QIcon
)

try:
    from models.musc import MuSc
except ImportError:
    print("Warning: Could not import MuSc from models.musc. Please adjust import path as needed.")
    MuSc = None

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class InferenceThread(QThread):
    inferenceFinished = pyqtSignal(list, np.ndarray, float, float)
    progressChanged = pyqtSignal(int)

    def __init__(self, model, frames, parent=None):
        super().__init__(parent)
        self.model = model  # Move the model to the selected device
        self.frames = frames
        self.device = device # Store device for use in run method

    def run(self):
        if not self.frames:
            self.inferenceFinished.emit([], np.array([]), 0.0, 0.0)
            return

        start_time = pytime.time()
        batch_tensors = []
        target_size = self.model.cfg['datasets']['img_resize']
        N = len(self.frames)

        # Set the model to evaluation mode
        if hasattr(self.model, 'eval'): # Check if the model has an eval method
            self.model.eval()

        for i, frame in enumerate(self.frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            frame_norm = frame_resized.astype(np.float32) / 255.0
            tensor = torch.tensor(frame_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            batch_tensors.append(tensor)
            progress_val = int(((i + 1) / N) * 50)
            self.progressChanged.emit(progress_val)

        if not batch_tensors:
            self.inferenceFinished.emit(self.frames, np.array([]), 0.0, 0.0)
            return

        batch_input = torch.cat(batch_tensors, dim=0)

        with torch.no_grad():
            anomaly_maps_tensor = self.model.infer_on_images([batch_input])

        self.progressChanged.emit(100)

        # anomaly_maps_tensor is already a NumPy array returned by infer_on_images
        anomaly_maps = anomaly_maps_tensor # CORRECTED LINE

        if anomaly_maps.ndim == 4:
            # This check might be relevant if infer_on_images sometimes returns 4D
            # and sometimes 3D based on internal logic.
            # From musc.py, pr_px (which becomes anomaly_maps_tensor) is
            # reshaped to [N, H, W] or similar 3D/2D before returning.
            # If it was [N, 1, H, W] it gets squeezed.
            # So, this condition might still be useful depending on the exact output shape guarantees
            # of your MuSc.infer_on_images.
            # If MuSc.infer_on_images always returns [N, H, W] or [N, patch_count],
            # and then reshapes to [N,H,W], then this squeeze might not be needed
            # or needs to be adapted.
            # Given the musc.py code, pr_px becomes a 3D array [N, H, W] before returning.
            # Let's assume it is [N, H, W].
            # If it's [B, 1, H, W] and then squeezed to [B,H,W] in infer_on_images, then this is fine.
            # If it can be [B, C, H, W] where C > 1, this squeeze is problematic.
            # Let's assume infer_on_images consistently returns shape like [N, H, W] or [N, side, side]
            # or it's squeezed to that from [N, 1, H, W] inside infer_on_images.
            # The original code had .squeeze(1) if ndim == 4. This implies it expected [N, 1, H, W].
            # The MuSc code's infer_on_images does:
            # if pr_px.ndim == 4 and pr_px.shape[1] == 1: pr_px = pr_px[:, 0, :, :]
            # This makes pr_px 3D. So the check anomaly_maps.ndim == 4 and then anomaly_maps.squeeze(1)
            # in industrial_gui.py is unlikely to be true if MuSc already handled it.
            # For safety, let's keep the check, but it might be redundant if MuSc always returns 3D.

            # If MuSc.infer_on_images *always* returns a 3D [N, H, W] array,
            # then the squeeze is not needed here.
            # Based on musc.py's infer_on_images, pr_px is either reshaped to 3D [N, side, side]
            # or squeezed from 4D [N, 1, H, W] to 3D. So anomaly_maps should be 3D.
            # Thus, the `if anomaly_maps.ndim == 4:` block might not be strictly necessary
            # if MuSc is consistent.
            # However, to be safe and match the original logic closely:
            if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1: # More specific check
                    anomaly_maps = anomaly_maps.squeeze(1)


        # Ensure anomaly_maps is a NumPy array before reshaping (it is, from the correction)
        max_score = float(anomaly_maps.reshape(anomaly_maps.shape[0], -1).max())

        elapsed = pytime.time() - start_time
        self.inferenceFinished.emit(self.frames, anomaly_maps, max_score, elapsed)



###########################################################
# Config Dialog and Updated Image Preview Dialog
###########################################################
class ConfigDialog(QDialog):
    def __init__(self, config_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.setMinimumWidth(700)
        self.config_data = config_data
        if "thresholds" not in self.config_data:
            self.config_data["thresholds"] = {}
        thresholds = self.config_data["thresholds"]

        # Import utilities
        from utils import (
            get_gpu_info, get_recommended_device, get_models_by_category,
            get_model_info, format_model_details, get_category_display_name
        )

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()

        # ===== Configuration Presets Section =====
        preset_group = QGroupBox("⚡ Quick Setup (Choose a Preset)")
        preset_layout = QVBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("🔧 Custom Configuration (Manual)", None)
        self.preset_combo.addItem("🧪 Testing & Demo (Fast, works on CPU)", "testing")
        self.preset_combo.addItem("⚖️ Production Line (Balanced) - RECOMMENDED", "production")
        self.preset_combo.addItem("🎯 High Precision Inspection (Slow, accurate)", "precision")
        self.preset_combo.addItem("🖥️ CPU Mode (No GPU available)", "cpu")
        self.preset_combo.setToolTip(
            "Quick Setup Presets:\n"
            "• Testing & Demo: Fast model, good for trying out the system\n"
            "• Production Line: Balanced speed and accuracy (RECOMMENDED)\n"
            "• High Precision: Maximum accuracy for critical inspection\n"
            "• CPU Mode: For computers without GPU (very slow)\n"
            "• Custom: Manually configure all settings"
        )
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)

        preset_desc = QLabel(
            "💡 Tip: Start with a preset, then customize if needed. "
            "The 'Production Line' preset works well for most cases."
        )
        preset_desc.setWordWrap(True)
        preset_desc.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        preset_layout.addWidget(preset_desc)

        preset_group.setLayout(preset_layout)
        main_layout.addWidget(preset_group)

        # ===== Model Selection with Categories =====
        model_group = QGroupBox("AI Model Selection")
        model_layout = QVBoxLayout()

        self.modelCombo = QComboBox()
        self.populate_model_dropdown()

        current_backbone = config_data["models"].get("backbone_name", "dinov2_vitb14")
        # Find and select current model
        for i in range(self.modelCombo.count()):
            if self.modelCombo.itemData(i) == current_backbone:
                self.modelCombo.setCurrentIndex(i)
                break

        self.modelCombo.currentIndexChanged.connect(self.update_model_details)
        model_layout.addWidget(self.modelCombo)

        # Model details display
        self.model_details_label = QLabel()
        self.model_details_label.setWordWrap(True)
        self.model_details_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 10px; border-radius: 5px; "
            "font-family: monospace; font-size: 10pt;"
        )
        model_layout.addWidget(self.model_details_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # ===== GPU Device Selection with Auto-Detection =====
        device_group = QGroupBox("Processing Device")
        device_layout = QVBoxLayout()

        self.deviceCombo = QComboBox()
        self.populate_device_dropdown()

        # Set current device
        current_device = config_data.get("device", 0)
        for i in range(self.deviceCombo.count()):
            if self.deviceCombo.itemData(i) == current_device:
                self.deviceCombo.setCurrentIndex(i)
                break

        device_layout.addWidget(self.deviceCombo)

        # Device info label
        device_info = get_recommended_device()
        self.device_info_label = QLabel()
        self.device_info_label.setWordWrap(True)
        if 'warning' in device_info:
            self.device_info_label.setText(f"⚠️ {device_info.get('warning', '')}")
            self.device_info_label.setStyleSheet("color: #ff9800; padding: 5px;")
        else:
            self.device_info_label.setText(f"✓ {device_info.get('recommendation', '')}")
            self.device_info_label.setStyleSheet("color: #4caf50; padding: 5px;")
        device_layout.addWidget(self.device_info_label)

        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)

        # ===== Other Settings =====
        settings_group = QGroupBox("Advanced Settings")
        settings_layout = QFormLayout()

        # Image Resize Selection using QComboBox
        self.imgResizeCombo = QComboBox()
        # Define possible image sizes based on model patch sizes.
        self.image_size_options = {
                    "dino_deitsmall16": [224, 256, 512],
                    "ViT-B-32": [224, 384],
                    "ViT-B-16": [224, 256, 384],
                    "ViT-L-14": [224, 384, 512],
                    "dino_vitbase16": [224, 256, 512],
                    "dino_vitbase8": [224, 256],
                    "dinov2_vitb14": [224, 384, 512],
                    "dinov2_vitl14": [224, 512],
                    "google/siglip-so400m-patch14-384": [384, 392],
                    "vit_small_patch32_224.augreg_in21k": [224, 256],
                    "vit_tiny_patch16_224.augreg_in21k": [224, 256],
                    "vit_small_patch16_224.dino": [224, 256],
                }
        self.modelCombo.currentIndexChanged.connect(self.updateImageSizeOptions)
        # Initialize the dropdown based on the current model.
        current_model_name = self.modelCombo.currentData()
        if current_model_name:
            self.updateImageSizeOptions(current_model_name)
        self.imgResizeCombo.setToolTip(
            "Image Processing Size:\n"
            "• 224: Standard, fastest processing (RECOMMENDED)\n"
            "• 384/512: Higher quality but slower\n"
            "Larger images capture more detail but take longer to analyze"
        )
        settings_layout.addRow("Image Size:", self.imgResizeCombo)

        # Detection Sensitivity (converted to 0-100% from 1.0-10.0)
        self.imageThresholdSpin = QDoubleSpinBox()
        self.imageThresholdSpin.setRange(0, 100)
        self.imageThresholdSpin.setSingleStep(5)
        # Convert from 1.0-10.0 scale to 0-100 percentage
        old_value = thresholds.get("image_threshold", 9.0)
        self.imageThresholdSpin.setValue(old_value * 10)
        self.imageThresholdSpin.setSuffix("%")
        self.imageThresholdSpin.setToolTip(
            "Detection Sensitivity - How strict the defect detection should be:\n"
            "• High (80-100%): Strict - only obvious defects trigger alerts\n"
            "• Medium (60-80%): Balanced - good for most cases (RECOMMENDED)\n"
            "• Low (0-60%): Sensitive - catches subtle issues but may have false alarms"
        )
        settings_layout.addRow("Detection Sensitivity:", self.imageThresholdSpin)

        # Heatmap Visibility (converted to 0-100% from 1.0-10.0)
        self.overlayThresholdSpin = QDoubleSpinBox()
        self.overlayThresholdSpin.setRange(0, 100)
        self.overlayThresholdSpin.setSingleStep(5)
        # Convert from 1.0-10.0 scale to 0-100 percentage
        old_overlay = thresholds.get("overlay_threshold", 3.5)
        self.overlayThresholdSpin.setValue(old_overlay * 10)
        self.overlayThresholdSpin.setSuffix("%")
        self.overlayThresholdSpin.setToolTip(
            "Heatmap Intensity - How much of the defect areas to highlight:\n"
            "• High (80-100%): Only highlight most severe areas\n"
            "• Medium (30-50%): Balanced highlighting (RECOMMENDED)\n"
            "• Low (0-30%): Highlight all anomalous regions"
        )
        settings_layout.addRow("Heatmap Intensity:", self.overlayThresholdSpin)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Now that all widgets are created, update model details for initial selection
        self.update_model_details()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    def populate_model_dropdown(self):
        """Populate model dropdown with categorized models"""
        from utils import get_models_by_category, get_model_info, get_category_display_name

        self.modelCombo.clear()

        # Add models by category
        for category in ['fast', 'balanced', 'accurate']:
            # Add category header (non-selectable)
            self.modelCombo.addItem(f"──── {get_category_display_name(category)} ────", None)
            model_index = self.modelCombo.count() - 1
            self.modelCombo.model().item(model_index).setEnabled(False)

            # Add models in this category
            models = get_models_by_category(category)
            for model_name in models:
                info = get_model_info(model_name)
                if info:
                    display_text = f"   {info['display_name']}"
                    self.modelCombo.addItem(display_text, model_name)

    def populate_device_dropdown(self):
        """Populate device dropdown with detected GPUs"""
        from utils import get_gpu_info

        self.deviceCombo.clear()

        gpus = get_gpu_info()
        if gpus:
            for gpu in gpus:
                label = f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f}GB)"
                if gpu['recommended']:
                    label += " ✓ RECOMMENDED"
                self.deviceCombo.addItem(label, gpu['index'])
        else:
            self.deviceCombo.addItem("⚠️ CPU Mode (Slow - No GPU detected)", "cpu")

    def update_model_details(self):
        """Update the model details display when selection changes"""
        from utils import get_model_info, format_model_details, check_model_compatibility, get_recommended_device

        model_name = self.modelCombo.currentData()
        if not model_name:
            self.model_details_label.setText("")
            return

        info = get_model_info(model_name)
        if not info:
            self.model_details_label.setText("No information available for this model.")
            return

        # Format basic details
        details_text = format_model_details(model_name)

        # Check compatibility with current hardware
        device_info = get_recommended_device()
        compat = check_model_compatibility(model_name, device_info)

        if not compat.get('compatible', True):
            details_text += f"\n\n⚠️ WARNING: {compat['warning']}"
            details_text += f"\n💡 Try: {compat.get('recommendation', 'a smaller model')}"
        elif 'warning' in compat:
            details_text += f"\n\n⚠️ {compat['warning']}"

        self.model_details_label.setText(details_text)

        # Update image size options for this model
        self.updateImageSizeOptions(model_name)

    def updateImageSizeOptions(self, model_name):
        """Update available image sizes based on selected model"""
        # Check if imgResizeCombo exists yet (may be called during initialization)
        if not hasattr(self, 'imgResizeCombo'):
            return

        # Clear previous options
        self.imgResizeCombo.clear()

        # Get model info to find recommended sizes
        from utils import get_model_info
        info = get_model_info(model_name)

        if info and 'recommended_image_sizes' in info:
            sizes = info['recommended_image_sizes']
        else:
            # Fallback to predefined options
            sizes = self.image_size_options.get(model_name, [224])

        # Save current value if it exists
        current_size = self.config_data.get("datasets", {}).get("img_resize", 224)

        # Populate the combo box with available sizes
        for size in sizes:
            self.imgResizeCombo.addItem(f"{size}×{size}", size)

        # Try to restore previous selection
        for i in range(self.imgResizeCombo.count()):
            if self.imgResizeCombo.itemData(i) == current_size:
                self.imgResizeCombo.setCurrentIndex(i)
                break

    def apply_preset(self):
        """Apply a configuration preset"""
        from utils import get_recommended_device

        preset_name = self.preset_combo.currentData()
        if not preset_name:
            return  # Custom mode, no preset to apply

        # Define presets
        device_info = get_recommended_device()
        device_value = device_info['device'] if device_info['device'] != 'cpu' else 0

        presets = {
            "testing": {
                "model": "vit_tiny_patch16_224.augreg_in21k",
                "img_size": 224,
                "image_threshold": 85,
                "overlay_threshold": 40,
                "device": "cpu"  # CPU for testing
            },
            "production": {
                "model": "dinov2_vitb14",
                "img_size": 224,
                "image_threshold": 90,
                "overlay_threshold": 35,
                "device": device_value  # Use detected GPU
            },
            "precision": {
                "model": "dinov2_vitl14",
                "img_size": 384,
                "image_threshold": 95,
                "overlay_threshold": 30,
                "device": device_value
            },
            "cpu": {
                "model": "dino_deitsmall16",
                "img_size": 224,
                "image_threshold": 85,
                "overlay_threshold": 40,
                "device": "cpu"
            }
        }

        preset = presets.get(preset_name)
        if not preset:
            return

        # Apply preset values
        # Set model
        for i in range(self.modelCombo.count()):
            if self.modelCombo.itemData(i) == preset["model"]:
                self.modelCombo.setCurrentIndex(i)
                break

        # Set image size (will be updated by model change, but set explicitly too)
        for i in range(self.imgResizeCombo.count()):
            if self.imgResizeCombo.itemData(i) == preset["img_size"]:
                self.imgResizeCombo.setCurrentIndex(i)
                break

        # Set thresholds
        self.imageThresholdSpin.setValue(preset["image_threshold"])
        self.overlayThresholdSpin.setValue(preset["overlay_threshold"])

        # Set device
        if preset["device"] == "cpu":
            # Find CPU option
            for i in range(self.deviceCombo.count()):
                if self.deviceCombo.itemData(i) == "cpu":
                    self.deviceCombo.setCurrentIndex(i)
                    break
        else:
            # Find GPU option
            for i in range(self.deviceCombo.count()):
                if self.deviceCombo.itemData(i) == preset["device"]:
                    self.deviceCombo.setCurrentIndex(i)
                    break

    def accept(self):
        # Get model name from itemData (not text which includes display formatting)
        model_name = self.modelCombo.currentData()
        if model_name:
            self.config_data["models"]["backbone_name"] = model_name

        # Get device from combo box
        device_value = self.deviceCombo.currentData()
        self.config_data["device"] = device_value

        # Save the selected image size (as an integer)
        self.config_data["datasets"]["img_resize"] = self.imgResizeCombo.currentData()

        # Convert percentages back to 1.0-10.0 scale for internal use
        self.config_data["thresholds"]["image_threshold"] = self.imageThresholdSpin.value() / 10.0
        self.config_data["thresholds"]["overlay_threshold"] = self.overlayThresholdSpin.value() / 10.0

        super().accept()

    def reject(self):
        super().reject()



# ----------------------------------------------------------------
# Updated Image Preview Dialog with Arrow Buttons for Navigation
# ----------------------------------------------------------------
class ImagePreviewDialog(QDialog):
    def __init__(self, pixmap, image_list=None, current_index=0, parent=None):
        """
        If image_list is provided and contains more than one QPixmap,
        left/right arrow buttons will appear to allow navigation.
        """
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.image_list = image_list  # List of QPixmaps (if any)
        self.current_index = current_index
        self.original_pixmap = pixmap  # Current image to display
        self.scale_factor = 1.0

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title layout with close button.
        title_layout = QHBoxLayout()
        self.close_btn = QPushButton("X")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setFixedWidth(40)
        title_layout.addStretch()
        title_layout.addWidget(self.close_btn)
        main_layout.addLayout(title_layout)

        # Image display label.
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)
        self.update_image_label()

        # Navigation buttons – only visible if image_list is provided and has more than one image.
        if self.image_list is not None and len(self.image_list) > 1:
            nav_layout = QHBoxLayout()
            self.prev_btn = QPushButton()
            self.prev_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
            self.prev_btn.clicked.connect(self.on_prev_clicked)
            nav_layout.addWidget(self.prev_btn)
            nav_layout.addStretch()
            self.next_btn = QPushButton()
            self.next_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
            self.next_btn.clicked.connect(self.on_next_clicked)
            nav_layout.addWidget(self.next_btn)
            main_layout.addLayout(nav_layout)

    def update_image_label(self):
        # If an image list is available, update from the list using current_index.
        if self.image_list is not None and len(self.image_list) > self.current_index:
            self.original_pixmap = self.image_list[self.current_index]
        w = self.original_pixmap.width() * self.scale_factor
        h = self.original_pixmap.height() * self.scale_factor
        scaled = self.original_pixmap.scaled(int(w), int(h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def on_prev_clicked(self):
        if self.image_list is None:
            return
        # Go to previous image if available.
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image_label()

    def on_next_clicked(self):
        if self.image_list is None:
            return
        # Go to next image if available.
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.update_image_label()

    def mousePressEvent(self, event):
        child_widget = self.childAt(event.pos())
        if child_widget is None:
            self.close()
        else:
            super().mousePressEvent(event)


###########################################################
# Helper Widgets for Displaying Images (Saved, Anomaly, Soft Memory)
###########################################################
class SavedImageItem(QWidget):
    def __init__(self, pixmap, raw_frame, thumb_size=200, parent=None):
        super().__init__(parent)
        self.raw_frame = raw_frame
        self.original_pixmap = pixmap
        self.thumb_size = thumb_size
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        layout.addStretch()
        self.refresh_thumbnail()

    def refresh_thumbnail(self):
        thumb = self.original_pixmap.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(thumb)

    def isChecked(self):
        return self.checkbox.isChecked()

    def mouseDoubleClickEvent(self, event):
        # For demonstration, arrow navigation is not provided here (pass image_list=None).
        dialog = ImagePreviewDialog(self.original_pixmap, image_list=None, parent=self)
        dialog.exec_()
        super().mouseDoubleClickEvent(event)


class AnomalyImageItem(QWidget):
    def __init__(self, overlay_pixmap, overlay_bgr, raw_frame, anomaly_score, thumb_size=200, parent=None):
        super().__init__(parent)
        self.overlay_pixmap = overlay_pixmap
        self.overlay_bgr = overlay_bgr
        self.raw_frame = raw_frame
        self.thumb_size = thumb_size
        self.show_overlay = True
        self.anomaly_score = anomaly_score
        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        self.checkbox = QCheckBox()
        top_layout.addWidget(self.checkbox)
        self.image_label = QLabel()
        top_layout.addWidget(self.image_label)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        self.score_label = QLabel(f"Score: {self.anomaly_score:.2f}")
        layout.addWidget(self.score_label, alignment=Qt.AlignCenter)
        self.refresh_thumbnail()

    def refresh_thumbnail(self):
        if self.show_overlay:
            pix = self.overlay_pixmap
        else:
            raw_rgb = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = raw_rgb.shape
            qimg = QImage(raw_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
        thumb = pix.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(thumb)

    def setDisplayMode(self, show_overlay):
        self.show_overlay = show_overlay
        self.refresh_thumbnail()

    def isChecked(self):
        return self.checkbox.isChecked()

    def mouseDoubleClickEvent(self, event):
        if self.show_overlay:
            pix = self.overlay_pixmap
        else:
            raw_rgb = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = raw_rgb.shape
            qimg = QImage(raw_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
        # Arrow navigation not provided by default here.
        dialog = ImagePreviewDialog(pix, image_list=None, parent=self)
        dialog.exec_()
        super().mouseDoubleClickEvent(event)


class SoftMemoryImageItem(QWidget):
    def __init__(self, pixmap, raw_frame, thumb_size=200, parent=None):
        super().__init__(parent)
        self.raw_frame = raw_frame
        self.original_pixmap = pixmap
        self.thumb_size = thumb_size
        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        layout.addWidget(self.checkbox)
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        layout.addStretch()
        self.refresh_thumbnail()

    def refresh_thumbnail(self):
        thumb = self.original_pixmap.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(thumb)

    def isChecked(self):
        return self.checkbox.isChecked()

    def mouseDoubleClickEvent(self, event):
        dialog = ImagePreviewDialog(self.original_pixmap, image_list=None, parent=self)
        dialog.exec_()
        super().mouseDoubleClickEvent(event)

class SidebarMixin:
    """Adds smart refreshers + revised sidebar construction."""

    # .............................................................
    # helpers
    # .............................................................
    def _make_scroll_area(self, container):
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(container)
        return sa

    def _capture_scroll(self, scroll):
        if scroll is None:
            return lambda: None
        bar = scroll.verticalScrollBar(); pos = bar.value()
        return lambda: bar.setValue(pos)


    def _capture_checked(self, widgets):
        return {id(w) for w in widgets if w.isChecked()}

    # -------------------------------------------------------------
    # public refreshers – anomaly / saved / soft memory
    # -------------------------------------------------------------
    def refresh_anomaly_sidebar(self):
        """Rebuild the anomaly grid.

        - newest first
        - default view == last 10 frames only
        - ‘all’ view grouped by minute (yyyy-mm-dd HH:MM)
        """
        restore_scroll = self._capture_scroll(self.anomaly_scroll)
        was_checked    = {id(w.raw_frame) for w in self.anomaly_item_widgets if w.isChecked()}
        show_overlay   = self.anomaly_display_mask_checkbox.isChecked()

        # --- start with a clean slate --------------------------------------
        while self.anomaly_grid.count():
            item = self.anomaly_grid.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.anomaly_item_widgets.clear()

        # --- flatten & sort newest → oldest --------------------------------
        flat = []
        for day in self.anomalies_by_day.values():
            for hour in day.values():
                flat.extend(hour)                    # (pix, overlay, raw, ts, score)
        flat.sort(key=lambda tup: tup[3], reverse=True)

        if not self.anomaly_show_all:
            flat = flat[:9]                         # show only the newest 10

        # --- rebuild grid ---------------------------------------------------
        row, col, max_cols = 0, 0, 3
        last_minute_key = None
        for pix, overlay_bgr, raw, ts, score in flat:
            # minute header when ‘all’ is shown
            if self.anomaly_show_all:
                minute_key = ts.strftime("%Y-%m-%d  %H:%M")
                if minute_key != last_minute_key:
                    header = QLabel(minute_key);  header.setStyleSheet("font-weight:600")
                    self.anomaly_grid.addWidget(header, row, 0, 1, max_cols)
                    row += 1;  col = 0;  last_minute_key = minute_key

            w = AnomalyImageItem(pix, overlay_bgr, raw, score,
                                thumb_size=self.anomaly_thumb_size)
            w.setDisplayMode(show_overlay)
            if id(raw) in was_checked:                 # ← restore tick mark
                w.checkbox.setChecked(True)

            self.anomaly_item_widgets.append(w)
            self.anomaly_grid.addWidget(w, row, col)
            col = (col + 1) % max_cols
            if col == 0:
                row += 1

        restore_scroll()


    # generic helper for saved & soft‑memory --------------------------------
    def _generic_thumb_refresh(self, data_iter, grid, widgets, thumb_size, mk_item):
        # ---------------------------------------------------------------
        # find the *real* QScrollArea that encloses this grid
        scroll_area = grid.parent()
        from PyQt5.QtWidgets import QScrollArea
        while scroll_area and not isinstance(scroll_area, QScrollArea):
            scroll_area = scroll_area.parent()
        restore_scroll = (lambda: None) if scroll_area is None \
                        else self._capture_scroll(scroll_area)
        # ---------------------------------------------------------------
        was_checked   = {id(w.raw_frame) for w in widgets if w.isChecked()}
        seen = {id(w.raw_frame) for w in widgets}
        row, col, max_cols = 0, 0, 3
        if grid.count():
            row = grid.rowCount()
        for payload in data_iter:
            key = id(payload[-1]) if isinstance(payload, tuple) else id(payload)
            if key in seen:
                continue
            w = mk_item(payload, thumb_size)
            widgets.append(w)
            grid.addWidget(w, row, col)
            col = (col + 1) % max_cols
            if col == 0:
                row += 1
            seen.add(key)
        for w in widgets:
            w.checkbox.setChecked(id(w) in was_checked)
        restore_scroll()

    def refresh_saved_sidebar(self):
        self._generic_thumb_refresh(
            self.saved_images_data,
            self.saved_grid,
            self.saved_item_widgets,
            self.saved_thumb_size,
            lambda pr, ts: SavedImageItem(pr[0], pr[1], thumb_size=ts)
        )

    def refresh_soft_memory_sidebar(self):
        def make_item(raw, ts):
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            h, w_, ch = rgb.shape
            qimg = QImage(rgb.data, w_, h, ch * w_, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg)
            return SoftMemoryImageItem(pix, raw, thumb_size=ts)
        self._generic_thumb_refresh(
            self.soft_memory_frames,
            self.soft_grid,
            self.soft_memory_item_widgets,
            self.soft_memory_thumb_size,
            make_item
        )

###########################################################
# Main Window with Unified Sidebar and Load Video Tab
###########################################################
class MainWindow(QMainWindow, SidebarMixin):
    def __init__(self, cfgPath, parent=None):
        from collections import deque

        super().__init__(parent)
        self.setWindowTitle("Batch Inference")
        self.resize(1280, 720)
        # Data storage for images:
        self.saved_images_data = []         # For "Saved" images: list of (QPixmap, raw_frame)
        self.saved_item_widgets = []          # Widgets for saved images
        self.soft_memory_frames = []          # Soft memory raw frames (BGR)
        self.soft_memory_item_widgets = []    # Widgets for soft memory images
        self.anomalies_by_day = {}            # Dictionary for anomalous images (organized by day/hour)
        self.anomaly_item_widgets = []
        # Thumbnail size defaults:
        self.saved_thumb_size = 200
        self.soft_memory_thumb_size = 200
        self.anomaly_thumb_size = 200

        self.anomaly_show_all = False  

        # Create main tabs for Live and Load Video.
        self.live_tab = QWidget()
        self.load_video_tab = QWidget()
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        self.tab_widget.addTab(self.live_tab, "Live Camera")
        self.tab_widget.addTab(self.load_video_tab, "Load Video")

        self.setup_live_tab()
        self.setup_video_tab()  # Restored load video tab functionality

        # Create unified image sidebar (dock widget) for Saved, Anomaly, and Soft Memory.
        self.setup_sidebar()

        # Load configuration file and initialize model.
        self.cfgPath = cfgPath
        with open(cfgPath, "r") as f:
            self.config_data = yaml.safe_load(f)

        # Add status bar with plain language status indicators (after config is loaded)
        self.setup_status_bar()
        try:
            if torch.cuda.is_available():
                dev_index = self.config_data.get('device', 0)
                self.config_data['device'] = dev_index
            else:
                self.config_data['device'] = 'cpu'
            self.model = MuSc(self.config_data)
        except Exception as e:
            print("Error creating model:", e)
            self.model = None

        # Live camera variables.
        self.anomaly_scores_history = deque(maxlen=10)
        self.cap = None  # Will be initialized when connecting to camera
        self.camera_source = None  # Store the current camera source (0 for USB or URL for IP)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.recording = False
        self.frame_queue = []
        self.batch_size = 0
        self.inference_thread = None
        self.inference_in_progress = False
        self.latest_frame = None

        # For live FPS selection
        self.live_frame_counter = 0
        self.live_frame_interval = 1  # Will be computed when starting recording

        # Initialize with USB camera by default
        self.connect_to_camera(0)

    # ------------------------------
    # Status Bar Setup
    # ------------------------------
    def setup_status_bar(self):
        """Add status bar with plain language indicators"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Left: Current operation status
        self.status_left = QLabel("Ready to start")
        self.status_left.setStyleSheet("color: green; font-weight: bold;")

        # Center: Model information
        model_name = self.config_data.get("models", {}).get("backbone_name", "Unknown")
        model_display = self._get_model_display_name(model_name)
        self.status_center = QLabel(f"AI Model: {model_display}")

        # Right: Camera/Device status
        self.status_right = QLabel("Camera: Not connected")
        self.status_right.setStyleSheet("color: orange;")

        self.status_bar.addWidget(self.status_left, 1)
        self.status_bar.addPermanentWidget(self.status_center)
        self.status_bar.addPermanentWidget(self.status_right)

    def _get_model_display_name(self, model_name):
        """Convert technical model names to friendly descriptions"""
        model_map = {
            "dinov2_vitb14": "DINOv2 Base (Balanced)",
            "dinov2_vitl14": "DINOv2 Large (High Accuracy)",
            "dino_deitsmall16": "DeiT Small (Fast)",
            "dino_vitbase16": "ViT Base (Balanced)",
            "vit_tiny_patch16_224.augreg_in21k": "ViT Tiny (Fastest)",
            "vit_small_patch16_224.dino": "ViT Small (Fast)",
            "ViT-B-32": "ViT Base-32",
            "ViT-B-16": "ViT Base-16",
            "ViT-L-14": "ViT Large-14",
            "google/siglip-so400m-patch14-384": "SigLIP (High Detail)"
        }
        return model_map.get(model_name, model_name)

    def update_status(self, state, message=None):
        """Update status bar with current operation state

        States: ready, connecting, connected, collecting, analyzing, complete, error
        """
        status_config = {
            "ready": {"text": "Ready to start", "color": "green"},
            "connecting": {"text": "Connecting to camera...", "color": "orange"},
            "connected": {"text": "Connected - Ready to analyze", "color": "green"},
            "collecting": {"text": "Collecting frames from camera...", "color": "blue"},
            "analyzing": {"text": "Analyzing for defects...", "color": "blue"},
            "complete": {"text": "Analysis complete", "color": "green"},
            "error": {"text": message or "Error occurred", "color": "red"}
        }

        config = status_config.get(state, {"text": message or state, "color": "black"})
        self.status_left.setText(config["text"])
        self.status_left.setStyleSheet(f"color: {config['color']}; font-weight: bold;")

    def update_camera_status(self, source, connected=True):
        """Update camera connection status"""
        if connected:
            if isinstance(source, int):
                text = f"Camera: USB Device {source} connected"
                color = "green"
            else:
                text = "Camera: IP Camera connected"
                color = "green"
        else:
            text = "Camera: Not connected"
            color = "orange"

        self.status_right.setText(text)
        self.status_right.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ------------------------------
    # Unified Sidebar Setup
    # ------------------------------
    def setup_sidebar(self):
        self.sidebar_dock = QDockWidget("Image Sidebar", self)
        self.sidebar_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.sidebar_tab_widget = QTabWidget(); self.sidebar_dock.setWidget(self.sidebar_tab_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.sidebar_dock)

        # ===================== Anomalous Images TAB =======================
        self.anomaly_sidebar = QWidget(); an_layout = QVBoxLayout(self.anomaly_sidebar)
        # buttons (copied verbatim from original implementation) ----------
        an_btn_layout = QHBoxLayout()
        self.anomaly_save_to_soft_btn = QPushButton("Save to Reference Library")
        self.anomaly_save_to_soft_btn.setToolTip(
            "Add selected images to Reference Library.\n"
            "These images help the AI learn what's normal in your process."
        )
        self.anomaly_save_to_soft_btn.clicked.connect(self.save_selected_anomaly_to_soft)
        self.anomaly_save_btn = QPushButton("Save Selected")
        self.anomaly_save_btn.setToolTip("Save selected defect images to a folder on your computer")
        self.anomaly_save_btn.clicked.connect(self.save_selected_anomaly_images)
        self.anomaly_delete_btn = QPushButton("Delete Selected")
        self.anomaly_delete_btn.setToolTip("Remove selected images from this list")
        self.anomaly_delete_btn.clicked.connect(self.delete_selected_anomaly_images)
        self.anomaly_select_all_btn = QPushButton("Select/Deselect All")
        self.anomaly_select_all_btn.setToolTip("Toggle selection of all images in this list")
        self.anomaly_select_all_btn.clicked.connect(self.toggle_select_all_anomaly)
        for b in (self.anomaly_save_to_soft_btn, self.anomaly_save_btn, self.anomaly_delete_btn, self.anomaly_select_all_btn):
            an_btn_layout.addWidget(b)
        an_layout.addLayout(an_btn_layout)
        # master Display Mask checkbox ------------------------------------
        self.anomaly_display_mask_checkbox = QCheckBox("Show Defect Highlights")
        self.anomaly_display_mask_checkbox.setChecked(True)
        self.anomaly_display_mask_checkbox.setToolTip(
            "Show/hide the colored overlay highlighting defect areas.\n"
            "Red areas = detected defects or anomalies"
        )
        self.anomaly_display_mask_checkbox.toggled.connect(self.on_anomaly_display_mask_toggled)
        an_layout.addWidget(self.anomaly_display_mask_checkbox)

        # --- Load-more toggle ----------------------------------------
        self.anomaly_load_more_btn = QPushButton("Load More")
        self.anomaly_load_more_btn.clicked.connect(self.toggle_load_more_anomalies)
        an_layout.addWidget(self.anomaly_load_more_btn)

        # grid + scroll -----------------------------------------------------
        self.anomaly_grid = QGridLayout(); an_container = QWidget(); an_container.setLayout(self.anomaly_grid)
        self.anomaly_scroll = self._make_scroll_area(an_container)
        an_layout.addWidget(self.anomaly_scroll)
        self.sidebar_tab_widget.addTab(self.anomaly_sidebar, "Detected Defects")
        self.sidebar_tab_widget.setTabToolTip(0, "Images where defects or anomalies were detected")

        # ===================== Saved Images TAB ===========================
        self.saved_sidebar = QWidget(); sv_layout = QVBoxLayout(self.saved_sidebar)
        sv_btn_layout = QHBoxLayout()
        self.saved_save_to_soft_btn = QPushButton("Save to Reference Library")
        self.saved_save_to_soft_btn.setToolTip(
            "Add selected images to Reference Library.\n"
            "Use good/normal images to help improve detection accuracy."
        )
        self.saved_save_to_soft_btn.clicked.connect(self.save_selected_to_soft_memory)
        self.saved_save_btn = QPushButton("Save Selected")
        self.saved_save_btn.setToolTip("Save selected images to a folder on your computer")
        self.saved_save_btn.clicked.connect(self.save_selected_saved_images)
        self.saved_delete_btn = QPushButton("Delete Selected")
        self.saved_delete_btn.setToolTip("Remove selected images from this list")
        self.saved_delete_btn.clicked.connect(self.delete_selected_saved_images)
        self.saved_select_all_btn = QPushButton("Select/Deselect All")
        self.saved_select_all_btn.setToolTip("Toggle selection of all images in this list")
        self.saved_select_all_btn.clicked.connect(self.toggle_select_all_saved)
        for b in (self.saved_save_to_soft_btn, self.saved_save_btn, self.saved_delete_btn, self.saved_select_all_btn):
            sv_btn_layout.addWidget(b)
        sv_layout.addLayout(sv_btn_layout)
        self.saved_grid = QGridLayout(); sv_container = QWidget(); sv_container.setLayout(self.saved_grid)
        self.saved_scroll = self._make_scroll_area(sv_container)
        sv_layout.addWidget(self.saved_scroll)
        self.sidebar_tab_widget.addTab(self.saved_sidebar, "Captured Images")
        self.sidebar_tab_widget.setTabToolTip(1, "Images you manually captured using the Capture button")

        # ===================== Soft Memory TAB ===========================
        self.soft_memory_sidebar = QWidget(); sm_layout = QVBoxLayout(self.soft_memory_sidebar)
        sm_btn_layout = QHBoxLayout()
        self.soft_delete_btn = QPushButton("Delete Selected")
        self.soft_delete_btn.setToolTip("Remove selected images from the Reference Library")
        self.soft_delete_btn.clicked.connect(self.soft_delete_selected)
        self.soft_clear_btn = QPushButton("Clear All")
        self.soft_clear_btn.setToolTip(
            "Remove ALL images from Reference Library.\n"
            "Warning: This will reset the AI's learned examples!"
        )
        self.soft_clear_btn.clicked.connect(self.soft_clear_all)
        self.soft_select_all_btn = QPushButton("Select/Deselect All")
        self.soft_select_all_btn.setToolTip("Toggle selection of all images in this list")
        self.soft_select_all_btn.clicked.connect(self.soft_toggle_select_all)
        for b in (self.soft_delete_btn, self.soft_clear_btn, self.soft_select_all_btn):
            sm_btn_layout.addWidget(b)
        sm_layout.addLayout(sm_btn_layout)
        self.soft_grid = QGridLayout(); sm_container = QWidget(); sm_container.setLayout(self.soft_grid)
        self.soft_scroll = self._make_scroll_area(sm_container)
        sm_layout.addWidget(self.soft_scroll)
        self.sidebar_tab_widget.addTab(self.soft_memory_sidebar, "Reference Library")
        self.sidebar_tab_widget.setTabToolTip(2,
            "Known good images that help the AI learn what's normal.\n"
            "Add examples of defect-free products to improve detection accuracy."
        )


    @pyqtSlot(bool)
    def on_anomaly_display_mask_toggled(self, checked):
        # Update every anomaly image item with the new display mode.
        for widget in self.anomaly_item_widgets:
            widget.setDisplayMode(checked)

    # ------------------------------
    # Refresh methods for sidebar views.
    # ------------------------------




    # ------------------------------
    # Actions for Saved / Soft / Anomalous images (unchanged from previous version)
    # ------------------------------
    @pyqtSlot()
    def save_selected_to_soft_memory(self):
        soft_memory_dir = os.path.join(os.getcwd(), "soft_memory")
        os.makedirs(soft_memory_dir, exist_ok=True)
        count = 0
        for widget in self.saved_item_widgets:
            if not widget.isChecked():
                continue
            self.soft_memory_frames.append(widget.raw_frame)
            filename = os.path.join(
                soft_memory_dir, f"soft_memory_{pytime.time():.0f}_{count}.png"
            )
            cv2.imwrite(filename, widget.raw_frame)
            count += 1
        print(f"Added {count} images to soft memory.")
        self.refresh_soft_memory_sidebar()


    @pyqtSlot()
    def delete_selected_saved_images(self):
        keep_data = []
        for widget, data_item in zip(self.saved_item_widgets, self.saved_images_data):
            if not widget.isChecked():
                keep_data.append(data_item)
        self.saved_images_data = keep_data
        self.refresh_saved_sidebar()

    @pyqtSlot()
    def clear_all_saved_images(self):
        self.saved_images_data = []
        self.refresh_saved_sidebar()

    @pyqtSlot()
    def soft_toggle_select_all(self):
        if not self.soft_memory_item_widgets:
            return
        all_selected = all(widget.isChecked() for widget in self.soft_memory_item_widgets)
        for widget in self.soft_memory_item_widgets:
            widget.checkbox.setChecked(not all_selected)

    @pyqtSlot()
    def soft_delete_selected(self):
        new_frames = []
        for widget, frame in zip(self.soft_memory_item_widgets, self.soft_memory_frames):
            if not widget.isChecked():
                new_frames.append(frame)
        self.soft_memory_frames = new_frames
        self.refresh_soft_memory_sidebar()

    @pyqtSlot()
    def soft_clear_all(self):
        self.soft_memory_frames = []
        soft_memory_dir = os.path.join(os.getcwd(), "soft_memory")
        if os.path.exists(soft_memory_dir):
            for f in os.listdir(soft_memory_dir):
                file_path = os.path.join(soft_memory_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        self.refresh_soft_memory_sidebar()
        print("Soft memory cleared.")

    @pyqtSlot()
    def toggle_select_all_saved(self):
        all_selected = all(widget.isChecked() for widget in self.saved_item_widgets)
        for widget in self.saved_item_widgets:
            widget.checkbox.setChecked(not all_selected)

    @pyqtSlot()
    def toggle_select_all_anomaly(self):
        if not self.anomaly_item_widgets:
            return
        all_selected = all(widget.isChecked() for widget in self.anomaly_item_widgets)
        for widget in self.anomaly_item_widgets:
            widget.checkbox.setChecked(not all_selected)

    @pyqtSlot()
    def save_selected_saved_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not folder:
            return
        count = 0
        for widget in self.saved_item_widgets:
            if not widget.isChecked():
                continue
            filename = os.path.join(
                folder, f"saved_image_{pytime.time():.0f}_{count}.png"
            )
            cv2.imwrite(filename, widget.raw_frame)
            count += 1
        print(f"Saved {count} images to {folder}.")


    @pyqtSlot()
    def save_selected_anomaly_images(self):
        """Write *only the ticked anomalies visible in the current view*."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not folder:
            return

        count = 0
        for widget in self.anomaly_item_widgets:
            if not widget.isChecked():
                continue
            # Choose which version you want to save ↓
            frame_to_write = widget.raw_frame          #   raw RGB image (current behaviour)
            # frame_to_write = widget.overlay_bgr      # ← save the coloured overlay instead
            filename = os.path.join(
                folder, f"anomaly_{pytime.time():.0f}_{count}.png"
            )
            cv2.imwrite(filename, frame_to_write)
            count += 1

        print(f"Saved {count} anomaly images to {folder}.")


    @pyqtSlot()
    def save_selected_anomaly_to_soft(self):
        soft_memory_dir = os.path.join(os.getcwd(), "soft_memory")
        os.makedirs(soft_memory_dir, exist_ok=True)

        count = 0
        for widget in self.anomaly_item_widgets:
            if not widget.isChecked():
                continue
            self.soft_memory_frames.append(widget.raw_frame)
            filename = os.path.join(
                soft_memory_dir, f"soft_memory_anom_{pytime.time():.0f}_{count}.png"
            )
            cv2.imwrite(filename, widget.raw_frame)
            count += 1

        print(f"Added {count} anomaly images to soft memory.")
        self.refresh_soft_memory_sidebar()


    @pyqtSlot()
    def toggle_load_more_anomalies(self):
        """Flip between ‘last-10’ and ‘all, grouped by minute’ views."""
        self.anomaly_show_all = not self.anomaly_show_all
        self.anomaly_load_more_btn.setText(
            "Show Recent" if self.anomaly_show_all else "Load More"
        )
        self.refresh_anomaly_sidebar()      # rebuild with the new rule


    @pyqtSlot()
    def delete_selected_anomaly_images(self):
        """
        Remove only those anomalies whose check-box is ticked in the *current view*
        (whether we’re showing 10 or showing all).  Works by matching the raw-frame
        object id instead of relying on widget ordering.
        """
        # 1) collect ids of the raw BGR frames that are checked
        to_remove_ids = {id(w.raw_frame) for w in self.anomaly_item_widgets if w.isChecked()}
        if not to_remove_ids:          # nothing selected
            return

        # 2) walk the nested dict and drop any entry whose raw-frame id matches
        for day in list(self.anomalies_by_day.keys()):
            for hour in list(self.anomalies_by_day[day].keys()):
                self.anomalies_by_day[day][hour] = [
                    rec for rec in self.anomalies_by_day[day][hour]
                    if id(rec[2]) not in to_remove_ids          # rec[2] == raw_frame
                ]
                if not self.anomalies_by_day[day][hour]:
                    del self.anomalies_by_day[day][hour]
            if not self.anomalies_by_day[day]:
                del self.anomalies_by_day[day]

        self.refresh_anomaly_sidebar()



    # ------------------------------
    # Live Tab Setup with additional Target FPS control.
    # ------------------------------
    def setup_live_tab(self):
        layout = QVBoxLayout(self.live_tab)

        # Camera source selection group
        camera_group = QGroupBox("Camera Source")
        camera_layout = QVBoxLayout()

        # USB Camera option
        usb_layout = QHBoxLayout()
        usb_label = QLabel("USB Camera (Device Index):")
        self.usb_device_spin = QSpinBox()
        self.usb_device_spin.setRange(0, 10)
        self.usb_device_spin.setValue(0)
        self.usb_device_spin.setToolTip(
            "USB Camera Device Number:\n"
            "• 0: First/default camera (usually built-in webcam)\n"
            "• 1, 2, etc.: Additional USB cameras\n"
            "Try different numbers if your camera isn't detected"
        )
        self.usb_connect_btn = QPushButton("Connect USB Camera")
        self.usb_connect_btn.setToolTip("Connect to a USB camera using the device number above")
        self.usb_connect_btn.clicked.connect(self.connect_usb_camera)
        usb_layout.addWidget(usb_label)
        usb_layout.addWidget(self.usb_device_spin)
        usb_layout.addWidget(self.usb_connect_btn)
        usb_layout.addStretch()
        camera_layout.addLayout(usb_layout)

        # IP Camera option
        ip_layout = QHBoxLayout()
        ip_label = QLabel("IP Camera URL:")
        self.ip_camera_input = QLineEdit()
        self.ip_camera_input.setPlaceholderText("rtsp://username:password@ip:port/path or http://ip:port/video")
        self.ip_camera_input.setMinimumWidth(400)
        self.ip_camera_input.setToolTip(
            "Enter IP Camera URL in one of these formats:\n"
            "• RTSP: rtsp://username:password@192.168.1.100:554/stream\n"
            "• HTTP: http://192.168.1.100:80/video\n"
            "Replace username, password, and IP address with your camera's details"
        )
        self.ip_connect_btn = QPushButton("Connect IP Camera")
        self.ip_connect_btn.setToolTip("Connect to an IP/network camera using the URL above")
        self.ip_connect_btn.clicked.connect(self.connect_ip_camera)
        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.ip_camera_input)
        ip_layout.addWidget(self.ip_connect_btn)
        camera_layout.addLayout(ip_layout)

        # Current connection status
        self.connection_status_label = QLabel("Status: Not Connected")
        self.connection_status_label.setStyleSheet("color: orange; font-weight: bold;")
        camera_layout.addWidget(self.connection_status_label)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Warning label and video display
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)
        self.warning_label = QLabel("❗")
        self.warning_label.setStyleSheet("color: red; font-size: 24px;")
        self.warning_label.setVisible(False)
        top_layout.addWidget(self.warning_label)
        top_layout.addStretch()
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 600)
        layout.addWidget(self.video_label)
        self.anomaly_label = QLabel("Defect Level: Not started")
        self.anomaly_label.setAlignment(Qt.AlignCenter)
        self.anomaly_label.setToolTip(
            "Defect Level indicates the severity of detected anomalies:\n"
            "• 0.0 - 0.3: Normal (no defects detected)\n"
            "• 0.3 - 0.7: Minor issues (small scratches, dirt)\n"
            "• 0.7 - 0.9: Moderate defects (visible damage)\n"
            "• 0.9 - 1.0: Major defects (significant problems)"
        )
        layout.addWidget(self.anomaly_label)
        self.collection_progress_bar = QProgressBar()
        self.collection_progress_bar.setValue(0)
        layout.addWidget(QLabel("Image Collection Progress:"))
        layout.addWidget(self.collection_progress_bar)
        self.inference_progress_bar = QProgressBar()
        self.inference_progress_bar.setValue(0)
        self.inference_progress_bar.setMaximum(100)
        layout.addWidget(QLabel("Inference Progress:"))
        layout.addWidget(self.inference_progress_bar)
        self.inference_time_label = QLabel("Analysis Time: ---")
        self.inference_time_label.setAlignment(Qt.AlignCenter)
        self.inference_time_label.setToolTip(
            "Time taken to analyze the batch of images.\n"
            "Faster times = better real-time performance."
        )
        layout.addWidget(self.inference_time_label)

        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        # Configuration button.
        config_layout = QVBoxLayout()
        config_layout.setAlignment(Qt.AlignCenter)
        self.config_btn = QPushButton("Configuration")
        self.config_btn.setToolTip(
            "Open configuration settings to:\n"
            "• Choose AI model (Fast, Balanced, or High Accuracy)\n"
            "• Adjust detection sensitivity\n"
            "• Change image processing settings"
        )
        self.config_btn.clicked.connect(self.open_config_dialog)
        config_layout.addWidget(self.config_btn, alignment=Qt.AlignCenter)
        controls_layout.addLayout(config_layout)

        # Duration control.
        duration_layout = QVBoxLayout()
        duration_layout.setAlignment(Qt.AlignCenter)
        self.duration_label = QLabel("Duration (s):")
        self.duration_label.setAlignment(Qt.AlignCenter)
        duration_layout.addWidget(self.duration_label)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 60.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(3.0)
        self.duration_spin.setToolTip(
            "How many seconds of video to collect before analyzing.\n"
            "Example: 3 seconds at 15 FPS = 45 frames per batch.\n"
            "Longer duration = more context but slower results."
        )
        duration_layout.addWidget(self.duration_spin, alignment=Qt.AlignCenter)
        controls_layout.addLayout(duration_layout)

        # New: Live Target FPS control.
        fps_layout = QVBoxLayout()
        self.live_target_fps_label = QLabel("Live Target FPS:")
        self.live_target_fps_label.setAlignment(Qt.AlignCenter)
        fps_layout.addWidget(self.live_target_fps_label)
        self.live_target_fps_spin = QDoubleSpinBox()
        self.live_target_fps_spin.setRange(1, 60)
        self.live_target_fps_spin.setDecimals(0)
        self.live_target_fps_spin.setSingleStep(1)
        self.live_target_fps_spin.setValue(15)
        self.live_target_fps_spin.setToolTip(
            "Target frame rate: How many frames per second to capture.\n"
            "Higher FPS = smoother detection but more processing.\n"
            "Recommended: 15 FPS for live inspection, 30 FPS for video playback."
        )
        fps_layout.addWidget(self.live_target_fps_spin, alignment=Qt.AlignCenter)
        controls_layout.addLayout(fps_layout)

        # Actions: Start, Stop, Capture.
        actions_layout = QVBoxLayout()
        continuous_row = QHBoxLayout()
        continuous_label = QLabel("Continuous:")
        self.continuous_checkbox = QCheckBox()
        self.continuous_checkbox.setToolTip(
            "Continuous Mode: Keep analyzing new batches automatically.\n"
            "Unchecked: Analyze once and stop.\n"
            "Checked: Continuously monitor for defects in real-time."
        )
        continuous_row.addWidget(continuous_label)
        continuous_row.addWidget(self.continuous_checkbox)
        actions_layout.addWidget(QLabel("Actions:"))
        actions_layout.addLayout(continuous_row)
        self.start_btn = QPushButton("Start")
        self.start_btn.setToolTip(
            "Start collecting frames from your camera.\n"
            "Frames will be analyzed in batches to detect defects or anomalies."
        )
        self.start_btn.clicked.connect(self.startRecording)
        actions_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip(
            "Stop collecting and analyzing frames.\n"
            "Any frames currently being analyzed will complete."
        )
        self.stop_btn.clicked.connect(self.stopRecording)
        actions_layout.addWidget(self.stop_btn)
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setToolTip(
            "Capture a single frame from the camera and save it.\n"
            "Use this to collect sample images for reference."
        )
        self.capture_btn.clicked.connect(self.capture_still)
        actions_layout.addWidget(self.capture_btn)
        controls_layout.addLayout(actions_layout)

    # ------------------------------
    # Restored Load Video Tab Setup and Methods (unchanged)
    # ------------------------------
    def setup_video_tab(self):
        layout = QVBoxLayout(self.load_video_tab)
        # Top row: Browse buttons and new Start Folder Inference button
        top_layout = QHBoxLayout()
        self.browse_video_btn = QPushButton("Browse Video")
        self.browse_video_btn.setToolTip("Select a video file (.mp4, .avi, .mov) to analyze")
        self.browse_video_btn.clicked.connect(self.on_browse_video)
        top_layout.addWidget(self.browse_video_btn)
        # New: Browse Folder Button for images
        self.browse_folder_btn = QPushButton("Browse Folder")
        self.browse_folder_btn.setToolTip("Select a folder containing images (.png, .jpg, .bmp) to analyze")
        self.browse_folder_btn.clicked.connect(self.on_browse_folder)
        top_layout.addWidget(self.browse_folder_btn)
        # New: Start Folder Inference Button
        self.folder_inference_btn = QPushButton("Analyze Folder Images")
        self.folder_inference_btn.setToolTip("Analyze all images in the loaded folder for defects")
        self.folder_inference_btn.clicked.connect(self.runFolderInference)
        top_layout.addWidget(self.folder_inference_btn)
        layout.addLayout(top_layout)

        # Video display label
        self.video_display_label = QLabel("Video Feed")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setMinimumSize(600, 600)
        layout.addWidget(self.video_display_label)
        # Anomaly score label
        self.video_anomaly_label = QLabel("Defect Level: ---")
        self.video_anomaly_label.setAlignment(Qt.AlignCenter)
        self.video_anomaly_label.setToolTip(
            "Defect Level indicates severity:\n"
            "• 0-30%: Normal (no defects)\n"
            "• 30-70%: Minor issues\n"
            "• 70-90%: Moderate defects\n"
            "• 90-100%: Major defects"
        )
        layout.addWidget(self.video_anomaly_label)
        # Progress bars and inference time for video
        self.video_collection_progress_bar = QProgressBar()
        self.video_collection_progress_bar.setValue(0)
        layout.addWidget(QLabel("Image Collection Progress:"))
        layout.addWidget(self.video_collection_progress_bar)
        self.video_inference_progress_bar = QProgressBar()
        self.video_inference_progress_bar.setValue(0)
        self.video_inference_progress_bar.setMaximum(100)
        layout.addWidget(QLabel("Inference Progress:"))
        layout.addWidget(self.video_inference_progress_bar)
        self.video_inference_time_label = QLabel("Analysis Time: ---")
        self.video_inference_time_label.setAlignment(Qt.AlignCenter)
        self.video_inference_time_label.setToolTip("Time taken to analyze the video or images")
        layout.addWidget(self.video_inference_time_label)
        # Controls: Configuration, Duration, and Target FPS
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)
        settings_layout = QVBoxLayout()
        self.video_config_btn = QPushButton("Configuration")
        self.video_config_btn.clicked.connect(self.open_config_dialog)
        settings_layout.addWidget(self.video_config_btn, alignment=Qt.AlignCenter)
        controls_layout.addLayout(settings_layout)
        duration_layout = QVBoxLayout()
        self.video_duration_label = QLabel("Duration (s):")
        self.video_duration_label.setAlignment(Qt.AlignCenter)
        duration_layout.addWidget(self.video_duration_label)
        self.video_duration_spin = QDoubleSpinBox()
        self.video_duration_spin.setRange(0.0, 60.0)
        self.video_duration_spin.setDecimals(2)
        self.video_duration_spin.setSingleStep(0.1)
        self.video_duration_spin.setValue(3.0)
        self.video_duration_spin.setToolTip(
            "Duration of video to analyze in one batch.\n"
            "Example: 3 seconds at 30 FPS = 90 frames analyzed together"
        )
        duration_layout.addWidget(self.video_duration_spin, alignment=Qt.AlignCenter)
        controls_layout.addLayout(duration_layout)
        target_fps_layout = QVBoxLayout()
        self.video_target_fps_label = QLabel("Target FPS:")
        self.video_target_fps_label.setAlignment(Qt.AlignCenter)
        target_fps_layout.addWidget(self.video_target_fps_label)
        self.video_target_fps_spin = QDoubleSpinBox()
        self.video_target_fps_spin.setRange(1, 60)
        self.video_target_fps_spin.setDecimals(0)
        self.video_target_fps_spin.setSingleStep(1)
        self.video_target_fps_spin.setValue(30)
        self.video_target_fps_spin.setToolTip(
            "How many frames per second to extract from the video.\n"
            "Higher = more detail but slower processing"
        )
        target_fps_layout.addWidget(self.video_target_fps_spin, alignment=Qt.AlignCenter)
        controls_layout.addLayout(target_fps_layout)
        # Playback and Recording Controls
        actions_layout = QVBoxLayout()
        self.video_play_btn = QPushButton("Play")
        self.video_play_btn.setToolTip("Play the loaded video")
        self.video_play_btn.clicked.connect(self.on_video_play)
        actions_layout.addWidget(self.video_play_btn)
        self.video_pause_btn = QPushButton("Pause")
        self.video_pause_btn.setToolTip("Pause video playback")
        self.video_pause_btn.clicked.connect(self.on_video_pause)
        actions_layout.addWidget(self.video_pause_btn)
        continuous_layout = QHBoxLayout()
        continuous_label = QLabel("Continuous:")
        self.video_continuous_checkbox = QCheckBox()
        self.video_continuous_checkbox.setToolTip(
            "Continuously analyze the video in batches.\n"
            "Unchecked: Analyze once and stop.\n"
            "Checked: Keep analyzing as video plays."
        )
        continuous_layout.addWidget(continuous_label)
        continuous_layout.addWidget(self.video_continuous_checkbox)
        actions_layout.addLayout(continuous_layout)
        self.video_start_recording_btn = QPushButton("Start Analysis")
        self.video_start_recording_btn.setToolTip("Start analyzing the video for defects")
        self.video_start_recording_btn.clicked.connect(self.on_video_start_recording)
        actions_layout.addWidget(self.video_start_recording_btn)
        self.video_stop_recording_btn = QPushButton("Stop Analysis")
        self.video_stop_recording_btn.setToolTip("Stop analyzing the video")
        self.video_stop_recording_btn.clicked.connect(self.on_video_stop_recording)
        actions_layout.addWidget(self.video_stop_recording_btn)
        controls_layout.addLayout(actions_layout)
        # Video progress slider and preview panel
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setValue(0)
        self.video_slider.sliderPressed.connect(self.on_video_slider_pressed)
        self.video_slider.sliderReleased.connect(self.on_video_slider_released)
        self.video_slider.sliderMoved.connect(self.on_video_slider_moved)
        layout.addWidget(self.video_slider)
        self.video_slider_preview_panel = QWidget(self.load_video_tab)
        self.video_slider_preview_panel.setVisible(False)
        self.video_slider_preview_panel.setStyleSheet("background-color: rgba(0,0,0,200); border: 1px solid white;")
        preview_layout = QVBoxLayout(self.video_slider_preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.video_slider_preview_label = QLabel()
        self.video_slider_preview_label.setFixedSize(160, 90)
        self.video_slider_preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.video_slider_preview_label)
        self.video_slider_timestamp_label = QLabel()
        self.video_slider_timestamp_label.setAlignment(Qt.AlignCenter)
        self.video_slider_timestamp_label.setStyleSheet("color: white;")
        preview_layout.addWidget(self.video_slider_timestamp_label)
        # Initialize video-related variables for Load Video tab
        self.video_cap = None
        self.video_preview_cap = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_playing = False
        self.video_recording = False
        self.video_frame_queue = []
        self.video_batch_size = 0
        self.video_frame_interval = 1
        self.loaded_images = []      # Will hold frames loaded from a folder.
        self.source_type = None      # To track if we are using a video file or images from a folder.

    def on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Images")
        if folder:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            self.loaded_images = []
            # Optionally sort the files for a predictable order.
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith(image_extensions):
                    filepath = os.path.join(folder, filename)
                    frame = cv2.imread(filepath)
                    if frame is not None:
                        self.loaded_images.append(frame)
            if self.loaded_images:
                self.source_type = 'folder'
                # Set slider maximum to the number of loaded images minus one.
                self.video_slider.setMaximum(len(self.loaded_images) - 1)
                # Display the first image as a preview.
                self.display_loaded_image(0)
                print(f"Loaded {len(self.loaded_images)} images from {folder}.")
            else:
                print("No valid image files found in the folder.")

    def display_loaded_image(self, index):
        if 0 <= index < len(self.loaded_images):
            frame = self.loaded_images[index]
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            qimg = QImage(display_frame.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.video_display_label.setPixmap(
                pix.scaled(self.video_display_label.width(), self.video_display_label.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def on_browse_video(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if fileName:
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(fileName)
            if self.video_preview_cap is not None:
                self.video_preview_cap.release()
            self.video_preview_cap = cv2.VideoCapture(fileName)
            if self.video_cap.isOpened():
                total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.setMaximum(total_frames)
                self.video_slider.setValue(0)
            else:
                print("Failed to open video file.")

    def on_video_slider_pressed(self):
        self.video_slider_preview_panel.setVisible(True)

    def on_video_slider_released(self):
        self.video_slider_preview_panel.setVisible(False)
        if self.video_cap is not None:
            frame_number = self.video_slider.value()
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.update_video_frame()

    def on_video_slider_moved(self, value):
        # If a folder of images is loaded, display the corresponding image.
        if self.source_type == 'folder' and self.loaded_images:
            self.display_loaded_image(value)
            # Optionally update timestamp label (here just showing the frame index)
            self.video_slider_timestamp_label.setText(f"Image {value+1}/{len(self.loaded_images)}")
        else:
            # Existing behavior for video preview using self.video_cap.
            slider = self.video_slider
            ratio = (value - slider.minimum()) / (slider.maximum() - slider.minimum()) if slider.maximum() > slider.minimum() else 0
            slider_pos = slider.pos()
            panel_width = self.video_slider_preview_panel.width() or 160
            x_offset = int(slider.width() * ratio) - panel_width // 2
            x = slider_pos.x() + max(0, min(x_offset, slider.width() - panel_width))
            y = slider_pos.y() - self.video_slider_preview_panel.height() - 10
            self.video_slider_preview_panel.move(x, y)
            cap = self.video_preview_cap if self.video_preview_cap is not None else self.video_cap
            if cap is None:
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                pix = pix.scaled(self.video_slider_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_slider_preview_label.setPixmap(pix)
                fps = cap.get(cv2.CAP_PROP_FPS)
                seconds = value / fps if fps > 0 else 0
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                time_str = f"{mins:02d}:{secs:02d}"
                self.video_slider_timestamp_label.setText(time_str)

    def on_video_play(self):
        if self.video_cap is None:
            print("No video loaded.")
            return
        self.video_playing = True
        self.video_timer.start(30)

    def on_video_pause(self):
        self.video_playing = False
        self.video_timer.stop()

    def on_video_start_recording(self):
        if self.video_cap is None:
            print("No video loaded.")
            return
        self.video_recording = True
        self.video_frame_queue = []
        video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        target_fps = self.video_target_fps_spin.value()
        if target_fps > video_fps:
            target_fps = video_fps
        self.video_frame_interval = int(round(video_fps / target_fps))
        duration = self.video_duration_spin.value()
        self.video_batch_size = int(target_fps * duration)
        self.video_collection_progress_bar.setMaximum(self.video_batch_size)
        print(f"Video recording started. Batch size = {self.video_batch_size}, frame interval = {self.video_frame_interval}")

    def on_video_stop_recording(self):
        self.video_recording = False
        self.video_frame_queue = []
        print("Video recording stopped.")

    def update_video_frame(self):
        if self.video_cap is None or not self.video_playing:
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_timer.stop()
            self.video_playing = False
            return
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.video_slider.setValue(current_frame)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        qimg = QImage(display_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_display_label.setPixmap(
            pix.scaled(self.video_display_label.width(), self.video_display_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        if self.video_recording:
            current_frame_number = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame_number % self.video_frame_interval == 0:
                self.video_frame_queue.append(frame.copy())
                current_progress = min(len(self.video_frame_queue), self.video_batch_size)
                self.video_collection_progress_bar.setValue(current_progress)
                if len(self.video_frame_queue) >= self.video_batch_size:
                    frames_for_inference = self.video_frame_queue[:self.video_batch_size]
                    self.video_frame_queue = self.video_frame_queue[self.video_batch_size:]
                    if self.soft_memory_frames:
                        frames_for_inference.extend(self.soft_memory_frames)
                    self.runVideoInference(frames_for_inference)
                    self.video_collection_progress_bar.setValue(0)

    def runVideoInference(self, frames_for_inference):
        if not frames_for_inference or self.model is None:
            return
        self.video_inference_progress_bar.setValue(0)
        self.video_inference_thread = InferenceThread(self.model, frames_for_inference)
        self.video_inference_thread.progressChanged.connect(self.updateVideoInferenceProgress)
        self.video_inference_thread.inferenceFinished.connect(self.onVideoInferenceFinished)
        self.video_inference_thread.start()

    @pyqtSlot(int)
    def updateVideoInferenceProgress(self, value):
        self.video_inference_progress_bar.setValue(value)

    @pyqtSlot(list, np.ndarray, float, float)
    def onVideoInferenceFinished(self, frames, anomaly_maps, max_score, elapsed_seconds):
        # Calculate number of frames and time per frame (in seconds and milliseconds)
        num_frames = len(frames)
        per_image_time = elapsed_seconds / num_frames if num_frames > 0 else 0.0
        per_image_time_ms = per_image_time * 1000  # convert seconds to milliseconds

        # Update label with both total and per-image analysis time.
        self.video_inference_time_label.setText(
            f"Total Analysis Time: {elapsed_seconds:.2f} s | Time per Image: {per_image_time_ms:.2f} ms"
        )

        thresholds = self.config_data.get("thresholds", {})
        frame_threshold = thresholds.get("image_threshold", 9.0) / 10.0
        overlay_threshold = thresholds.get("overlay_threshold", 3.0) / 10.0

        if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
            anomaly_maps = anomaly_maps.squeeze(1)

        anomaly_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
        min_score = anomaly_scores.min()
        avg_score = anomaly_scores.mean()
        self.anomaly_scores_history.append(avg_score)
        rolling_avg = np.mean(self.anomaly_scores_history)

        # Convert to percentages
        min_percent = min_score * 100
        avg_percent = rolling_avg * 100
        max_percent = max_score * 100

        self.video_anomaly_label.setText(
            f"Defect Level - Min: {min_percent:.1f}%  Avg: {avg_percent:.1f}%  Max: {max_percent:.1f}%"
        )

        high_anomaly_indices = np.where(anomaly_scores > frame_threshold)[0]
        for idx in high_anomaly_indices:
            raw_frame_bgr = frames[idx]
            anomaly_map = anomaly_maps[idx]
            if (anomaly_map.shape[0] != raw_frame_bgr.shape[0] or 
                anomaly_map.shape[1] != raw_frame_bgr.shape[1]):
                anomaly_map = cv2.resize(
                    anomaly_map, (raw_frame_bgr.shape[1], raw_frame_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            denom = (map_max - map_min) + 1e-8
            anomaly_map_float = (anomaly_map - map_min) / denom
            mask = (anomaly_map_float > overlay_threshold).astype(np.uint8)
            mask_3 = np.dstack([mask, mask, mask])
            anomaly_map_8bit = (anomaly_map_float * 255).astype(np.uint8)
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_8bit, cv2.COLORMAP_JET)
            alpha = 0.3
            blended = cv2.addWeighted(raw_frame_bgr, 1 - alpha, anomaly_map_colored, alpha, 0)
            overlay = np.where(mask_3 == 1, blended, raw_frame_bgr)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = overlay_rgb.shape
            qimg_overlay = QImage(overlay_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix_overlay = QPixmap.fromImage(qimg_overlay)
            now = datetime.now()
            day_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")
            if day_str not in self.anomalies_by_day:
                self.anomalies_by_day[day_str] = {}
            if hour_str not in self.anomalies_by_day[day_str]:
                self.anomalies_by_day[day_str][hour_str] = []
            self.anomalies_by_day[day_str][hour_str].append(
                (pix_overlay, overlay.copy(), raw_frame_bgr.copy(), now, anomaly_scores[idx])
            )
        self.refresh_anomaly_sidebar()
        if not self.video_continuous_checkbox.isChecked():
            self.video_recording = False



    # New: Start Folder Inference method for loaded folder images.
    def runFolderInference(self):
        if not self.loaded_images:
            print("No folder images loaded.")
            return
        if self.model is None:
            print("No model loaded; cannot run inference.")
            return
        # Make a copy of the loaded images
        frames_for_inference = self.loaded_images.copy()
        # Append images from soft memory if any exist.
        if self.soft_memory_frames:
            frames_for_inference.extend(self.soft_memory_frames)
        self.video_inference_progress_bar.setValue(0)
        self.video_inference_thread = InferenceThread(self.model, frames_for_inference)
        self.video_inference_thread.progressChanged.connect(self.updateVideoInferenceProgress)
        self.video_inference_thread.inferenceFinished.connect(self.onVideoInferenceFinished)
        self.video_inference_thread.start()


    # ------------------------------
    # Additional methods for live camera mode.
    # ------------------------------
    def capture_still(self):
        if self.cap is None:
            QMessageBox.warning(self, "Camera Error", "No camera connected. Please connect to a camera first.")
            return
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                self.saved_images_data.append((pix, frame.copy()))
                self.refresh_saved_sidebar()
            else:
                print("Failed to read frame for capture.")
        else:
            print("Capture device is not open.")

    def open_config_dialog(self):
        dlg = ConfigDialog(self.config_data, self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                if torch.cuda.is_available():
                    dev_index = self.config_data.get('device', 0)
                    self.config_data['device'] = dev_index
                else:
                    self.config_data['device'] = 'cpu'

                # Update status to show we're loading model
                self.update_status("analyzing", "Loading AI model...")

                self.model = MuSc(self.config_data)

                # Update status bar with new model name
                model_name = self.config_data.get("models", {}).get("backbone_name", "Unknown")
                model_display = self._get_model_display_name(model_name)
                self.status_center.setText(f"AI Model: {model_display}")

                self.update_status("ready")
            except Exception as e:
                print("Error re-initializing model:", e)
                self.update_status("error", f"Failed to load model: {str(e)}")

    def startRecording(self):
        if self.cap is None:
            QMessageBox.warning(self, "Camera Error", "No camera connected. Please connect to a camera first.")
            self.update_status("error", "No camera connected")
            return
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Camera is not opened. Please reconnect to your camera.")
            self.update_status("error", "Camera not opened")
            return
        if self.model is None:
            print("No model loaded; cannot start inference.")
            self.update_status("error", "AI model not loaded")
            return
        self.recording = True
        self.update_status("collecting")
        # Determine the camera FPS; if unavailable, default to 30.
        camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30
        # Get user-selected target FPS from the live tab.
        target_fps = self.live_target_fps_spin.value()
        if target_fps > camera_fps:
            target_fps = camera_fps
        self.live_frame_interval = int(round(camera_fps / target_fps))
        self.batch_size = int(target_fps * self.duration_spin.value())
        self.frame_queue = []
        self.collection_progress_bar.setMaximum(self.batch_size)
        self.live_frame_counter = 0
        print(f"Recording started. Batch size = {self.batch_size}, frame interval = {self.live_frame_interval}")

    def stopRecording(self):
        self.recording = False
        self.frame_queue = []
        self.inference_in_progress = False
        self.update_status("ready")
        print("Recording / inference stopped.")

    def connect_to_camera(self, source):
        """Connect to a camera source (USB device index or IP camera URL)."""
        # Update status to show we're trying to connect
        self.update_status("connecting")

        # Release existing camera if any
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            self.cap = cv2.VideoCapture(source)
            # Give the camera a moment to initialize
            pytime.sleep(0.5)

            # Test if the camera is actually opened
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    self.camera_source = source
                    if isinstance(source, int):
                        self.connection_status_label.setText(f"Status: Connected to USB Camera (Device {source})")
                        self.connection_status_label.setStyleSheet("color: green; font-weight: bold;")
                    else:
                        self.connection_status_label.setText(f"Status: Connected to IP Camera")
                        self.connection_status_label.setStyleSheet("color: green; font-weight: bold;")
                    print(f"Successfully connected to camera: {source}")

                    # Update status bar
                    self.update_status("connected")
                    self.update_camera_status(source, connected=True)
                    return True
                else:
                    raise Exception("Could not read frame from camera")
            else:
                raise Exception("Failed to open camera")
        except Exception as e:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_source = None
            self.connection_status_label.setText(f"Status: Connection Failed")
            self.connection_status_label.setStyleSheet("color: red; font-weight: bold;")

            # Update status bar to show error
            self.update_status("error", "Camera connection failed")
            self.update_camera_status(None, connected=False)

            QMessageBox.warning(self, "Camera Connection Error",
                              f"Failed to connect to camera:\n{str(e)}\n\nPlease check your camera source and try again.")
            print(f"Failed to connect to camera {source}: {e}")
            return False

    def connect_usb_camera(self):
        """Connect to a USB camera using the selected device index."""
        device_index = self.usb_device_spin.value()
        self.connect_to_camera(device_index)

    def connect_ip_camera(self):
        """Connect to an IP camera using the provided URL."""
        ip_url = self.ip_camera_input.text().strip()
        if not ip_url:
            QMessageBox.warning(self, "Input Error", "Please enter an IP camera URL.")
            return

        # Validate URL format (basic check)
        if not (ip_url.startswith('rtsp://') or ip_url.startswith('http://') or
                ip_url.startswith('https://') or ip_url.startswith('tcp://')):
            QMessageBox.warning(self, "Input Error",
                              "Invalid URL format. URL should start with rtsp://, http://, https://, or tcp://")
            return

        self.connect_to_camera(ip_url)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.latest_frame = frame.copy()
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        qimg = QImage(display_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        if self.recording:
            self.live_frame_counter += 1
            if self.live_frame_counter % self.live_frame_interval == 0:
                self.frame_queue.append(frame.copy())
                current_progress = min(len(self.frame_queue), self.batch_size)
                self.collection_progress_bar.setValue(current_progress)
                if not self.inference_in_progress and len(self.frame_queue) >= self.batch_size:
                    frames_for_inference = self.frame_queue[:self.batch_size]
                    self.frame_queue = self.frame_queue[self.batch_size:]
                    if self.soft_memory_frames:
                        frames_for_inference.extend(self.soft_memory_frames)
                    self.runInference(frames_for_inference)
                    self.collection_progress_bar.setValue(0)

    def runInference(self, frames_for_inference):
        if not frames_for_inference or self.model is None:
            return
        self.inference_in_progress = True
        self.update_status("analyzing")
        self.inference_progress_bar.setValue(0)
        self.inference_thread = InferenceThread(self.model, frames_for_inference)
        self.inference_thread.progressChanged.connect(self.updateInferenceProgress)
        self.inference_thread.inferenceFinished.connect(self.onInferenceFinished)
        self.inference_thread.start()

    @pyqtSlot(int)
    def updateInferenceProgress(self, value):
        self.inference_progress_bar.setValue(value)

    @pyqtSlot(list, np.ndarray, float, float)
    def onInferenceFinished(self, frames, anomaly_maps, max_score, elapsed_seconds):
        self.inference_in_progress = False
        user_duration = self.duration_spin.value()
        if elapsed_seconds > user_duration:
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
        
        thresholds = self.config_data.get("thresholds", {})
        frame_threshold = thresholds.get("image_threshold", 9.0) / 10.0
        overlay_threshold = thresholds.get("overlay_threshold", 3.0) / 10.0

        # Calculate number of frames and the time per image.
        num_frames = len(frames)
        per_image_time = elapsed_seconds / num_frames if num_frames > 0 else 0.0
        per_image_time_ms = per_image_time * 1000

        # Compute anomaly scores for each frame in the batch.
        anomaly_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)
        min_score = anomaly_scores.min()
        avg_score = anomaly_scores.mean()
        self.anomaly_scores_history.append(avg_score)
        rolling_avg = np.mean(self.anomaly_scores_history)

        # Convert scores to percentages for better understanding
        max_percent = max_score * 100
        min_percent = min_score * 100
        avg_percent = rolling_avg * 100

        self.anomaly_label.setText(
            f"Defect Level - Max: {max_percent:.1f}%  Min: {min_percent:.1f}%  Avg: {avg_percent:.1f}%"
        )

        # Update analysis time label with both total and per-image times.
        self.inference_time_label.setText(
            f"Total Analysis Time: {elapsed_seconds:.2f} s | Time per Image: {per_image_time_ms:.2f} ms"
        )

        high_anomaly_indices = np.where(anomaly_scores > frame_threshold)[0]
        for idx in high_anomaly_indices:
            raw_frame_bgr = frames[idx]
            anomaly_map = anomaly_maps[idx]
            if (anomaly_map.shape[0] != raw_frame_bgr.shape[0] or 
                anomaly_map.shape[1] != raw_frame_bgr.shape[1]):
                anomaly_map = cv2.resize(
                    anomaly_map, (raw_frame_bgr.shape[1], raw_frame_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            map_min, map_max = anomaly_map.min(), anomaly_map.max()
            denom = (map_max - map_min) + 1e-8
            anomaly_map_float = (anomaly_map - map_min) / denom
            mask = (anomaly_map_float > overlay_threshold).astype(np.uint8)
            mask_3 = np.dstack([mask, mask, mask])
            anomaly_map_8bit = (anomaly_map_float * 255).astype(np.uint8)
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_8bit, cv2.COLORMAP_JET)
            alpha = 0.3
            blended = cv2.addWeighted(raw_frame_bgr, 1 - alpha, anomaly_map_colored, alpha, 0)
            overlay = np.where(mask_3 == 1, blended, raw_frame_bgr)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = overlay_rgb.shape
            qimg_overlay = QImage(overlay_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix_overlay = QPixmap.fromImage(qimg_overlay)
            now = datetime.now()
            day_str = now.strftime("%Y-%m-%d")
            hour_str = now.strftime("%H")
            if day_str not in self.anomalies_by_day:
                self.anomalies_by_day[day_str] = {}
            if hour_str not in self.anomalies_by_day[day_str]:
                self.anomalies_by_day[day_str][hour_str] = []
            self.anomalies_by_day[day_str][hour_str].append(
                (pix_overlay, overlay.copy(), raw_frame_bgr.copy(), now, anomaly_scores[idx])
            )
        self.refresh_anomaly_sidebar()
        if not self.continuous_checkbox.isChecked():
            self.recording = False
            self.update_status("complete")
        else:
            self.update_status("collecting")  # Back to collecting for next batch

    def closeEvent(self, event):
        """Clean up resources when closing the window."""
        # Stop timers
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'video_timer'):
            self.video_timer.stop()

        # Release cameras
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'video_cap') and self.video_cap is not None:
            self.video_cap.release()
        if hasattr(self, 'video_preview_cap') and self.video_preview_cap is not None:
            self.video_preview_cap.release()

        event.accept()

def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(16)
    app.setFont(font)
    yaml_path = os.path.join(os.getcwd(), r"C:\Users\dylan\MuSc_original\MuSc\configs\musc.yaml")
    win = MainWindow(yaml_path)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()