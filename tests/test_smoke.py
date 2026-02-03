"""
Smoke tests for MuSc Industrial GUI.

These tests verify basic functionality without requiring GPU or large models.
Run with: pytest tests/test_smoke.py -v
"""

import os
import sys

import pytest
import torch
import numpy as np


class TestImports:
    """Test that all main modules can be imported."""

    def test_import_musc_model(self):
        """Test importing the main MuSc model class."""
        from models.musc import MuSc
        assert MuSc is not None

    def test_import_lnamd(self):
        """Test importing LNAMD module."""
        from models.modules._LNAMD import LNAMD
        assert LNAMD is not None

    def test_import_msm(self):
        """Test importing MSM module."""
        from models.modules._MSM import MSM
        assert MSM is not None

    def test_import_rscin(self):
        """Test importing RsCIN module."""
        from models.modules._RsCIN import RsCIN
        assert RsCIN is not None

    def test_import_backbones(self):
        """Test importing backbone loader."""
        import models.backbone._backbones as _backbones
        assert hasattr(_backbones, "load")

    def test_import_metrics(self):
        """Test importing metrics utilities."""
        from utils.metrics import compute_metrics
        assert compute_metrics is not None


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_default_config_structure(self, default_config):
        """Test that default config has required keys."""
        assert "datasets" in default_config
        assert "models" in default_config
        assert "device" in default_config
        assert "thresholds" in default_config

    def test_config_datasets(self, default_config):
        """Test datasets config section."""
        datasets = default_config["datasets"]
        assert "img_resize" in datasets
        assert "dataset_name" in datasets
        assert datasets["img_resize"] in [224, 256, 384, 512]

    def test_config_models(self, default_config):
        """Test models config section."""
        models = default_config["models"]
        assert "backbone_name" in models
        assert "batch_size" in models
        assert "feature_layers" in models
        assert isinstance(models["feature_layers"], list)


class TestTorchSetup:
    """Test PyTorch environment setup."""

    def test_torch_available(self):
        """Test that PyTorch is available."""
        assert torch is not None
        assert hasattr(torch, "__version__")

    def test_device_detection(self, device):
        """Test device detection works."""
        assert device.type in ["cuda", "cpu"]

    def test_tensor_creation(self, device):
        """Test basic tensor operations."""
        tensor = torch.zeros(1, 3, 224, 224, device=device)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.device.type == device.type


class TestSyntheticInference:
    """Test inference pipeline with synthetic data (no model weights)."""

    def test_image_preprocessing(self, sample_image_array):
        """Test image preprocessing pipeline."""
        # Simulate preprocessing
        img = sample_image_array
        assert img.shape == (224, 224, 3)

        # Normalize
        img_norm = img.astype(np.float32) / 255.0
        assert img_norm.max() <= 1.0
        assert img_norm.min() >= 0.0

        # Convert to tensor format [C, H, W]
        tensor = torch.tensor(img_norm).permute(2, 0, 1)
        assert tensor.shape == (3, 224, 224)

    def test_anomaly_map_shape(self, device):
        """Test expected anomaly map shapes."""
        # Simulated anomaly map [batch, height, width]
        batch_size = 2
        map_size = 14  # Typical patch grid for 224x224

        anomaly_map = torch.rand(batch_size, map_size, map_size, device=device)
        assert anomaly_map.shape == (batch_size, map_size, map_size)

        # Interpolation to original size
        import torch.nn.functional as F
        upsampled = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=True,
        )
        assert upsampled.shape == (batch_size, 1, 224, 224)


class TestOutputFormats:
    """Test output file formats and structures."""

    def test_heatmap_colormap(self):
        """Test heatmap color mapping."""
        import cv2

        # Create normalized anomaly scores
        anomaly_map = np.random.rand(224, 224).astype(np.float32)
        heatmap = (anomaly_map * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        assert colored.shape == (224, 224, 3)
        assert colored.dtype == np.uint8

    def test_score_normalization(self):
        """Test anomaly score normalization."""
        # Raw scores
        scores = np.array([0.1, 0.5, 0.9, 1.2])

        # Normalize to 0-1
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for model loading test",
)
class TestModelLoading:
    """Tests that require GPU and download model weights."""

    @pytest.mark.slow
    def test_musc_initialization(self, default_config):
        """Test MuSc model initialization (downloads weights)."""
        from models.musc import MuSc

        # This will download model weights on first run
        model = MuSc(default_config)
        assert model is not None
        assert model.device.type == "cuda"

    @pytest.mark.slow
    def test_single_image_inference(self, default_config, sample_image_tensor):
        """Test inference on a single synthetic image."""
        from models.musc import MuSc

        model = MuSc(default_config)
        with torch.no_grad():
            result = model.infer_on_images([sample_image_tensor])

        assert result is not None
        assert isinstance(result, np.ndarray)
        # Should be [1, H, W] or similar
        assert result.ndim in [2, 3]


class TestDemoScript:
    """Test the demo.py CLI script utilities."""

    def test_demo_config_loading(self):
        """Test demo config loading function."""
        from demo import load_config

        # Load default config
        cfg = load_config("nonexistent.yaml")
        assert "datasets" in cfg
        assert "models" in cfg

    def test_demo_image_loading(self, tmp_path):
        """Test demo image loading function."""
        import cv2
        from demo import load_image

        # Create a test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), test_img)

        # Load and preprocess
        original, tensor = load_image(str(img_path), target_size=224)
        assert original.shape == (100, 100, 3)
        assert tensor.shape == (1, 3, 224, 224)
