"""
Pytest configuration and shared fixtures for MuSc tests.
"""

import os
import sys

import pytest
import torch
import numpy as np

# Add MuSc package to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def device():
    """Get the compute device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def default_config():
    """Default configuration for testing."""
    return {
        "datasets": {
            "img_resize": 224,
            "dataset_name": "mvtec_ad",
            "class_name": "ALL",
            "divide_num": 1,
            "data_path": "./data/",
        },
        "device": 0 if torch.cuda.is_available() else "cpu",
        "models": {
            "backbone_name": "dinov2_vitb14",
            "batch_size": 1,
            "feature_layers": [11],
            "pretrained": "openai",
            "r_list": [1],
        },
        "testing": {
            "output_dir": "output_test",
            "vis": False,
            "vis_type": "single_norm",
            "save_excel": False,
        },
        "thresholds": {
            "image_threshold": 9.0,
            "overlay_threshold": 3.5,
        },
    }


@pytest.fixture
def sample_image_tensor(device):
    """Create a synthetic test image tensor."""
    # Create a random image tensor [1, 3, 224, 224]
    torch.manual_seed(42)
    tensor = torch.rand(1, 3, 224, 224, device=device)
    return tensor


@pytest.fixture
def sample_image_array():
    """Create a synthetic test image as numpy array."""
    np.random.seed(42)
    # Create a random RGB image [224, 224, 3]
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_patch_tokens(device):
    """Create synthetic patch tokens for testing modules."""
    torch.manual_seed(42)
    batch_size = 2
    num_patches = 196 + 1  # 14x14 patches + 1 CLS token for 224x224 image with patch 16
    feature_dim = 768
    # Single layer patch tokens
    return torch.rand(batch_size, num_patches, feature_dim, device=device)
