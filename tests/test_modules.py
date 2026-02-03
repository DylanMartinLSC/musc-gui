"""
Unit tests for MuSc algorithm modules (LNAMD and MSM).

These tests verify the core algorithm components work correctly
with synthetic data, without requiring real images or GPU.

Run with: pytest tests/test_modules.py -v
"""

import pytest
import torch
import numpy as np


class TestLNAMD:
    """Tests for Local Neighborhood Aggregation with Multi-scale Distance module."""

    def test_lnamd_import(self):
        """Test that LNAMD can be imported."""
        from models.modules._LNAMD import LNAMD
        assert LNAMD is not None

    def test_lnamd_initialization(self, device):
        """Test LNAMD initialization with default parameters."""
        from models.modules._LNAMD import LNAMD

        lnamd = LNAMD(
            device=device,
            r=1,
            feature_dim=768,
            feature_layer=[1],
        )
        assert lnamd is not None
        assert lnamd.r == 1

    def test_lnamd_initialization_multi_layer(self, device):
        """Test LNAMD with multiple feature layers."""
        from models.modules._LNAMD import LNAMD

        lnamd = LNAMD(
            device=device,
            r=3,
            feature_dim=1024,
            feature_layer=[1, 2, 3, 4],
        )
        assert lnamd is not None
        assert len(lnamd.LNA.preprocessing_modules) == 4

    def test_lnamd_embed_shape(self, device):
        """Test LNAMD _embed output shape."""
        from models.modules._LNAMD import LNAMD

        batch_size = 2
        num_patches = 196 + 1  # 14x14 + CLS token
        feature_dim = 768

        lnamd = LNAMD(
            device=device,
            r=1,
            feature_dim=feature_dim,
            feature_layer=[1],
        )

        # Create synthetic patch tokens
        torch.manual_seed(42)
        patch_tokens = [
            torch.rand(batch_size, num_patches, feature_dim, device=device)
        ]

        # Run embedding
        features = lnamd._embed(patch_tokens)

        # Check output shape: [B, num_patches, num_layers, feature_dim]
        assert features.ndim == 4
        assert features.shape[0] == batch_size
        assert features.shape[3] == feature_dim

    def test_lnamd_embed_multi_layer(self, device):
        """Test LNAMD embedding with multiple layers."""
        from models.modules._LNAMD import LNAMD

        batch_size = 2
        num_patches = 196 + 1
        feature_dim = 768
        num_layers = 4

        lnamd = LNAMD(
            device=device,
            r=1,
            feature_dim=feature_dim,
            feature_layer=list(range(1, num_layers + 1)),
        )

        # Create synthetic patch tokens for multiple layers
        torch.manual_seed(42)
        patch_tokens = [
            torch.rand(batch_size, num_patches, feature_dim, device=device)
            for _ in range(num_layers)
        ]

        features = lnamd._embed(patch_tokens)

        # Check that layer dimension matches
        assert features.shape[2] == num_layers

    def test_lnamd_with_aggregation_radius(self, device):
        """Test LNAMD with different aggregation radii."""
        from models.modules._LNAMD import LNAMD

        batch_size = 2
        num_patches = 196 + 1
        feature_dim = 768

        for r in [1, 3]:
            lnamd = LNAMD(
                device=device,
                r=r,
                feature_dim=feature_dim,
                feature_layer=[1],
            )

            torch.manual_seed(42)
            patch_tokens = [
                torch.rand(batch_size, num_patches, feature_dim, device=device)
            ]

            features = lnamd._embed(patch_tokens)
            assert features is not None
            assert features.ndim == 4


class TestMSM:
    """Tests for Mutual Scoring Module."""

    def test_msm_import(self):
        """Test that MSM can be imported."""
        from models.modules._MSM import MSM, compute_scores_fast, compute_scores_slow
        assert MSM is not None
        assert compute_scores_fast is not None
        assert compute_scores_slow is not None

    def test_msm_basic(self, device):
        """Test MSM with minimal synthetic data."""
        from models.modules._MSM import MSM

        # Create minimal batch for testing
        batch_size = 3  # Need at least 3 images for mutual scoring
        num_patches = 16  # Small patch count for fast test
        feature_dim = 64  # Small feature dim for fast test

        torch.manual_seed(42)
        Z = torch.rand(batch_size, num_patches, feature_dim, device=device)

        # Run MSM
        anomaly_scores = MSM(Z, device, topmin_min=0, topmin_max=0.3)

        # Check output shape
        assert anomaly_scores.shape == (batch_size, num_patches)

    def test_msm_output_range(self, device):
        """Test that MSM outputs are non-negative (distances)."""
        from models.modules._MSM import MSM

        batch_size = 4
        num_patches = 16
        feature_dim = 32

        torch.manual_seed(42)
        Z = torch.rand(batch_size, num_patches, feature_dim, device=device)

        anomaly_scores = MSM(Z, device, topmin_min=0, topmin_max=0.3)

        # Scores should be non-negative (they are distances)
        assert (anomaly_scores >= 0).all()

    def test_msm_identical_images(self, device):
        """Test MSM with identical images (should have low scores)."""
        from models.modules._MSM import MSM

        batch_size = 4
        num_patches = 16
        feature_dim = 32

        # Create identical images
        torch.manual_seed(42)
        base_features = torch.rand(1, num_patches, feature_dim, device=device)
        Z = base_features.repeat(batch_size, 1, 1)

        anomaly_scores = MSM(Z, device, topmin_min=0, topmin_max=0.3)

        # All images are identical, so scores should be very low
        assert anomaly_scores.max() < 1e-5

    def test_msm_with_outlier(self, device):
        """Test MSM with one outlier image."""
        from models.modules._MSM import MSM

        batch_size = 5
        num_patches = 16
        feature_dim = 32

        torch.manual_seed(42)
        # Create similar images
        base_features = torch.rand(1, num_patches, feature_dim, device=device)
        Z = base_features.repeat(batch_size, 1, 1)

        # Add noise to one image to make it an outlier
        Z[0] = Z[0] + torch.rand_like(Z[0]) * 5.0

        anomaly_scores = MSM(Z, device, topmin_min=0, topmin_max=0.3)

        # The outlier (index 0) should have higher scores
        outlier_max = anomaly_scores[0].max()
        normal_max = anomaly_scores[1:].max()
        assert outlier_max > normal_max

    def test_compute_scores_fast_shape(self, device):
        """Test compute_scores_fast output shape."""
        from models.modules._MSM import compute_scores_fast

        batch_size = 4
        num_patches = 16
        feature_dim = 32

        torch.manual_seed(42)
        Z = torch.rand(batch_size, num_patches, feature_dim, device=device)

        # Get scores for first image
        scores = compute_scores_fast(Z, 0, device)
        assert scores.shape == (num_patches,)

    def test_compute_scores_fast_invalid_shape(self, device):
        """Test compute_scores_fast raises error for invalid input."""
        from models.modules._MSM import compute_scores_fast

        # 2D tensor (missing feature dimension)
        torch.manual_seed(42)
        Z_invalid = torch.rand(4, 16, device=device)

        with pytest.raises(ValueError, match="Expected Z to have 3 dimensions"):
            compute_scores_fast(Z_invalid, 0, device)


class TestPatchMaker:
    """Tests for PatchMaker helper class."""

    def test_patchmaker_import(self):
        """Test PatchMaker import."""
        from models.modules._LNAMD import PatchMaker
        assert PatchMaker is not None

    def test_patchmaker_initialization(self):
        """Test PatchMaker initialization."""
        from models.modules._LNAMD import PatchMaker

        pm = PatchMaker(patchsize=3, stride=1)
        assert pm.patchsize == 3
        assert pm.stride == 1

    def test_patchmaker_patchify(self, device):
        """Test PatchMaker patchify operation."""
        from models.modules._LNAMD import PatchMaker

        pm = PatchMaker(patchsize=3, stride=1)

        # Create input features [B, C, H, W]
        features = torch.rand(2, 768, 14, 14, device=device)

        # Patchify
        patches = pm.patchify(features)

        # Check output has correct structure
        assert patches.ndim == 5  # [B, num_patches, C, patch_h, patch_w]

    def test_patchmaker_with_spatial_info(self, device):
        """Test PatchMaker returns spatial info when requested."""
        from models.modules._LNAMD import PatchMaker

        pm = PatchMaker(patchsize=3, stride=1)
        features = torch.rand(2, 768, 14, 14, device=device)

        patches, spatial_info = pm.patchify(features, return_spatial_info=True)

        assert patches is not None
        assert spatial_info is not None
        assert len(spatial_info) == 2  # (height, width)


class TestPreprocessing:
    """Tests for Preprocessing and MeanMapper helper classes."""

    def test_preprocessing_import(self):
        """Test Preprocessing import."""
        from models.modules._LNAMD import Preprocessing, MeanMapper
        assert Preprocessing is not None
        assert MeanMapper is not None

    def test_meanmapper_forward(self, device):
        """Test MeanMapper forward pass."""
        from models.modules._LNAMD import MeanMapper

        output_dim = 768
        mm = MeanMapper(output_dim).to(device)

        # Input: [B, patch_h * patch_w * C] flattened patch features
        batch_size = 2
        input_features = torch.rand(batch_size, 3 * 3 * 768, device=device)

        output = mm(input_features)
        assert output.shape == (batch_size, output_dim)

    def test_preprocessing_forward(self, device):
        """Test Preprocessing forward pass."""
        from models.modules._LNAMD import Preprocessing

        input_layers = [1, 2]  # Two layers
        output_dim = 768
        pp = Preprocessing(input_layers, output_dim).to(device)

        # Input: list of feature tensors
        batch_size = 2
        features = [
            torch.rand(batch_size, 3 * 3 * 768, device=device),
            torch.rand(batch_size, 3 * 3 * 768, device=device),
        ]

        output = pp(features)
        # Output: [B, num_layers, output_dim]
        assert output.shape == (batch_size, len(input_layers), output_dim)


class TestModuleIntegration:
    """Integration tests for LNAMD + MSM pipeline."""

    def test_lnamd_to_msm_pipeline(self, device):
        """Test the full LNAMD → MSM pipeline."""
        from models.modules._LNAMD import LNAMD
        from models.modules._MSM import MSM

        # Configuration
        batch_size = 4
        num_patches = 196 + 1  # 14x14 + CLS
        feature_dim = 768
        r = 1

        # Create LNAMD
        lnamd = LNAMD(
            device=device,
            r=r,
            feature_dim=feature_dim,
            feature_layer=[1],
        )

        # Create synthetic patch tokens
        torch.manual_seed(42)
        patch_tokens = [
            torch.rand(batch_size, num_patches, feature_dim, device=device)
        ]

        # Run LNAMD
        features = lnamd._embed(patch_tokens)
        features = features.to(device)
        features = features / features.norm(dim=-1, keepdim=True)

        # Squeeze layer dimension for MSM
        features = features.squeeze(2)  # [B, L, C]

        # Run MSM
        anomaly_scores = MSM(Z=features, device=device, topmin_min=0, topmin_max=0.3)

        # Verify output
        assert anomaly_scores.shape[0] == batch_size
        assert (anomaly_scores >= 0).all()
