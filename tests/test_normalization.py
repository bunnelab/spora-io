"""Tests for normalization classes."""

import numpy as np
import pandas as pd
import pytest
import torch

from spatialprot_data.utils.dataset.normalize import (
    IdentityNormalizer,
    build_normalizer,
)
from spatialprot_data.utils.dataset.transforms import FilterFactory


@pytest.fixture
def dummy_channels_per_image():
    return pd.DataFrame(
        {"ch0": [True, True], "ch1": [True, False]},
        index=["tissue_a", "tissue_b"],
    )


class TestIdentityNormalizer:
    def test_passthrough_numpy(self, dummy_channels_per_image):
        norm = IdentityNormalizer(
            modality_dir="/tmp",
            channels_per_image=dummy_channels_per_image,
            verbose=False,
        )
        x = np.random.rand(3, 32, 32).astype(np.float32)
        out, refined = norm.apply(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 32, 32)
        assert refined is None
        assert torch.allclose(out, torch.from_numpy(x))

    def test_passthrough_tensor(self, dummy_channels_per_image):
        norm = IdentityNormalizer(
            modality_dir="/tmp",
            channels_per_image=dummy_channels_per_image,
            verbose=False,
        )
        x = torch.rand(5, 16, 16)
        out, _ = norm.apply(x)
        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, x.float())

    def test_ensure_tensor_numpy(self, dummy_channels_per_image):
        norm = IdentityNormalizer(
            modality_dir="/tmp",
            channels_per_image=dummy_channels_per_image,
            verbose=False,
        )
        x = np.array([1.0, 2.0, 3.0])
        t = norm._ensure_tensor(x)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_ensure_tensor_torch(self, dummy_channels_per_image):
        norm = IdentityNormalizer(
            modality_dir="/tmp",
            channels_per_image=dummy_channels_per_image,
            verbose=False,
        )
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        t = norm._ensure_tensor(x)
        assert t.dtype == torch.float32


class TestBuildNormalizer:
    def test_identity(self, dummy_channels_per_image):
        norm = build_normalizer(
            normalization="identity",
            modality_dir="/tmp",
            channels_per_image=dummy_channels_per_image,
        )
        assert isinstance(norm, IdentityNormalizer)

    def test_unknown_raises(self, dummy_channels_per_image):
        with pytest.raises(NotImplementedError):
            build_normalizer(
                normalization="unknown",
                modality_dir="/tmp",
                channels_per_image=dummy_channels_per_image,
            )

    def test_he_raises(self, dummy_channels_per_image):
        with pytest.raises(NotImplementedError):
            build_normalizer(
                normalization="he",
                modality_dir="/tmp",
                channels_per_image=dummy_channels_per_image,
            )


class TestFilterFactory:
    def test_gaussian_blur_changes_values(self):
        ff = FilterFactory(["gaussian_blur"], {})
        x = torch.rand(1, 32, 32)
        out = ff.apply_filters(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_empty_filters_noop(self):
        ff = FilterFactory([], {})
        x = torch.rand(3, 16, 16)
        out = ff.apply_filters(x)
        assert torch.allclose(out, x)
