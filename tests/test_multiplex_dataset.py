"""Tests for MultiplexImagingDataset."""

import numpy as np
import pytest
import torch
from pathlib import Path

TISSUE_ID = "lin2023high_aijcgfvk_0000"

pytestmark = pytest.mark.skipif(
    not Path("/mnt/aimm/scratch/datasets_v2/lin2023high").exists(),
    reason="Dataset not found",
)


class TestMxTissueIds:
    def test_nonempty(self, mx_dataset):
        ids = mx_dataset.get_tissue_ids()
        assert len(ids) > 0


class TestMxChannelList:
    def test_columns(self, mx_dataset):
        cl = mx_dataset.channel_list
        assert "channel_name" in cl.columns
        assert "qc_pass" in cl.columns

    def test_channel_count(self, mx_dataset):
        assert len(mx_dataset.channel_list) == 19


class TestMxGetTissueComplete:
    def test_shape_and_type(self, mx_dataset):
        tissue = mx_dataset.get_tissue(TISSUE_ID, kind="complete", preprocess=False)
        assert isinstance(tissue.tissue, torch.Tensor)
        C, H, W = tissue.tissue.shape
        assert C > 0 and H > 0 and W > 0
        assert tissue.measured_mask is not None
        assert tissue.image_loading_mask is not None

    def test_channel_names_consistent(self, mx_dataset):
        tissue = mx_dataset.get_tissue(TISSUE_ID, kind="complete", preprocess=False)
        assert len(tissue.channel_names) == tissue.tissue.shape[0]


class TestMxGetTissueQcFiltered:
    def test_fewer_or_equal_channels(self, mx_dataset):
        complete = mx_dataset.get_tissue(TISSUE_ID, kind="complete", preprocess=False)
        qc = mx_dataset.get_tissue(TISSUE_ID, kind="qc_filtered", preprocess=False)
        assert qc.tissue.shape[0] <= complete.tissue.shape[0]

    def test_channel_names_consistent(self, mx_dataset):
        tissue = mx_dataset.get_tissue(TISSUE_ID, kind="qc_filtered", preprocess=False)
        assert len(tissue.channel_names) == tissue.tissue.shape[0]


class TestMxGetTissueFiltered:
    def test_fewer_or_equal_to_qc(self, mx_dataset):
        qc = mx_dataset.get_tissue(TISSUE_ID, kind="qc_filtered", preprocess=False)
        filtered = mx_dataset.get_tissue(TISSUE_ID, kind="filtered", preprocess=False)
        assert filtered.tissue.shape[0] <= qc.tissue.shape[0]


class TestMxNormalization:
    def test_identity_preserves_values(self, mx_dataset):
        raw = mx_dataset.get_tissue(TISSUE_ID, kind="complete", preprocess=False)
        preprocessed = mx_dataset.get_tissue(TISSUE_ID, kind="complete", preprocess=True)
        # Identity normalizer should just convert to float tensor
        assert torch.allclose(raw.tissue.float(), preprocessed.tissue, atol=1e-5)


class TestMxInvalidKind:
    def test_raises(self, mx_dataset):
        with pytest.raises(ValueError):
            mx_dataset.get_tissue(TISSUE_ID, kind="invalid")


class TestMxGetCrop:
    def test_crop_shape(self, mx_dataset):
        crop = mx_dataset.get_crop(TISSUE_ID, crop_id=0, preprocess=False, kind="complete")
        C, H, W = crop.tissue.shape
        assert H == 224 and W == 224
        assert C > 0
