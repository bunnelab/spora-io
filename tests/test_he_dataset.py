"""Tests for HEImagingDataset."""

import numpy as np
import pytest
import re
import torch
from pathlib import Path

DATASET_PATH = Path("/mnt/aimm/scratch/datasets_v2/lin2023high")
DATASET_NAME = "lin2023high"
TISSUE_ID = "lin2023high_aijcgfvk_0000"
RESOLUTION = 1.0
CROP_SIZE = 224

pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(), reason=f"Dataset not found at {DATASET_PATH}"
)


class TestHETissueIds:
    def test_nonempty(self, he_dataset):
        ids = he_dataset.get_tissue_ids()
        assert len(ids) > 0

    def test_id_format(self, he_dataset):
        pattern = re.compile(r"^[a-z0-9]+_[a-z]{8}_[0-9]{4}$")
        for tid in he_dataset.get_tissue_ids():
            assert pattern.match(tid), f"Invalid tissue ID format: {tid}"


class TestHEGetTissue:
    def test_shape_and_type_chw(self, he_dataset):
        tissue = he_dataset.get_tissue(TISSUE_ID, preprocess=False, image_mode="CHW")
        assert tissue.tissue.shape[0] == 3
        assert tissue.tissue.ndim == 3
        assert isinstance(tissue.tissue, torch.Tensor)

    def test_shape_hwc(self, he_dataset):
        tissue = he_dataset.get_tissue(TISSUE_ID, preprocess=False, image_mode="HWC")
        assert tissue.tissue.shape[2] == 3

    def test_preprocess_normalizes(self, he_dataset):
        tissue = he_dataset.get_tissue(TISSUE_ID, preprocess=True, image_mode="CHW")
        # After ImageNet normalization, values should not be in [0, 255]
        assert tissue.tissue.dtype == torch.float32
        assert tissue.tissue.mean().abs() < 10  # roughly centered


class TestHENormalization:
    def test_imagenet_vs_hibou_differ(self, he_dataset):
        from spatialprot_data import HEImagingDataset

        hibou = HEImagingDataset(
            name=DATASET_NAME,
            path=DATASET_PATH,
            resolution=RESOLUTION,
            crop_size=CROP_SIZE,
            verbose=False,
            mean_std_type="hibou",
        )
        t1 = he_dataset.get_tissue(TISSUE_ID, preprocess=True)
        t2 = hibou.get_tissue(TISSUE_ID, preprocess=True)
        assert not torch.allclose(t1.tissue, t2.tissue)


class TestHETissueSize:
    def test_returns_chw(self, he_dataset):
        c, h, w = he_dataset._get_tissue_size(TISSUE_ID)
        assert c == 3
        assert h > 0 and w > 0


class TestHEGetCrop:
    def test_crop_shape(self, he_dataset):
        crop = he_dataset.get_crop(TISSUE_ID, crop_id=0, preprocess=False)
        assert crop.tissue.shape == (3, 224, 224)
        assert crop.kind == "crop"
        assert crop.crop_id == 0

    def test_crop_preprocessed(self, he_dataset):
        crop = he_dataset.get_crop(TISSUE_ID, crop_id=0, preprocess=True)
        assert crop.tissue.dtype == torch.float32
        # Preprocessed values shouldn't be in [0, 255] range
        assert crop.tissue.max() < 20


class TestHETissueMask:
    def test_mask_shape_and_dtype(self, he_dataset):
        mask = he_dataset.get_tissue_mask(TISSUE_ID)
        assert mask.mask.ndim == 2
        assert mask.mask.dtype == np.bool_ or mask.mask.dtype == bool
        assert mask.mask.any()


class TestHEPatientMap:
    def test_nonempty(self, he_dataset):
        assert len(he_dataset.patient_tissue_map) > 0
        for patient, tissues in he_dataset.patient_tissue_map.items():
            assert isinstance(tissues, list)
            assert len(tissues) > 0
