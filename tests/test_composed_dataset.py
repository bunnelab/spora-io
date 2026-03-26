"""Tests for ComposedImagingDataset."""

import pytest
import torch
from pathlib import Path

TISSUE_ID = "lin2023high_aijcgfvk_0000"

pytestmark = pytest.mark.skipif(
    not Path("/mnt/aimm/scratch/datasets_v2/lin2023high").exists(),
    reason="Dataset not found",
)


class TestComposedModalities:
    def test_available(self, composed_dataset):
        mods = composed_dataset.get_available_modalities()
        assert "he" in mods
        assert "cycif" in mods


class TestComposedTissueIds:
    def test_nonempty(self, composed_dataset):
        ids = composed_dataset.get_tissue_ids()
        assert len(ids) > 0

    def test_per_modality(self, composed_dataset):
        he_ids = composed_dataset.get_tissue_ids(modality="he")
        cycif_ids = composed_dataset.get_tissue_ids(modality="cycif")
        assert len(he_ids) > 0
        assert len(cycif_ids) > 0


class TestComposedModalitiesOfTissue:
    def test_both_present(self, composed_dataset):
        mods = composed_dataset.get_modalities_of_tissue(TISSUE_ID)
        assert "he" in mods
        assert "cycif" in mods


class TestComposedGetDataset:
    def test_returns_correct_types(self, composed_dataset):
        from spatialprot_data import HEImagingDataset, MultiplexImagingDataset
        assert isinstance(composed_dataset.get_dataset("he"), HEImagingDataset)
        assert isinstance(composed_dataset.get_dataset("cycif"), MultiplexImagingDataset)

    def test_invalid_modality_raises(self, composed_dataset):
        with pytest.raises(KeyError):
            composed_dataset.get_dataset("invalid")


class TestComposedGetTissue:
    def test_composed_tissue(self, composed_dataset):
        ct = composed_dataset.get_composed_tissue(TISSUE_ID, kind="complete", preprocess=False)
        assert "he" in ct.modalities
        assert "cycif" in ct.modalities
        he = ct.modalities["he"]
        mx = ct.modalities["cycif"]
        assert he.tissue.shape[0] == 3
        assert mx.tissue.shape[0] > 3


class TestComposedUnimodal:
    def test_he(self, composed_dataset):
        from spatialprot_data import HETissue
        tissue = composed_dataset.get_unimodal_tissue(TISSUE_ID, "he", preprocess=False)
        assert isinstance(tissue, HETissue)

    def test_cycif(self, composed_dataset):
        from spatialprot_data import MultiplexTissue
        tissue = composed_dataset.get_unimodal_tissue(TISSUE_ID, "cycif", kind="complete", preprocess=False)
        assert isinstance(tissue, MultiplexTissue)
