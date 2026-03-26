"""Shared fixtures for spatialprot-data tests."""

import pytest
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/mnt/aimm/scratch/datasets_v2/lin2023high")
DATASET_NAME = "lin2023high"
TISSUE_ID = "lin2023high_aijcgfvk_0000"
RESOLUTION = 1.0
CROP_SIZE = 224

skip_no_dataset = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=f"Dataset not found at {DATASET_PATH}",
)


@pytest.fixture(scope="session")
def he_dataset():
    from spatialprot_data import HEImagingDataset
    return HEImagingDataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        resolution=RESOLUTION,
        crop_size=CROP_SIZE,
        verbose=False,
    )


@pytest.fixture(scope="session")
def mx_dataset():
    from spatialprot_data import MultiplexImagingDataset
    return MultiplexImagingDataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        modality="cycif",
        normalization="identity",
        resolution=RESOLUTION,
        crop_size=CROP_SIZE,
        verbose=False,
    )


@pytest.fixture(scope="session")
def composed_dataset():
    from spatialprot_data import ComposedImagingDataset
    return ComposedImagingDataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        modalities=["he", "cycif"],
        resolution=RESOLUTION,
        crop_size=CROP_SIZE,
        verbose=False,
        modality_kwargs={"cycif": {"normalization": "identity"}},
    )
