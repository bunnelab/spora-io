"""spora_io: A data structure and loader library for spatial proteomics."""

from spora_io.datasets._types import (
    HEModality,
    IHCModality,
    IMCModality,
    CODEXModality,
    CycIFModality,
    HETissue,
    MultiplexTissue,
    IHCTissue,
    ComposedTissue,
    TissueMask,
    CellMask,
    get_modality_from_str,
    is_valid_modality_instance,
)
from spora_io.datasets.base import BaseImagingDataset
from spora_io.datasets.he import HEImagingDataset
from spora_io.datasets.ihc import SingleIHCImagingDataset
from spora_io.datasets.multiplex import MultiplexImagingDataset
from spora_io.datasets.compose import ComposedImagingDataset

__all__ = [
    # Modalities
    "HEModality",
    "IHCModality",
    "IMCModality",
    "CODEXModality",
    "CycIFModality",
    # Tissue types
    "HETissue",
    "MultiplexTissue",
    "IHCTissue",
    "ComposedTissue",
    "TissueMask",
    "CellMask",
    # Dataset classes
    "BaseImagingDataset",
    "HEImagingDataset",
    "SingleIHCImagingDataset",
    "MultiplexImagingDataset",
    "ComposedImagingDataset",
    # Utilities
    "get_modality_from_str",
    "is_valid_modality_instance",
]
