"""spatialprot_data: A data structure and loader library for spatial proteomics."""

from spatialprot_data.datasets._types import (
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
from spatialprot_data.datasets.base import BaseImagingDataset
from spatialprot_data.datasets.he import HEImagingDataset
from spatialprot_data.datasets.ihc import SingleIHCImagingDataset
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset
from spatialprot_data.datasets.compose import ComposedImagingDataset

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
