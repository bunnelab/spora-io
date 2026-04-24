from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import get_args, Any, Union, Protocol, Dict, Optional
import torch
import numpy as np
from numpy.typing import NDArray

@dataclass(kw_only=True)
class HEModality:
    """Modality class to represent the H&E modality."""
    name: str = "he"
    canonical_dir: str = "he" # Directory name to look for in the dataset structure for this modality

@dataclass(kw_only=True)
class IHCModality:
    """Modality class to represent the IHC modality."""
    name: str = "ihc"
    canonical_dir: str = ""

    def __post_init__(self):
        if not self.canonical_dir:
            self.canonical_dir = f"ihc/{self.name}"

@dataclass(kw_only=True)
class IMCModality:
    """Modality class to represent the IMC modality."""
    name: str = "imc"
    canonical_dir: str = "imc" 


@dataclass(kw_only=True)
class CODEXModality:
    """Modality class to represent the CODEX modality."""
    name: str = "codex"
    alt_name: str = "phenocycler"
    canonical_dir: str = "codex"

@dataclass(kw_only=True)
class CycIFModality:
    """Modality class to represent the CycIF modality."""
    name: str = "cycif"
    canonical_dir: str = "cycif"

@dataclass(kw_only=True)
class MIBIModality:
    """Modality class to represent the MIBI modality."""
    name: str = "mibi"
    canonical_dir: str = "mibi"

MultiplexModality = IMCModality | CODEXModality | CycIFModality | MIBIModality

Modality = HEModality | IHCModality | MultiplexModality
ModKey = Union[str, Modality] 

@dataclass(kw_only=True)
class HETissue:
    """HETissue class to represent a H&E Tissue sample."""
    image: torch.Tensor | NDArray[np.float32]
    tissue_id: str
    channels: str = "RGB"
    kind: str = "tissue"

@dataclass(kw_only=True)
class MultiplexTissue:
    """MultiplexTissue class to represent a Multiplex Tissue sample."""
    image: torch.Tensor | NDArray[np.float32]
    tissue_id: str
    kind: str = "tissue"
    measured_mask: NDArray[np.bool_] | None
    image_loading_mask: NDArray[np.bool_] | None
    channel_names: NDArray[np.str_] | None = None
    uniprot_ids: NDArray[np.object_] | None = None

@dataclass(kw_only=True)
class IHCTissue:
    """IHCTissue class to represent an IHC Tissue sample."""
    image: torch.Tensor | NDArray[np.float32]
    tissue_id: str
    channels: str = "RGB"
    kind: str = "tissue"

Tissue = HETissue | MultiplexTissue | IHCTissue


@dataclass(kw_only=True)
class TissueMask:
    """TissueMask class to represent a Tissue Mask sample."""
    mask: NDArray[np.bool_]
    tissue_id: str

@dataclass(kw_only=True)
class CellMask:
    """Cell Segmentation/Instance Mask class to represent a Cell Segmentation/Instance Mask sample."""
    mask: NDArray[np.int_]
    tissue_id: str
    mapping: Dict[int, str] | None = None

@dataclass(kw_only=True)
class ComposedTissue:
    """ComposedTissue class to represent a tissue sample composed of multiple modalities."""
    tissue_id: str
    modalities: Dict[str, Tissue]

def get_modality_from_str(modality_str: str, union_type: Any = Modality) -> Any:
    """Convert a modality string (case-insensitive) to the appropriate modality instance."""
    for cls in get_args(union_type):
        if getattr(cls, '__origin__', None) is Union:
            try:
                return get_modality_from_str(modality_str, cls)
            except ValueError:
                continue
        if is_dataclass(cls) and callable(cls):  # Ensure it's a dataclass class, not an instance
            instance = cls()
            # if hasattr(instance, "name") and instance.name.lower() == modality_str.lower(): # type: ignore
            if hasattr(instance, "name") and modality_str.lower() in [instance.name.lower()] + ([instance.alt_name.lower()] if hasattr(instance, "alt_name") else []): # type: ignore
                return instance
    raise ValueError(f"Unknown modality: {modality_str}")

def is_valid_modality_instance(obj: Any, union_type: Any = Modality) -> bool:
    """Check if the object is a valid instance of the given Modality union type."""
    for cls in get_args(union_type):
        if getattr(cls, '__origin__', None) is Union:
            if is_valid_modality_instance(obj, cls):
                return True
        elif is_dataclass(cls) and isinstance(obj, cls): # type: ignore
            return True
    return False