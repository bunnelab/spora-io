from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from loguru import logger
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Union, Tuple, Sequence, Callable, Optional
import json

from spatialprot_data.datasets._types import get_modality_from_str, is_valid_modality_instance, ModKey, Tissue, \
                                            TissueMask, CellMask
from spatialprot_data.utils.utils import print_verbose




class BaseImagingDataset(ABC):
    """
    Base class for all imaging datasets.
    """

    def __init__(self, 
                 name: str,
                 path: os.PathLike | str,
                 modality: ModKey,
                 resolution: float | str,
                 load_cell_metadata: bool = False,
                 verbose: bool = True,
                 label: Optional[str] = None,
                 labels_to_keep: Optional[Sequence[str]] = None, 
                 label_modifying_fn: Optional[Callable] = None,
                 label_type: str = "classification"
                 ):
        self.name = name
        self.path = Path(path)
        self.verbose = verbose
        self.resolution = resolution
        self.label = label 
        self.labels_to_keep = labels_to_keep
        self.label_modifying_fn = label_modifying_fn
        self.label_type = label_type
        
        if not isinstance(self.resolution, (float, str)):
            try:
                self.resolution = float(self.resolution)
            except Exception as e:
                print_verbose(f"Failed auto-conversion of resolution argument. Expected str/float, but got {type(self.resolution)}")
                raise e
        self.resolution = f"{str(self.resolution).replace(".", "_")}mpp"

        if isinstance(modality, str):
            self.modality = get_modality_from_str(modality)
        else:
            self.modality = modality
            assert is_valid_modality_instance(modality), f"Invalid modality instance {type(modality)} provided."

        # check existence of tissue masks
        self.tissue_masks_dir: Any | Path = self.path / "segmentations" / self.modality.canonical_dir / "tissue_masks" / self.resolution

        if not self.tissue_masks_dir.exists():
            print_verbose(f"Tissue masks directory {self.tissue_masks_dir} does not exist. Tissue masks will not be loaded.", level="WARNING")
            self.tissue_masks_dir = None

        self.tissue_metadata = pd.read_parquet(self.path / "metadata" / "tissues.parquet").set_index("tissue_id")
        if self.label is not None:
            self.tissue_metadata = self.tissue_metadata[self.tissue_metadata[self.label].isin(self.labels_to_keep)]
            if self.label_modifying_fn is None:
                print_verbose(f"Did not find label_modifying_fn, using identity.")
                self.label_modifying_fn = lambda x: x
            self.tissue_metadata[self.label] = self.tissue_metadata[self.label].map(self.label_modifying_fn)

            if self.label_type == "classification":
                self.unique_labels = self.tissue_metadata[self.label].unique().to_numpy()
                self.label_encoder = {unique_label: i for (i, unique_label) in enumerate(self.unique_labels)}
            

        if load_cell_metadata:
            print_verbose(f"Loading cell metadata")
            self.cell_metadata = pd.read_parquet(self.path / "metadata" / "cells.parquet").set_index("tissue_id")

        self.tissue_modality_metadata = self.tissue_metadata[self.tissue_metadata["modality"] == self.modality.name]
        self.patient_tissue_map = self.tissue_modality_metadata.groupby("patient_id").apply(lambda df: df.index.tolist(), include_groups=False).to_dict()


    def get_tissue_ids(self, kind="modality") -> NDArray[np.str_]:
        """
        Get the unique tissue IDs from the tissue annotations.
        Args:
            kind (str): The kind of tissue IDs to retrieve. Default is "modality", which returns tissue ids for tissues that have the specified modality. If "all", returns tissue ids for all tissues in the dataset.  
        Returns:
            NDArray[np.str_]: An array of unique tissue IDs.
        """
        if kind == "modality":
            return self.tissue_modality_metadata.index.values
        elif kind == "all":
            return self.tissue_metadata.index.values
        else:
            raise ValueError(f"Invalid kind {kind} provided. Expected 'modality' or 'all'.")
        

    @abstractmethod 
    def get_tissue(self, tissue_id: str, kind="complete", preprocess: bool = True, image_mode: str = "CHW") -> Tissue:
        """
        Get the tissue image for a given tissue id. The kind argument specifies whether to return the complete tissue image (all channels) or the modality-specific tissue image (filtered channels).
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Default is "complete". However, subclasses can change the default to the most commonly used kind.
                        Valid options for kind:
                            - "complete": returns the complete tissue image with all channels
                            - "qc_filtered": returns the tissue image with quality control 
                            - "filtered": returns the tissue image with only the channels relevant to the dataset's modality based on the priors and quality control filtering
            preprocess (bool): Whether to preprocess the image (e.g. normalize) before returning it. Default is True.
            image_mode (str): The desired image mode of the returned tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            Tissue: The tissue image as a Tissue instance.
        """
        pass

    def get_tissue_mask(self, tissue_id: str, image_mode: str = "HWC") -> TissueMask:
        """
        Get the tissue mask for a given tissue id. If the tissue masks directory does not exist, return a full mask.
        Args:
            tissue_id (str): The tissue ID to retrieve the mask for.
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "HWC".
        Returns:
            TissueMask: The tissue mask as a TissueMask instance.
        """
        if self.tissue_masks_dir is None:
            raise ValueError("Tissue masks directory is not set.")
        
        mask_path = self.tissue_masks_dir / f"{tissue_id}.npz"
        if not mask_path.exists():
            print_verbose(f"Tissue mask file {mask_path} does not exist. Returning full mask.",
                            level="WARNING")
            return TissueMask(mask=np.ones((self._get_tissue_size(tissue_id, image_mode=image_mode)[1], self._get_tissue_size(tissue_id, image_mode=image_mode)[2]), dtype=np.bool_),
                              tissue_id=tissue_id)
        return TissueMask(
            mask=np.load(mask_path)["mask"],
            tissue_id=tissue_id
        )
    @abstractmethod
    def _get_tissue_size(self, tissue_id: str, image_mode: str = "CHW") -> Tuple[int, int, int]:
        """
        Get the tissue size (C,H,W) for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the size for.
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            Tuple[int, int, int]: The tissue size as a tuple (C, H, W).
        """
        pass


    @abstractmethod
    def get_crop(self, tissue_id: str, crop_id: int) -> Tissue:
        """
        Get a specific crop based on the tissue id and crop id. 
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for.
            crop_id (int): The crop ID to retrieve.
        Returns:
            Tissue: The crop image as a Tissue instance.
        """

    def get_cell_instance_mask(self, tissue_id: str, match_tissue_resolution: bool = False) -> CellMask:
        """
        Get the cell instance mask for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the cell instance mask for.
            match_tissue_resolution (bool): If True, downsample the mask to match
                the dataset's working resolution using nearest-neighbor interpolation.
                Default is False (return at native resolution).
        Returns:
            CellMask: The cell instance mask as a CellMask instance.

        """
        ci_mask_path = self.path / "segmentations" / self.modality.canonical_dir / "cell_masks" / "instances" / f"{tissue_id}.npz"
        if not ci_mask_path.exists():
            raise ValueError(f"Cell instance mask file {ci_mask_path} does not exist for tissue_id {tissue_id}.")
        mask = np.load(ci_mask_path)["mask"]
        if match_tissue_resolution:
            target_h, target_w = self._get_tissue_size(tissue_id, image_mode="CHW")[1:]
            if mask.shape != (target_h, target_w):
                from PIL import Image
                mask = np.array(Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST))
        return CellMask(
            mask=mask,
            tissue_id=tissue_id
        )
    
    def get_cell_task_mask(self, tissue_id: str, mask_type: str) -> CellMask:
        """
        Get the cell task mask for a given tissue id and mask type.
        Args:            
            tissue_id (str): The tissue ID to retrieve the cell task mask for.
            mask_type (str): The type of cell task mask to retrieve. Valid options can be retrieved from `get_cell_task_mask_types` method.
        Returns:
            CellMask: The cell task mask as a CellMask instance.
        """
        ct_mask_path = self.path / "segmentations" / self.modality.canonical_dir / "cell_masks" / mask_type / f"{tissue_id}.npz"
        if not ct_mask_path.exists():
            raise ValueError(f"Cell task mask file {ct_mask_path} does not exist for tissue_id {tissue_id} and mask_type {mask_type}.")
        if hasattr(self, f"{mask_type}_label_encoder"):
            label_encoder = getattr(self, f"{mask_type}_label_encoder")
            mapping = getattr(self, f"{mask_type}_mapping")
        else:
            label_encoder = pd.read_parquet(self.path / "segmentations" / self.modality.canonical_dir / "cell_masks" / mask_type / "label_encoder.parquet")
            mapping = {row["id"]: row["name"] for _, row in label_encoder.iterrows()}
            setattr(self, f"{mask_type}_label_encoder", label_encoder)
            setattr(self, f"{mask_type}_mapping", mapping)
        # label_encoder is df with columns name and id 
        mask = np.load(ct_mask_path)["mask"]
        return CellMask(
            mask=mask,
            tissue_id=tissue_id,
            mapping=mapping
        )

    def get_cell_task_mask_types(self) -> Sequence[str]:
        """
        Get the available cell task mask types for the dataset.
        Returns:
            Sequence[str]: A list of available cell task mask types.
        """
        categories_dir = self.path / "segmentations" / self.modality.canonical_dir / "cell_masks" 
        if not categories_dir.exists():
            raise ValueError(f"Categories directory {categories_dir} does not exist.")
        return [d.name for d in categories_dir.iterdir() if d.is_dir() and d.name != "instances"]
        
    def get_tissue_by_patient(self, patient_id: str, kind="complete", preprocess: bool = True, image_mode: str = "CHW") -> Sequence[Tissue]:
        """
        Get all tissues for a given patient id.
        Args:
            patient_id (str): The patient ID to retrieve the tissues for.
            kind (str): The kind of tissue image to retrieve. Default is "complete". However, subclasses can change the default to the most commonly used kind.
                        Valid options for kind:
                            - "complete": returns the complete tissue image with all channels
                            - "qc_filtered": returns the tissue image with quality control 
                            - "filtered": returns the tissue image with only the channels relevant to the dataset's modality based on the priors and quality control filtering
            preprocess (bool): Whether to preprocess the image (e.g. normalize) before returning it. Default is True.
            image_mode (str): The desired image mode of the returned tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            Sequence[Tissue]: A list of tissue images as Tissue instances.
        """
        tissue_ids = self.patient_tissue_map.get(str(patient_id), [])
        return [self.get_tissue(tissue_id, kind=kind, preprocess=preprocess, image_mode=image_mode) for tissue_id in tissue_ids]


    def _try_to_load_crop_coords(self):
        # crop coordinates
        crop_coords_path = self.path / "segmentations" / self.modality.name / "crop_coordinates" / self.resolution / f"{self.crop_size}_tiles_coordinates.json"
        if crop_coords_path.exists():
            self.crop_coordinates = json.load(crop_coords_path.open())
            if self.verbose:
                print_verbose(f"Loaded crop coordinates from {crop_coords_path}")
        else:
            self.crop_coordinates = None
            if self.verbose:
                print_verbose(f"No crop coordinates found at {crop_coords_path}. get_crop will return random crops.", level="WARNING")

        crop_count = self._count_crops()
        if self.verbose:
            print_verbose(f"Dataset {self.name} has {crop_count} crops of size {self.crop_size} at resolution {self.resolution}.",
                          level="DEBUG" if crop_count > 0 else "WARNING")

    def _count_crops(self) -> int:
        """
        Count the number of crops in the dataset.
        Args:
            crop_folder_path (str | None): The path to the crop folder. If None, uses the default crop folder.
        Returns:
            int: The number of crops in the dataset.
        """
        if self.crop_coordinates is not None:
            return sum(len(coords) for coords in self.crop_coordinates.values())
        else:
            return 0






