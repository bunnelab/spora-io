from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from spora_io.datasets.he import HEImagingDataset
from spora_io.datasets.ihc import SingleIHCImagingDataset
from spora_io.datasets.multiplex import MultiplexImagingDataset
from spora_io.datasets._types import ModKey, ComposedTissue
from spora_io.utils.utils import print_verbose


def _norm_modality_key(mod: ModKey) -> str:
    if isinstance(mod, str):
        return mod
    elif hasattr(mod, "name"):
        return mod.name
    else:
        raise ValueError(f"Invalid modality key: {mod}. Must be a string or an object with a 'name' attribute.")


class ComposedImagingDataset:
    """
    Compose multiple unimodal datasets (HE, Multiplex, etc.) into a single handle.

    - Uniform interface to fetch tissues/tiles per modality.
    - Ensures consistent tile strategy across modalities by construction.
    - Extensible via `modality_kwargs` to pass per-modality constructor arguments.

    Notes on behavior:
    - `get_tissue_ids()` returns the union of tissue IDs across all instantiated modalities.
    - `get_modalities_of_tissue(tissue_id)` lists which modalities contain that tissue ID.
    - Marker-specific helpers (indices/metadata) are forwarded to each unimodal dataset when available.
    """
    def __init__(
        self,
        name: str,
        path: Union[str, Path],
        modalities: Iterable[ModKey],
        tile_size: int,
        resolution: float | str,
        verbose: bool = True,
        load_cell_metadata: bool = False,
        tile_strategy: Optional[str] = None,
        split: Optional[str] = None,
        *,
        modality_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.name = name
        self.path = Path(path)
        self.verbose = verbose
        self.tile_size = tile_size
        self.tile_strategy = tile_strategy
        self.resolution = resolution
        self.load_cell_metadata = load_cell_metadata
        self.split = split

        self._unimodal: Dict[str, Any] = {}
        self._raw_modality_keys: List[str] = []

        per_mod_kwargs = {k: dict(v) for k, v in (modality_kwargs or {}).items()}
        modality_keys = [_norm_modality_key(mod) for mod in modalities]

        if "ihc" in modality_keys:
            ihc_dir = self.path / "ihc"
            if not ihc_dir.exists():
                raise FileNotFoundError(f"Requested all IHC markers, but IHC directory does not exist: {ihc_dir}")
            all_ihc_markers = os.listdir(self.path / "ihc")
            modality_keys = [key for key in modality_keys if key != "ihc"] + sorted(all_ihc_markers)

        for key in modality_keys:
            if self.verbose:
                print_verbose(f"Initializing modality '{key}'...")
            nominal_key = key if not key.startswith("ihc_") else "ihc"
            self._raw_modality_keys.append(key)

            kwargs_common = dict(
                name=name,
                path=str(self.path),
                verbose=verbose,
                tile_size=self.tile_size,
                resolution=self.resolution,
                load_cell_metadata=False,
                tile_strategy=self.tile_strategy,
                split=self.split,
            )
            kwargs_extra = dict(per_mod_kwargs.get(nominal_key, {}))

            if key == "he":
                ds = HEImagingDataset(**kwargs_common, **kwargs_extra)
                self._unimodal[key] = ds
            elif key in MultiplexImagingDataset.VALID_MODALITIES:
                standardization = kwargs_extra.pop("standardization", "identity")
                if standardization == "identity":
                    print_verbose(f"No standardization will be applied to modality '{key}'. Ensure this is intentional.", level="WARNING")
                ds = MultiplexImagingDataset(
                    **kwargs_common,
                    modality=key,
                    standardization=standardization,
                    **kwargs_extra,
                )
                self._unimodal[key] = ds
            elif key.startswith("ihc_"):
                ds = SingleIHCImagingDataset(
                    **kwargs_common,
                    marker_name=key,
                    **kwargs_extra,
                )
                self._unimodal[key] = ds
            else:
                raise ValueError(
                    f"Unsupported modality '{key}' in ComposedImagingDataset. "
                    "Supported: he, imc, codex, cycif, mibi, ihc, and ihc_<marker>."
                )

        self.modalities: List[str] = list(self._unimodal.keys())

        self._tissue_id_to_modalities: Dict[str, List[str]] = {}
        for mod_key, ds in self._unimodal.items():
            for tissue_id in ds.get_tissue_ids():
                self._tissue_id_to_modalities.setdefault(str(tissue_id), []).append(mod_key)

        if self.load_cell_metadata:
            print_verbose(f"Loading cell metadata")
            self.cell_metadata = pd.read_parquet(self.path / "metadata" / "cells.parquet").set_index("tissue_id")

        self.all_tissue_metadata = pd.read_parquet(self.path / "metadata" / "tissues.parquet").set_index("tissue_id")
        if self.split is not None:
            if "split" not in self.all_tissue_metadata.columns:
                raise ValueError(f"Split column not found in tissue metadata, but split argument {self.split} was provided.")
            self.all_tissue_metadata = self.all_tissue_metadata[self.all_tissue_metadata["split"] == self.split]
            if self.all_tissue_metadata.empty:
                raise ValueError(f"No tissue metadata found for split {self.split}. Please check the split argument and the contents of the tissue metadata.")
        self.all_tissue_metadata = self.all_tissue_metadata[
            self.all_tissue_metadata["modality"].isin(self._raw_modality_keys)
        ]
        self.all_tissue_ids = self.all_tissue_metadata.index.unique().tolist()
        self.patient_tissue_map = self.all_tissue_metadata.groupby("patient_id").apply(lambda df: df.index.tolist(), include_groups=False).to_dict()

        print_verbose(f"Composed dataset initialized with modalities: {self.modalities}")
        print_verbose(f"Total unique tissue samples across all modalities: {len(self.all_tissue_ids)}")

    def get_dataset(self, modality: ModKey) -> Any:
        """ 
        Get the unimodal dataset instance for a given modality key.
        Args:
            modality (ModKey): The modality key (string or object with 'name' attribute) to retrieve the dataset for.
        Returns:
            Any: The unimodal dataset instance corresponding to the modality key.
        Raises:
            KeyError: If the modality key is not part of this composed dataset.
        """
        key = _norm_modality_key(modality)
        if key not in self._unimodal:
            raise KeyError(f"Modality '{key}' is not part of this composed dataset.")
        return self._unimodal[key]

    def get_available_modalities(self) -> List[str]:
        """
        Get the list of available modalities in this composed dataset.
        Returns:
            List[str]: A list of modality keys representing the available modalities.
        """
        return list(self.modalities)
    
    def get_tissue_ids(self, modality: Optional[ModKey] = None) -> List[str]:
        """
        Get the list of tissue IDs available in the dataset. If a modality is specified, return only tissue IDs for that modality.
        Args:
            modality (Optional[ModKey]): The modality key to filter tissue IDs by. If None, returns tissue IDs across all modalities.
        Returns:            
            List[str]: A list of tissue IDs available in the dataset (filtered by modality if specified).
        """
        return self.all_tissue_ids if modality is None else self.get_dataset(modality).get_tissue_ids()

    def get_modalities_of_tissue(self, tissue_id: str) -> List[str]:
        """
        Get the list of modalities available for a given tissue ID.
        Args:
            tissue_id (str): The tissue ID to query modalities for.
        Returns:
            List[str]: A list of modality keys representing the modalities available for the given tissue ID.
        """
        return self._tissue_id_to_modalities[str(tissue_id)]

    def get_unimodal_tissue(self, tissue_id: str, modality: ModKey, kind: str = "uniprot_filtered", preprocess: bool = True,  image_mode="CHW"):
        """
        Get the tissue image for a given tissue ID and modality, with options for kind of image and preprocessing.
        Args:
            tissue_id (str): The tissue ID to retrieve.
            modality (ModKey): The modality key to specify which unimodal dataset to query.
            kind (str): The kind of tissue image to retrieve. Default is "uniprot_filtered". Valid options are "complete", "qc_filtered", and "uniprot_filtered".
            preprocess (bool): If True, preprocess the image (normalize). Default is True.
            image_mode (str): The desired image mode of the returned tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            Tissue: The tissue image as returned by the unimodal dataset's `get_tissue` method.
        """
        ds = self.get_dataset(modality)
        return ds.get_tissue(tissue_id, kind=kind, preprocess=preprocess, image_mode=image_mode)

    def get_unimodal_tissue_mask(self, tissue_id: str, modality: ModKey):
        """
        Get the quality control mask for a given tissue ID and modality.
        Args:
            tissue_id (str): The tissue ID to retrieve the mask for.
            modality (ModKey): The modality key to specify which unimodal dataset to query.
        Returns:
            np.ndarray: The quality control mask as returned by the unimodal dataset's `get_tissue_mask` method.
        """
        ds = self.get_dataset(modality)
        return ds.get_tissue_mask(tissue_id)

    def get_unimodal_tissue_size(self, tissue_id: str, modality: ModKey) -> Tuple[int, int, int]:
        """
        Get the tissue size (C,H,W) for a given tissue ID and modality.
        Args:            
            tissue_id (str): The tissue ID to retrieve the size for.
            modality (ModKey): The modality key to specify which unimodal dataset to query.
        Returns:
            Tuple[int, int, int]: The tissue size (C,H,W) as returned by the unimodal dataset's `_get_tissue_size` method.
        """
        ds = self.get_dataset(modality)
        return ds._get_tissue_size(tissue_id)

    def get_unimodal_tile(
        self,
        tissue_id: str,
        tile_id: int,
        modality: ModKey,
        kind: str = "uniprot_filtered",
        preprocess: bool = True,
        image_mode: str = "CHW",
    ):
        """
        Get a specific tile for a given tissue ID and modality.
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_id (int): The tile ID to retrieve.
            modality (ModKey): The modality key to specify which unimodal dataset to query.
            kind (str): The kind of image to retrieve. Valid options depend on modality.
            preprocess (bool): If True, preprocess the tile before returning.
            image_mode (str): The returned image layout, usually "CHW" or "HWC".
        Returns:
            Tissue: The tile image as returned by the unimodal dataset's `get_tile` method.
        """
        ds = self.get_dataset(modality)
        return ds.get_tile(tissue_id, tile_id, kind=kind, preprocess=preprocess, image_mode=image_mode)

    def get_composed_tissue(self, tissue_id: str, kind: str = "uniprot_filtered", preprocess: bool = True, image_mode="CHW") -> ComposedTissue:
        """
        Get a composed tissue sample for a given tissue ID, which includes all available modalities for that tissue.

        Args:
            tissue_id (str): The tissue ID to retrieve.
            kind (str): The kind of tissue image to retrieve. Default is "uniprot_filtered". Valid options are "complete", "qc_filtered", and "uniprot_filtered".
            preprocess (bool): If True, preprocess the images (normalize). Default is True.
            image_mode (str): The desired image mode of the returned tissue images. Valid options are "CHW" and "HWC". Default is "CHW".

        Returns:
            ComposedTissue: A ComposedTissue instance containing the tissue ID and a dictionary of modality-specific Tissue instances.
        """
        modalities = self.get_modalities_of_tissue(tissue_id)
        modality_tissues = {}
        for mod in modalities:
            modality_tissues[mod] = self.get_unimodal_tissue(tissue_id, mod, kind=kind, preprocess=preprocess, image_mode=image_mode)
        
        return ComposedTissue(
            tissue_id=tissue_id,
            modalities=modality_tissues
        )


    def get_composed_tissue_by_patient(self, patient_id: str, kind: str = "uniprot_filtered", preprocess: bool = True, image_mode="CHW") -> Sequence[ComposedTissue]:
        """
        Get composed tissue samples for all tissues associated with a given patient ID.

        Args:
            patient_id (str): The patient ID to retrieve tissues for.
            kind (str): The kind of tissue image to retrieve. Default is "uniprot_filtered". Valid options are "complete", "qc_filtered", and "uniprot_filtered".
            preprocess (bool): If True, preprocess the images (normalize). Default is True.
            image_mode (str): The desired image mode of the returned tissue images. Valid options are "CHW" and "HWC". Default is "CHW".

        Returns:
            Sequence[ComposedTissue]: A list of ComposedTissue instances for each tissue associated with the patient.
        """
        patient_tissues = self.patient_tissue_map.get(str(patient_id), [])
        
        return [self.get_composed_tissue(tid, kind=kind, preprocess=preprocess, image_mode=image_mode) for tid in patient_tissues]

    def __repr__(self) -> str:
        n_tiles = {
            modality: dataset._count_tiles()
            for modality, dataset in self._unimodal.items()
            if hasattr(dataset, "_count_tiles")
        }
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, modalities={self.modalities!r}, resolution={self.resolution!r}, "
            f"tile_size={self.tile_size!r}, tile_strategy={self.tile_strategy!r}, "
            f"split={self.split!r}, n_tissues={len(self.all_tissue_ids)}, n_tiles={n_tiles})"
        )
