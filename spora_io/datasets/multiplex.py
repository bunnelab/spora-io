from __future__ import annotations

import os
import torch
import numpy as np
from numpy.typing import NDArray
from einops import rearrange
from loguru import logger 
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional
import zarr
import json

from spora_io.datasets.base import BaseImagingDataset
from spora_io.utils.dataset.standardize import build_standardizer
from spora_io.utils.utils import print_verbose
from spora_io.utils.dataset.transforms import FilterFactory
from spora_io.datasets._types import MultiplexTissue, TissueMask, CellMask

class MultiplexImagingDataset(BaseImagingDataset):
    """
    Class for handling multiplex imaging datasets.

    Attributes:
        VALID_MODALITIES (set): A set of valid modalities for multiplex imaging datasets. Valid options are "imc", "codex", and "cycif".
        standardization (str): The type of standardization to apply to the images. This is passed to the build_standardizer function to create a standardizer instance.
        disable_quantile_mask (bool): Quantile masking will search for channels with 0 quantile / variance and exclude them from standardization. Setting this to True will disable this behavior and include all channels in standardization. This is passed to the build_standardizer function.
        filter_list (List[str]): A list of filter names to apply to the dataset. Currently supported filters are "gaussian_blur" and "median_filter". These are applied sequentially, and the parameters for each filter can be specified in the filter_params dictionary in kwargs, with the filter name as the key and the parameters as a dictionary of parameters for that filter.
        use_mean_std (bool): Whether to use mean and standard deviation for standardization. If False, only quantile normalization will be applied if not disabled. This is passed to the build_standardizer function.
        return_uniprot_ids (bool): Whether to return uniprot IDs for the channels. This requires a "uniprot_id" column in the channels.parquet file. If this column is not present, uniprot IDs will not be returned regardless of this setting.
    """
    VALID_MODALITIES = {"imc", "codex", "cycif", "mibi"}
    def __init__(self,
                 name: str,
                 path: os.PathLike | str,
                 modality: str,
                 standardization: str, 
                 resolution: float | str,
                 tile_size: Optional[int] = None,
                 verbose: bool = True,
                 load_cell_metadata: bool = False,
                 disable_quantile_mask: bool = False,
                 filter_list: List[str] | None = None,
                 use_mean_std: bool = True,
                 return_uniprot_ids: bool = True,
                 tile_strategy: Optional[str] = None,
                 split: Optional[str] = None,
                 **kwargs
    ):
        assert modality in self.VALID_MODALITIES, f"Invalid modality {modality}. Valid options are: {self.VALID_MODALITIES}"
        label_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in ("label", "labels_to_keep", "label_modifying_fn", "label_type")}
        super().__init__(name=name, path=path, 
                         modality=modality,
                         resolution=resolution, 
                         tile_size=tile_size,
                         load_cell_metadata=load_cell_metadata, 
                         verbose=verbose,
                         tile_strategy=tile_strategy,
                         split=split,
                         **kwargs, 
                         **label_kwargs)
        self.return_uniprot_ids = return_uniprot_ids
        self.kwargs = kwargs
        self.img_folder = self.path / self.modality.canonical_dir / self.resolution / "images"

        self.channel_list = pd.read_parquet(self.path / self.modality.canonical_dir / "channels.parquet")

        if "qc_pass" not in self.channel_list.columns:
            print_verbose(f"No 'qc_pass' column found in channel list at {self.path / self.modality.canonical_dir / 'channels.parquet'}. All channels will be considered as passing quality control.", level="WARNING")
            self.channel_list["qc_pass"] = True
        self.quality_control_mask = self.channel_list["qc_pass"].to_numpy(dtype=bool)


        filter_params = self.kwargs.get("filter_params", {})
        self._set_optional_filters(filter_list if filter_list is not None else [], filter_params=filter_params)
        

        self.image_channel_map = None
        default_image_channel_map_path = self.path / self.modality.canonical_dir / f"channels_per_tissue.parquet"
        if default_image_channel_map_path.exists():
            self.image_channel_map = pd.read_parquet(default_image_channel_map_path)
            if self.verbose:
                print_verbose(f"Using image-channel map from {default_image_channel_map_path}")
        else:
            if self.verbose:
                print_verbose(f"No image-channel map found at {default_image_channel_map_path}. Proceeding to create with all channels included.",
                              level="WARNING")
            
            self.image_channel_map = pd.DataFrame(
                index=self.tissue_metadata.index,
                columns=self.channel_list["channel_name"].to_numpy(),
            )
            self.image_channel_map.fillna(1, inplace=True)  # Include all channels by default

        self.image_channel_map.replace(0, False, inplace=True)
        self.image_channel_map.replace(1, True, inplace=True)

        
        self.standardizer = build_standardizer(
            standardization=standardization,
            modality_dir=self.path / self.modality.canonical_dir / self.resolution,
            channels_per_image=self.image_channel_map,
            disable_quantile_mask=disable_quantile_mask,
            filter_factory=self.filter_factory,
            use_mean_std=use_mean_std,
        )
    
        if self.verbose:
            print_verbose(f"Using Multiplex standardization: {self.standardizer.__class__.__name__}")
        # generating marker indices
        self._try_to_create_uniprot_mask()
        self._try_to_load_tile_coords()

    def _try_to_create_uniprot_mask(self):
        if "uniprot_id" not in self.channel_list.columns:
            print_verbose(f"No 'uniprot_id' column found in channel list at {self.path / self.modality.canonical_dir / 'channels.parquet'}. uniprot_ids will not be returned.", level="WARNING")
            self.channel_list["uniprot_id"] = np.nan
            self.uniprot_mask = np.zeros(len(self.channel_list), dtype=bool)
            if self.return_uniprot_ids:
                self.return_uniprot_ids = False
                print_verbose(f"Setting return_uniprot_ids to False.", level="WARNING")
            return

        uniprot_regex = r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?$"
        has_uniprot_id = self.channel_list["uniprot_id"].astype("string").str.match(uniprot_regex, na=False)
        self.uniprot_mask = has_uniprot_id.to_numpy(dtype=bool)
        if self.verbose:
            print_verbose(f"Uniprot mask created with {self.uniprot_mask.sum()} channels.")

    def _get_uniprot_ids(self, mask: np.ndarray | None) -> NDArray[np.object_] | None:
        if not self.return_uniprot_ids or mask is None:
            return None
        return self.channel_list["uniprot_id"].to_numpy(dtype=object, copy=False)[mask]

    def _set_optional_filters(self, filter_list: List[str], filter_params: dict = {}) -> None:
        """
        Set optional filters for the dataset based on the provided filter list and parameters.
        Args:
            filter_list (List[str]): A list of filter names to apply. Supported filters depend on the implementation of FilterFactory.
            filter_params (dict): A dictionary of parameters for the filters. Keys should correspond to filter names, and values should be dictionaries of parameters for those filters.
        Returns:
            None
        """
        self.filter_factory = FilterFactory(filter_list, filter_params=filter_params)
        self.filter_factory.print_filters()

    def _get_tissue_all_channels(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the full tissue image without filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
        Returns:
            MultiplexTissue: Data class containing the full tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        img = torch.from_numpy(zarr.open(img_path, mode='r')[:]).float()
        return MultiplexTissue(
            image=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=np.ones(measured_mask.sum(), dtype=bool),
            channel_names=self.get_channel_names(tissue_id, kind="complete", measured_mask=measured_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="complete", measured_mask=measured_mask),
        )
    
    def _get_tissue_qc_filtered(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the tissue image filtered by quality control for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for. 
        Returns:
            MultiplexTissue: Data class containing the quality control filtered tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        image_loading_mask = self.quality_control_mask[measured_mask]
        img = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask)]).float()
        qc_mask = self.quality_control_mask & measured_mask
        return MultiplexTissue(
            image=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask,
            channel_names=self.get_channel_names(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
        )
    
    def _get_tissue_uniprot_filtered(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the tissue image filtered by quality control and valid UniProt availability for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
        Returns:
            MultiplexTissue: Data class containing the filtered tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        qc_mask = self.quality_control_mask & measured_mask
        filtered_mask = self.uniprot_mask & qc_mask
        image_loading_mask = filtered_mask[measured_mask]
        img = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask)]).float()
        return MultiplexTissue(
            image=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask,
            channel_names=self.get_channel_names(tissue_id, kind="uniprot_filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="uniprot_filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
        )


    def get_channel_names(self, tissue_id: str, kind: str = "complete", 
                          measured_mask=None, qc_mask=None, filtered_mask=None) -> NDArray[np.str_]:
        """
        Get the channel names for a given tissue id and kind.
        Args:
            tissue_id (str): The tissue ID to retrieve the channel names for.
            kind (str): The kind of tissue image to retrieve channel names for. Valid options are "complete", "qc_filtered", and "uniprot_filtered".
            measured_mask (np.ndarray | None): A boolean array indicating which channels are measured for the given tissue. If None, it will be retrieved from the image_channel_map. This is used to determine which channels to consider when applying the kind filtering.
            qc_mask (np.ndarray | None): A boolean array indicating which channels pass quality control for the given tissue. If None, it will be computed from the quality_control_mask and measured_mask. This is used when kind is "qc_filtered" or "uniprot_filtered" to determine which channels to include.
            filtered_mask (np.ndarray | None): A boolean array indicating which channels have valid UniProt IDs for the given tissue. If None, it will be computed from the uniprot_mask and qc_mask. This is used when kind is "uniprot_filtered" to determine which channels to include.
        Returns:
            NDArray[np.str_]: The channel names as a 1D array of shape (n_channels,).
        """
        if measured_mask is None:
            measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        if kind == "complete":
            return self.channel_list["channel_name"][measured_mask].values
        if kind == "qc_filtered":
            if qc_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
            return self.channel_list["channel_name"][qc_mask].values
        if kind == "uniprot_filtered":
            if filtered_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
                filtered_mask = self.uniprot_mask & qc_mask
            return self.channel_list["channel_name"][filtered_mask].values
        raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'uniprot_filtered'.")

    def get_uniprot_ids(self, tissue_id: str, kind: str = "complete", measured_mask=None, qc_mask=None, filtered_mask=None) -> NDArray[np.object_] | None:
        """
        Get the uniprot IDs for a given tissue id and kind. This will return None if return_uniprot_ids is False or if there are no valid uniprot IDs for the given kind.
        
        Args:
            tissue_id (str): The tissue ID to retrieve the uniprot IDs for.
            kind (str): The kind of tissue image to retrieve uniprot IDs for. Valid options are "complete", "qc_filtered", and "uniprot_filtered".
            measured_mask (np.ndarray | None): A boolean array indicating which channels are measured for the given tissue. If None, it will be retrieved from the image_channel_map. This is used to determine which channels to consider when applying the kind filtering.
            qc_mask (np.ndarray | None): A boolean array indicating which channels pass quality control for the given tissue. If None, it will be computed from the quality_control_mask and measured_mask. This is used when kind is "qc_filtered" or "uniprot_filtered" to determine which channels to include.
            filtered_mask (np.ndarray | None): A boolean array indicating which channels have valid UniProt IDs for the given tissue. If None, it will be computed from the uniprot_mask and qc_mask. This is used when kind is "uniprot_filtered" to determine which channels to include.

        Returns:
            NDArray[np.object_] | None: The uniprot IDs as a 1D array of shape (n_channels,). Returns None if return_uniprot_ids is False or if there are no valid uniprot IDs for the given kind.
        """
        if not self.return_uniprot_ids:
            return None
        if measured_mask is None:
            measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        if kind == "complete":
            return self._get_uniprot_ids(measured_mask)
        if kind == "qc_filtered":
            if qc_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
            return self._get_uniprot_ids(qc_mask)
        if kind == "uniprot_filtered":
            if filtered_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
                filtered_mask = self.uniprot_mask & qc_mask
            return self._get_uniprot_ids(filtered_mask)
        raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'uniprot_filtered'.")
        
    def  _refine_channel_metadata(
        self,
        image_loading_mask: np.ndarray | None,
        channel_names: NDArray[np.str_] | None,
        uniprot_ids: NDArray[np.object_] | None,
        refined_mask: np.ndarray | None,
    ) -> tuple[np.ndarray | None, NDArray[np.str_] | None, NDArray[np.object_] | None]:
        """Keep metadata aligned with standardized outputs, with a zero-copy fast path when nothing changed."""
        if image_loading_mask is None or refined_mask is None:
            return image_loading_mask, channel_names, uniprot_ids

        if refined_mask is image_loading_mask or refined_mask.sum() == image_loading_mask.sum():
            return image_loading_mask, channel_names, uniprot_ids

        keep_in_loaded = np.asarray(refined_mask[image_loading_mask], dtype=bool)
        channel_names_out = channel_names[keep_in_loaded] if channel_names is not None else None
        uniprot_ids_out = uniprot_ids[keep_in_loaded] if uniprot_ids is not None else None
        return refined_mask, channel_names_out, uniprot_ids_out


    def get_tissue(self, tissue_id: str, kind="uniprot_filtered", preprocess=True, image_mode="CHW") -> MultiplexTissue:
        """ 
        Get the tissue image for a given tissue id, with options for filtering channels and preprocessing.
        Args:            
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Valid options are "complete", "qc_filtered", and "uniprot_filtered". Default is "uniprot_filtered".
            preprocess (bool): Whether to preprocess the image using the standardizer. Default is True.
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            MultiplexTissue: The tissue image as a MultiplexTissue instance.
        """ 
        if kind == "complete":
            tissue = self._get_tissue_all_channels(tissue_id)
        elif kind == "qc_filtered":
            tissue = self._get_tissue_qc_filtered(tissue_id)
        elif kind == "uniprot_filtered":
            tissue = self._get_tissue_uniprot_filtered(tissue_id)
        else:
            raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'uniprot_filtered'.")

        if preprocess:
            img, refined_mask = self.standardizer.apply(tissue.image, tissue_id, tissue.measured_mask, tissue.image_loading_mask)
            image_loading_mask, channel_names, uniprot_ids = self._refine_channel_metadata(
                tissue.image_loading_mask,
                tissue.channel_names,
                tissue.uniprot_ids,
                refined_mask,
            )
            if image_mode == "HWC":
                img = rearrange(img, "C H W -> H W C")
            return MultiplexTissue(
                image=img,
                tissue_id=tissue_id,
                measured_mask=tissue.measured_mask,
                image_loading_mask=image_loading_mask,
                channel_names=channel_names,
                uniprot_ids=uniprot_ids,
            )
        if image_mode == "HWC":
            tissue.image = rearrange(tissue.image, "C H W -> H W C")
        return tissue

    def _get_tissue_size(self, tissue_id: str, image_mode: str = "CHW") -> Tuple[int, int, int]:
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        img = zarr.open(img_path, mode='r')
        if image_mode == "CHW":
            return img.shape[0], img.shape[1], img.shape[2] #type: ignore
        else:
            return img.shape[2], img.shape[0], img.shape[1] #type: ignore


    def get_tile_by_coordinates(self, tissue_id: str, row: int, col: int,
                 kind: str = "uniprot_filtered",
                 image_mode: str = "CHW",
                 preprocess: bool = True,
                 ) -> MultiplexTissue:
        """
        Get a specific tile based on the tissue id and tile row and column coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            row (int): The row coordinate of the top-left corner of the tile.
            col (int): The column coordinate of the top-left corner of the tile.
            preprocess (bool): Whether to preprocess the tile using the standardizer. Default is True.
            kind (str): The kind of tissue image to retrieve for the tile. Valid options are "complete", "qc_filtered", and "uniprot_filtered". Default is "uniprot_filtered".
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            MultiplexTissue: The specific tile as a MultiplexTissue instance.
        """

        if kind == "complete":
            tile = self._get_tile_all_channels(tissue_id, col, row)
        elif kind == "qc_filtered":
            tile = self._get_tile_qc_filtered(tissue_id, col, row)
        elif kind == "uniprot_filtered":
            tile = self._get_tile_uniprot_filtered(tissue_id, col, row)
        else:
            raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'uniprot_filtered'.")

        if preprocess:
            img, refined_mask = self.standardizer.apply(tile.image, tissue_id, tile.measured_mask, tile.image_loading_mask)
            image_loading_mask, channel_names, uniprot_ids = self._refine_channel_metadata(
                tile.image_loading_mask,
                tile.channel_names,
                tile.uniprot_ids,
                refined_mask,
            )
            if image_mode == "HWC":
                img = rearrange(img, "C H W -> H W C")
            return MultiplexTissue(
                image=img,
                tissue_id=tissue_id,
                measured_mask=tile.measured_mask,
                image_loading_mask=image_loading_mask,
                channel_names=channel_names,
                uniprot_ids=uniprot_ids,
                kind="tile",
            )
        if image_mode == "HWC":
            tile.image = rearrange(tile.image, "C H W -> H W C")
        return tile



    def get_tile(self, tissue_id: str, tile_id: int, kind="uniprot_filtered", image_mode="CHW", preprocess=True,) -> MultiplexTissue:
        """
        Get a specific tile based on the tissue id and tile id
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_id (int): The tile ID to retrieve.
            preprocess (bool): Whether to preprocess the tile using the standardizer. Default is True.
            kind (str): The kind of tissue image to retrieve for the tile. Valid options are "complete", "qc_filtered", and "uniprot_filtered". Default is "uniprot_filtered".
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            MultiplexTissue: The specific tile as an MultiplexTissue instance.
        """
        if self.tile_coordinates is None: # fallback
            C, H, W = self._get_tissue_size(tissue_id)
            col = np.random.randint(0, W - self.tile_size)
            row = np.random.randint(0, H - self.tile_size)
        else:
            row, col = self.tile_coordinates[tissue_id][tile_id]

        return self.get_tile_by_coordinates(tissue_id, row, col, preprocess=preprocess, kind=kind, image_mode=image_mode)
        
        
    
    def _get_tile_all_channels(self, tissue_id: str, tile_x: int, tile_y: int) -> MultiplexTissue:
        """
        Get the full tissue tile without filtering channels for a given tissue id and tile coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_x (int): The x coordinate of the top-left corner of the tile.
            tile_y (int): The y coordinate of the top-left corner of the tile.
        Returns:
            MultiplexTissue: Data class containing the full tissue tile as a torch.Tensor of shape (C, tile_size, tile_size) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        tile = torch.from_numpy(zarr.open(img_path, mode='r')[:, tile_y:tile_y+self.tile_size, tile_x:tile_x+self.tile_size]).float()
        return MultiplexTissue(
            image=tile,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=np.ones(measured_mask.sum(), dtype=bool),
            channel_names=self.get_channel_names(tissue_id, kind="complete", measured_mask=measured_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="complete", measured_mask=measured_mask),
        )
    
    def _get_tile_qc_filtered(self, tissue_id: str, tile_x: int, tile_y: int) -> MultiplexTissue:
        """
        Get the tissue tile filtered by quality control for a given tissue id and tile coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for. 
            tile_x (int): The x coordinate of the top-left corner of the tile.
            tile_y (int): The y coordinate of the top-left corner of the tile.
        Returns:
            MultiplexTissue: Data class containing the quality control filtered tissue tile as a torch.Tensor of shape (C, tile_size, tile_size) and the tissue ID.
        """ 
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        image_loading_mask = self.quality_control_mask[measured_mask]
        tile = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask), tile_y:tile_y+self.tile_size, tile_x:tile_x+self.tile_size]).float()
        qc_mask = self.quality_control_mask & measured_mask
        return MultiplexTissue(
            image=tile,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask,
            channel_names=self.get_channel_names(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
        )
    
    def _get_tile_uniprot_filtered(self, tissue_id: str, tile_x: int, tile_y: int) -> MultiplexTissue:
        """
        Get the tissue tile filtered by quality control and valid UniProt availability for a given tissue id and tile coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_x (int): The x coordinate of the top-left corner of the tile.
            tile_y (int): The y coordinate of the top-left corner of the tile.
        Returns:
            MultiplexTissue: Data class containing the filtered tissue tile as a torch.Tensor of shape (C, tile_size, tile_size) and the tissue ID.
        """ 
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        qc_mask = self.quality_control_mask & measured_mask
        filtered_mask = self.uniprot_mask & qc_mask
        image_loading_mask = filtered_mask[measured_mask]
        tile = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask), tile_y:tile_y+self.tile_size, tile_x:tile_x+self.tile_size]).float()
        return MultiplexTissue(
            image=tile,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask, 
            channel_names=self.get_channel_names(tissue_id, kind="uniprot_filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
            uniprot_ids=self.get_uniprot_ids(tissue_id, kind="uniprot_filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
        )

