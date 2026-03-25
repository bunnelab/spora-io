import os
import torch
import numpy as np
from numpy.typing import NDArray
from einops import rearrange
from loguru import logger 
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import zarr

from spatialprot_data.datasets.base import BaseImagingDataset
from spatialprot_data.utils.dataset.normalize import build_normalizer
from spatialprot_data.utils.utils import print_verbose
from spatialprot_data.utils.dataset.transforms import FilterFactory
from spatialprot_data.datasets._types import MultiplexTissue, TissueMask, CellMask


class MultiplexImagingDataset(BaseImagingDataset):
    """
    Class for handling multiplex imaging datasets.
    """
    VALID_MODALITIES = {"imc", "codex", "cycif"}
    def __init__(self,
                 name: str,
                 path: os.PathLike | str,
                 modality: str,
                 normalization: str, 
                 resolution: float | str,
                 crop_size: int,
                 verbose: bool = True,
                 load_cell_metadata: bool = False,
                 marker_embedding_type: str = "esm",
                 disable_quantile_mask: bool = True,
                 filter_list: List[str] | None = None,
                 **kwargs
    ):
        assert modality in self.VALID_MODALITIES, f"Invalid modality {modality}. Valid options are: {self.VALID_MODALITIES}"
        super().__init__(name=name, path=path, modality=modality,
                         resolution=resolution, load_cell_metadata=load_cell_metadata, verbose=verbose)
        self.crop_size = crop_size
        self.kwargs = kwargs
        marker_embedding_dir = self.path.parent / "marker_embeddings"
        self.img_folder = self.path / self.modality.canonical_dir / self.resolution #type: ignore

        self.channel_list = pd.read_parquet(self.path / self.modality.canonical_dir / "channels.parquet")

        if "qc_pass" not in self.channel_list.columns:
            print_verbose(f"No 'qc_pass' column found in channel list at {self.path / self.modality.canonical_dir / 'channels.parquet'}. All channels will be considered as passing quality control.", level="WARNING")
            self.channel_list["qc_pass"] = True
        self.quality_control_mask = self.channel_list["qc_pass"].to_numpy(dtype=bool)

        if marker_embedding_type == "esm":
            from spatialprot_data.utils.dataset.marker import load_esm_marker_embedding_dict
            esm_model_name = self.kwargs.get("esm_model_name", "esm2_t30_150M_UR50D")
            dir_name = esm_model_name
            self.marker_index_map = load_esm_marker_embedding_dict(marker_embedding_dir / dir_name)
            self.channel_list_column = "uniprot_id"
        else:
            raise NotImplementedError(f"Marker embedding type {marker_embedding_type} not implemented. Currently only 'esm' is supported.")

        self.marker_embedding_dir = marker_embedding_dir / dir_name


        filter_params = self.kwargs.get("filter_params", {})
        self._set_optional_filters(filter_list if filter_list is not None else [], filter_params=filter_params)
        # Load gene dictionary if provided
        

        self.image_channel_map = None
        default_image_channel_map_path = self.path / self.modality.canonical_dir / f"channels_per_tissue.parquet"
        if default_image_channel_map_path.exists():
            self.image_channel_map = pd.read_parquet(default_image_channel_map_path)
            self.image_channel_map.set_index("tissue_id", inplace=True)
            if self.verbose:
                print_verbose(f"Using image-channel map from {default_image_channel_map_path}")
        else:
            if self.verbose:
                print_verbose(f"No image-channel map found at {default_image_channel_map_path}. Proceeding to create with all channels included.",
                              level="WARNING")
            
            self.image_channel_map = pd.DataFrame(
                index=self.tissue_metadata.index,
                columns=self.channel_list[self.channel_list_column].values,
            )
            self.image_channel_map.fillna(1, inplace=True)  # Include all channels by default

        self.image_channel_map.replace(0, False, inplace=True)
        self.image_channel_map.replace(1, True, inplace=True)

        
        self.normalizer = build_normalizer(
            normalization=normalization,
            modality_dir=self.path / self.modality.canonical_dir / self.resolution,
            channels_per_image=self.image_channel_map,
            disable_quantile_mask=disable_quantile_mask,
            filter_factory=self.filter_factory
        )

        if self.verbose:
            print_verbose(f"Using Multiplex normalization: {self.normalizer.__class__.__name__}")
        # generating marker indices
        self._index_channel_embeddings()

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

        
    def _index_channel_embeddings(self) -> None:
        """
        Compute:
        - self.marker_indices: np.int64[ n_with_embed ]
                Row indices into the embedding matrix (aligned to markers that *have* embeddings).
        - self.channel_indices_with_embedding: np.int64[ n_with_embed ]
                Row indices into self.channel_list for which an embedding exists.
        - self.mask_channels_with_embeddings: np.bool_[ n_all ]
                Boolean mask over all channels: True iff embedding was found.

        Requires:
        - self.channel_list with columns ["name", "protein_id"]
        - self.uniprot_to_index: Dict[str, int] mapping UniProt ID -> embedding row index
        """
        # df = self.gene_dict.reset_index(drop=True)
        df = self.channel_list.reset_index(drop=True)

        self.idx_series = df[self.channel_list_column].map(self.marker_index_map)

        # Boolean mask over *all* channels.
        mask = self.idx_series.notna().to_numpy(dtype=bool)

        # Channel indices (rows in gene_dict) that *have* embeddings.
        channel_idxs = np.flatnonzero(mask).astype(np.int64)

        # Embedding matrix row indices aligned to the above channels.
        # Safe to astype(int) after masking (no NaNs).
        marker_idxs = self.idx_series[mask].astype(int).to_numpy(dtype=np.int64)

        # Save
        # boolean mask for which embeddings exist
        self.mask_channels_with_embeddings = mask
        # channel indices for which embeddings exist (0..n-1) indexed
        self.channel_indices_with_embedding = channel_idxs
        # corresponding embedding row indices for those channels from index map
        self.marker_indices = marker_idxs

        # Logging
        if self.verbose:
            n_all = len(df)
            n_found = int(mask.sum())
            n_miss = n_all - n_found
            print_verbose(f"Found embeddings for {n_found}/{n_all} markers ({n_found/n_all:.1%}).")
            if n_miss:
                print_verbose(
                    f"Embeddings missing for {n_miss} markers.",
                    level="WARNING"
                )

    def _get_tissue_all_channels(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the full tissue image without filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
        Returns:
            MultiplexTissue: Data class containing the full tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        img = torch.from_numpy(zarr.open(img_path, mode='r')[:]).float()
        return MultiplexTissue(
            tissue=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=np.ones(measured_mask.sum(), dtype=bool), # for the unfiltered image, all measured channels are loaded
            channel_names=self.get_channel_names(tissue_id, kind="complete", measured_mask=measured_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="complete", measured_mask=measured_mask)
        )
    
    def _get_tissue_qc_filtered(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the tissue image filtered by quality control for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for. 
        Returns:
            MultiplexTissue: Data class containing the quality control filtered tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        # qc_mask = self.quality_control_mask & measured_mask
        # this doesn't work when loading the image because the image has only measured_mask=True channels,
        # but measured_mask and qc_mask are defined over the full channel set. So we need to apply the measured_mask to the qc_mask to get the correct channels to load from the image.
        image_loading_mask = self.quality_control_mask[measured_mask]
        img = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask)]).float()
        qc_mask = self.quality_control_mask & measured_mask
        return MultiplexTissue(
            tissue=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask,
            channel_names=self.get_channel_names(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask)
        )
    
    def _get_tissue_filtered(self, tissue_id: str) -> MultiplexTissue:
        """
        Get the tissue image filtered by quality control and embedding availability for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
        Returns:
            MultiplexTissue: Data class containing the filtered tissue image as a torch.Tensor of shape (C, H, W) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        qc_mask = self.quality_control_mask & measured_mask
        filtered_mask = self.mask_channels_with_embeddings & qc_mask 
        # similar to above, we need to apply the measured_mask to the filtered_mask to get the correct channels to load from the image.
        image_loading_mask = filtered_mask[measured_mask]
        img = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask)]).float()
        return MultiplexTissue(
            tissue=img,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask, 
            channel_names=self.get_channel_names(tissue_id, kind="filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask)
        )




    def get_channel_names(self, tissue_id: str, kind: str = "complete", 
                          measured_mask=None, qc_mask=None, filtered_mask=None) -> NDArray[np.str_]:
        """
        Get the channel names for a given tissue id and kind.
        Args:
            tissue_id (str): The tissue ID to retrieve the channel names for.
            kind (str): The kind of tissue image to retrieve channel names for. Default is "complete". Valid options are "complete", "qc_filtered", and "filtered".
            For H&E datasets, "qc_filtered" and "filtered" will return the same channel names since there is only one modality channel.
        Returns:
            NDArray[np.str_]: The channel names as a 1D array of shape (n_channels,).
        """
        if measured_mask is None:
            measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        if kind == "complete":
            return self.channel_list["channel_name"][measured_mask].values
        elif kind == "qc_filtered":
            if qc_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
            return self.channel_list["channel_name"][qc_mask].values
        elif kind == "filtered":
            if filtered_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
                filtered_mask = self.mask_channels_with_embeddings & qc_mask 
            return self.channel_list["channel_name"][filtered_mask].values
        

    def get_channel_indices(self, tissue_id: str, kind: str = "complete",
                            measured_mask=None, qc_mask=None, filtered_mask=None) -> NDArray[np.int_]:
        """
        Get the channel indices for a given tissue id and kind. This corresponds to the marker_embedding_type channel indices.
        For filtered kind, this will return the channels with embeddings that also pass quality control.
        However for qc_filtered and complete, there might be channels that have no embeddings: for those it returns np.nan to indicate missing ebeddings.
        Args:
            tissue_id (str): The tissue ID to retrieve the channel indices for.
            kind (str): The kind of tissue image to retrieve channel indices for. Default is "complete". Valid options are "complete", "qc_filtered", and "filtered".
        Returns:
            NDArray[np.int_]: The channel indices as a 1D array of shape (n_channels,).
        """

        if measured_mask is None:
            measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)  
        if kind == "complete":
            return self.idx_series[measured_mask].values.astype(np.int64)
        elif kind == "qc_filtered":
            if qc_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
            return self.idx_series[qc_mask].values.astype(np.int64)
        elif kind == "filtered":
            if filtered_mask is None:
                qc_mask = self.quality_control_mask & measured_mask
                filtered_mask = self.mask_channels_with_embeddings & qc_mask 
            return self.idx_series[filtered_mask].values.astype(np.int64)

    def get_tissue(self, tissue_id: str, kind="filtered", preprocess=True, image_mode="CHW") -> MultiplexTissue:
        """ 
        Get the tissue image for a given tissue id, with options for filtering channels and preprocessing.
        """ 
        if kind == "complete":
            tissue = self._get_tissue_all_channels(tissue_id)
        elif kind == "qc_filtered":
            tissue = self._get_tissue_qc_filtered(tissue_id)
        elif kind == "filtered":
            tissue = self._get_tissue_filtered(tissue_id)
        else:
            raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'filtered'.")

        if preprocess:
            img, refined_mask = self.normalizer.apply(tissue.tissue, tissue_id, tissue.measured_mask, tissue.image_loading_mask)
            return MultiplexTissue(
                tissue=img,
                tissue_id=tissue_id,
                measured_mask=tissue.measured_mask,
                image_loading_mask=tissue.image_loading_mask,
                channel_names=tissue.channel_names,
                channel_idxs=tissue.channel_idxs
            )
        else:
            return tissue

    def _get_tissue_size(self, tissue_id: str, image_mode: str = "CHW") -> Tuple[int, int, int]:
        img_path = self.img_folder / f"{tissue_id}.zarr"
        img = zarr.open(img_path, mode='r')
        if image_mode == "CHW":
            return img.shape[0], img.shape[1], img.shape[2] #type: ignore
        else:
            return img.shape[2], img.shape[0], img.shape[1] #type: ignore



    def get_crop(self, tissue_id: str, crop_id: int, preprocess=True, kind="filtered") -> MultiplexTissue:
        """
        Get a specific crop based on the tissue id and crop id
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for.
            crop_id (int): The crop ID to retrieve.
        Returns:
            MultiplexTissue: The specific crop as an MultiplexTissue instance.
        """
        # Currently it is a placeholder function, since we don't pre-generate the crops
        # Instead we can just get a random crop
        C, H, W = self._get_tissue_size(tissue_id)
        x = np.random.randint(0, W - self.crop_size)
        y = np.random.randint(0, H - self.crop_size)
        if kind == "complete":
            crop = self._get_crop_all_channels(tissue_id, x, y)
        elif kind == "qc_filtered":
            crop = self._get_crop_qc_filtered(tissue_id, x, y)
        elif kind == "filtered":
            crop = self._get_crop_filtered(tissue_id, x, y)
        else:
            raise ValueError(f"Invalid kind {kind}. Valid options are: 'complete', 'qc_filtered', 'filtered'.")

        if preprocess:
            img, refined_mask = self.normalizer.apply(crop.tissue, tissue_id, crop.measured_mask, crop.image_loading_mask)
            return MultiplexTissue(
                tissue=img,
                tissue_id=tissue_id,
                measured_mask=crop.measured_mask,
                image_loading_mask=crop.image_loading_mask,
                channel_names=crop.channel_names,
                channel_idxs=crop.channel_idxs,
                kind="crop",
                crop_id=crop_id
            )
        else:
            return crop
        
    
    def _get_crop_all_channels(self, tissue_id: str, crop_x: int, crop_y: int) -> MultiplexTissue:
        """
        Get the full tissue crop without filtering channels for a given tissue id and crop coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for.
            crop_x (int): The x coordinate of the top-left corner of the crop.
            crop_y (int): The y coordinate of the top-left corner of the crop.
        Returns:
            MultiplexTissue: Data class containing the full tissue crop as a torch.Tensor of shape (C, crop_size, crop_size) and the tissue ID.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        crop = torch.from_numpy(zarr.open(img_path, mode='r')[:, crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]).float()
        return MultiplexTissue(
            tissue=crop,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=np.ones(measured_mask.sum(), dtype=bool), # for the unfiltered crop, all measured channels are loaded
            channel_names=self.get_channel_names(tissue_id, kind="complete", measured_mask=measured_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="complete", measured_mask=measured_mask)
        )
    
    def _get_crop_qc_filtered(self, tissue_id: str, crop_x: int, crop_y: int) -> MultiplexTissue:
        """
        Get the tissue crop filtered by quality control for a given tissue id and crop coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for. 
            crop_x (int): The x coordinate of the top-left corner of the crop.
            crop_y (int): The y coordinate of the top-left corner of the crop.
        Returns:
            MultiplexTissue: Data class containing the quality control filtered tissue crop as a torch.Tensor of shape (C, crop_size, crop_size) and the tissue ID.
        """ 
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        # qc_mask = self.quality_control_mask & measured_mask
        # this doesn't work when loading the image because the image has only measured_mask=True channels,
        # but measured_mask and qc_mask are defined over the full channel set. So we need to apply the measured_mask to the qc_mask to get the correct channels to load from the image.
        image_loading_mask = self.quality_control_mask[measured_mask]
        crop = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask), crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]).float()
        qc_mask = self.quality_control_mask & measured_mask
        return MultiplexTissue(
            tissue=crop,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask,
            channel_names=self.get_channel_names(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="qc_filtered", measured_mask=measured_mask, qc_mask=qc_mask)
        )
    
    def _get_crop_filtered(self, tissue_id: str, crop_x: int, crop_y: int) -> MultiplexTissue:
        """
        Get the tissue crop filtered by quality control and embedding availability for a given tissue id and crop coordinates.
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for.
            crop_x (int): The x coordinate of the top-left corner of the crop.
            crop_y (int): The y coordinate of the top-left corner of the crop.
        Returns:
            MultiplexTissue: Data class containing the filtered tissue crop as a torch.Tensor of shape (C, crop_size, crop_size) and the tissue ID.
        """ 
        img_path = self.img_folder / f"{tissue_id}.zarr"
        measured_mask = self.image_channel_map.loc[tissue_id].to_numpy(dtype=bool)
        qc_mask = self.quality_control_mask & measured_mask
        filtered_mask = self.mask_channels_with_embeddings & qc_mask 
        # similar to above, we need to apply the measured_mask to the filtered_mask to get the correct channels to load from the image.
        image_loading_mask = filtered_mask[measured_mask]
        crop = torch.from_numpy(zarr.open(img_path, mode='r')[np.flatnonzero(image_loading_mask), crop_y:crop_y+self.crop_size, crop_x:crop_x+self.crop_size]).float()
        return MultiplexTissue(
            tissue=crop,
            tissue_id=tissue_id,
            measured_mask=measured_mask,
            image_loading_mask=image_loading_mask, 
            channel_names=self.get_channel_names(tissue_id, kind="filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask),
            channel_idxs=self.get_channel_indices(tissue_id, kind="filtered", measured_mask=measured_mask, qc_mask=qc_mask, filtered_mask=filtered_mask)
        )

    def _count_crops(self, crop_folder_path: str | None = None) -> int:
        """
        Count the number of crops in the dataset.
        Args:
            crop_folder_path (str | None): The path to the crop folder. If None, uses the default crop folder.
        Returns:
            int: The number of crops in the dataset.
        """
        return 0