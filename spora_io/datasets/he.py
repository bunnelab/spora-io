from __future__ import annotations

import os
from typing import Optional, Tuple
import torch
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from einops import rearrange
from loguru import logger 
from PIL import Image
from pathlib import Path
import zarr

from spora_io.datasets.base import BaseImagingDataset
from spora_io.utils.utils import is_rank0, print_verbose
from spora_io.datasets._types import HETissue, TissueMask, CellMask

max_width = int(os.environ.get("MAX_HE_WIDTH", 50000))
max_height = int(os.environ.get("MAX_HE_HEIGHT", 50000))
Image.MAX_IMAGE_PIXELS = max_width * max_height


class HEImagingDataset(BaseImagingDataset):
    """
    Class for handling H&E stained imaging datasets.

    Attributes:
        IMAGENET_MEAN (torch.Tensor): The mean values for ImageNet normalization.
        IMAGENET_STD (torch.Tensor): The standard deviation values for ImageNet normalization.
        HIBOU_MEAN (torch.Tensor): The mean values for HIBOU normalization.
        HIBOU_STD (torch.Tensor): The standard deviation values for HIBOU normalization.
        IDENTITY_MEAN (torch.Tensor): The mean values for identity normalization.
        IDENTITY_STD (torch.Tensor): The standard deviation values for identity normalization.
        mean_std_type (str): The type of mean and standard deviation to use for normalization. Valid options are "imagenet", "hibou", and "identity".
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.722])[:, None, None]
    HIBOU_STD = torch.tensor([0.195, 0.2316, 0.1816])[:, None, None]
    IDENTITY_MEAN = torch.tensor([0.0, 0.0, 0.0])[:, None, None]
    IDENTITY_STD = torch.tensor([1.0, 1.0, 1.0])[:, None, None]

    def __init__(self,
                 name: str,
                 resolution: float | str,
                 path: os.PathLike | str | None = None,
                 load_cell_metadata: bool = False,
                 verbose: bool = True,
                 mean_std_type: str = "imagenet",
                 tile_size: Optional[int] = None,
                 tile_strategy: Optional[str] = None,
                 split: Optional[str] = None,
                 **kwargs
    ):
        super().__init__(
            name=name,
            path=path,
            modality="he",
            resolution=resolution,
            tile_size=tile_size,
            load_cell_metadata=load_cell_metadata,
            verbose=verbose,
            tile_strategy=tile_strategy,
            split=split,
            **kwargs,
        )
        self.mean_std_type = mean_std_type

        self.img_folder = self.path / self.modality.canonical_dir / self.resolution / "images"
        assert self.img_folder.exists(), f"Image folder {self.img_folder} does not exist."

        if self.mean_std_type == "imagenet":
            self.mean = self.IMAGENET_MEAN
            self.std = self.IMAGENET_STD
        elif self.mean_std_type == "hibou":
            self.mean = self.HIBOU_MEAN
            self.std = self.HIBOU_STD
        elif self.mean_std_type == "identity":
            self.mean = self.IDENTITY_MEAN
            self.std = self.IDENTITY_STD
        else:
            raise ValueError(f"Invalid mean_std_type {self.mean_std_type}. Valid options are 'imagenet', 'hibou', and 'identity'.")

        self._try_to_load_tile_coords()

    def _get_tissue_all_channels(self, tissue_id: str, kind: str = "complete", preprocess: bool=False, image_mode: str = "CHW") -> HETissue:
        """
        Get the full tissue image without filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Only "complete" is supported for H&E datasets since there is only one modality channel. Default is "complete".
            preprocess (bool): If True, preprocess the image (normalize). Default is False.
        Returns:
            HETissue: The full tissue image as an HETissue instance.
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        img = torch.from_numpy(zarr.open(img_path, mode='r')[:]).float()
        if image_mode == "HWC":
            img = rearrange(img, "C H W -> H W C")
        if preprocess:
            img = self._preprocess(img)
        return HETissue(
            image=img,
            tissue_id=tissue_id
        )
    
    def _preprocess(self, img: NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        """
        Preprocess the image by standardizing it.
        Args:
            img (NDArray[np.float32] | torch.Tensor): The image to preprocess.
        Returns:
            torch.Tensor: The preprocessed image.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img / 255.0).float() 
        else:
            img = img / 255.0
        img = (img - self.mean) / self.std
        return img 

    def get_tissue(self, tissue_id: str, kind: str = "complete", preprocess: bool = True, image_mode="CHW") -> HETissue:
        """
        Get the normalized tissue image post filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Only "complete" is supported for H&E datasets since there is only one modality channel. Default is "complete".
            preprocess (bool): If True, preprocess the image (normalize). Default is True.
        Returns:
            HETissue: The normalized tissue image as an HETissue instance.
        """
        return self._get_tissue_all_channels(tissue_id, kind=kind, preprocess=preprocess, image_mode=image_mode)
    
    def _get_tissue_size(self, tissue_id: str) -> Tuple[int, int, int]:
        """
        Get the tissue size (C,H,W) for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the size for.
        Returns:
            Tuple[int, int, int]: The tissue size as a tuple (C, H, W).
        """
        img_path = self.img_folder / f"{tissue_id}.ome.zarr" / "0"
        img = zarr.open(img_path, mode='r')
        return img.shape[0], img.shape[1], img.shape[2] #type: ignore

    def get_tile_by_coordinates(self, tissue_id: str, row: int, col: int,
                 kind: str = "complete",
                 image_mode: str = "CHW",
                 preprocess: bool = True) -> HETissue:
        """
        Get a specific tile based on the tissue id and tile id
        Args:            
            tissue_id (str): The tissue ID to retrieve the tile for.
            row (int): The row coordinate of the tile.
            col (int): The column coordinate of the tile.
            kind (str): The kind of tile image to retrieve. Default is "complete".
            image_mode (str): The image mode of the tile image. Valid options are "CHW" and "HWC". Default is "CHW".
            preprocess (bool): If True, preprocess the image (normalize). Default is True.
        Returns:
            HETissue: The specific tile as an HETissue instance.
        """
        img = zarr.open(self.img_folder / f"{tissue_id}.ome.zarr" / "0", mode='r')
        tile = self._load_padded_tile_chw(img, row, col)
        if image_mode == "HWC":
            tile = rearrange(tile, "C H W -> H W C")
        if preprocess:
            tile = self._preprocess(tile)
        return HETissue(
            image=tile,
            tissue_id=tissue_id,
            channels="RGB",
            kind="tile"
        )


    def get_tile(self, tissue_id: str, tile_id: int,
                 kind: str = "complete",
                 image_mode: str = "CHW",
                 preprocess: bool = True) -> HETissue:
        """
        Get a specific tile based on the tissue id and tile id
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_id (int): The tile ID to retrieve.
        Returns:
            HETissue: The specific tile as an HETissue instance.
        """
        if self.tile_coordinates is None: # fallback
            C, H, W = self._get_cached_tissue_size(tissue_id)
            row = np.random.randint(0, H - self.tile_size)
            col = np.random.randint(0, W - self.tile_size)
        else:
            row, col = self.tile_coordinates[tissue_id][tile_id]
        
        return self.get_tile_by_coordinates(tissue_id, row, col, preprocess=preprocess, image_mode=image_mode)
    

