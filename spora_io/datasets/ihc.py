from __future__ import annotations

import os
from typing import Tuple, Optional
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
from spora_io.datasets._types import IHCTissue, TissueMask, CellMask, IHCModality

max_width = int(os.environ.get("MAX_HE_WIDTH", 50000))
max_height = int(os.environ.get("MAX_HE_HEIGHT", 50000))
Image.MAX_IMAGE_PIXELS = max_width * max_height


class SingleIHCImagingDataset(BaseImagingDataset):
    """
    Class for handling IHC stained imaging datasets.
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.722])[:, None, None]
    HIBOU_STD = torch.tensor([0.195, 0.2316, 0.1816])[:, None, None]

    def __init__(self,
                 name: str,
                 path: os.PathLike | str,
                 marker_name: str,
                 resolution: float | str,
                 tile_size: int, 
                 load_cell_metadata: bool = False,
                 verbose: bool = True,
                 mean_std_type: str = "imagenet",
                 tile_strategy: Optional[str] = None,
                 **kwargs
    ):
        self.marker_name = marker_name
        if not self.marker_name.startswith("ihc_"):
            self.marker_name = f"ihc_{self.marker_name}"
        super().__init__(
            name=name,
            path=path,
            modality=IHCModality(name=self.marker_name, canonical_dir=f"ihc/{self.marker_name}"), 
            resolution=resolution,
            tile_size=tile_size,
            load_cell_metadata=load_cell_metadata,
            verbose=verbose,
            tile_strategy=tile_strategy,
        )
        self.mean_std_type = mean_std_type

        self.img_folder = self.path / self.modality.canonical_dir / self.resolution #type: ignore
        assert self.img_folder.exists(), f"Image folder {self.img_folder} does not exist."

        if self.mean_std_type == "imagenet":
            self.mean = self.IMAGENET_MEAN
            self.std = self.IMAGENET_STD
        elif self.mean_std_type == "hibou":
            self.mean = self.HIBOU_MEAN
            self.std = self.HIBOU_STD
        else:
            raise ValueError(f"Invalid mean_std_type {self.mean_std_type}. Valid options are 'imagenet' and 'hibou'.")

        self._try_to_load_tile_coords()

    def _get_tissue_all_channels(self, tissue_id: str, preprocess: bool=False, image_mode: str = "CHW") -> IHCTissue:
        """
        Get the full tissue image without filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            preprocess (bool): If True, preprocess the image (normalize). Default is False.
        Returns:
            IHCTissue: The full tissue image as an IHCTissue instance.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        img = torch.from_numpy(zarr.open(img_path, mode='r')[:]).float()
        if image_mode == "HWC":
            img = rearrange(img, "C H W -> H W C")
        if preprocess:
            img = self._preprocess(img)
        return IHCTissue(
            tissue=img,
            tissue_id=tissue_id,
            channels=self.modality.name.replace("ihc_", "")
        )
    
    def _preprocess(self, img: NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        """
        Preprocess the image by normalizing it.
        Args:
            img (NDArray[np.float32] | torch.Tensor): The image to preprocess.
        Returns:
            torch.Tensor: The preprocessed image.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img / 255.0).float() # type: ignore
        else:
            img = img / 255.0
        img = (img - self.mean) / self.std
        return img 

    def get_tissue(self, tissue_id: str, kind: str = "complete", preprocess: bool = True, image_mode: str = "CHW") -> IHCTissue:
        """
        Get the normalized tissue image post filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Default is "complete". Valid options are "complete", "qc_filtered", and "filtered".
            For H&E datasets, "qc_filtered" and "filtered" will return the same image since there is only one modality channel.
            preprocess (bool): If True, preprocess the image (normalize). Default is True.
        Returns:
            IHCTissue: The normalized tissue image as an IHCTissue instance.
        """
        return self._get_tissue_all_channels(tissue_id, preprocess=preprocess, image_mode=image_mode)
    
    def _get_tissue_size(self, tissue_id: str, image_mode: str = "CHW") -> Tuple[int, int, int]:
        """
        Get the tissue size (C,H,W) for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the size for.
            image_mode (str): The image mode of the tissue image. Valid options are "CHW" and "HWC". Default is "CHW".
        Returns:
            Tuple[int, int, int]: The tissue size as a tuple (C, H, W).
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        img = zarr.open(img_path, mode='r')
        return img.shape[0], img.shape[1], img.shape[2] #type: ignore

    def get_tile(self, tissue_id: str, tile_id: int,
                 image_mode: str = "CHW",
                 preprocess: bool = True) -> IHCTissue:
        """
        Get a specific tile based on the tissue id and tile id
        Args:
            tissue_id (str): The tissue ID to retrieve the tile for.
            tile_id (int): The tile ID to retrieve.
        Returns:
            IHCTissue: The specific tile as an IHCTissue instance.
        """
        if self.tile_coordinates is None: # fallback
            C, H, W = self._get_tissue_size(tissue_id)
            col = np.random.randint(0, W - self.tile_size)
            row = np.random.randint(0, H - self.tile_size)
        else:
            row, col = self.tile_coordinates[tissue_id][tile_id]
        tile = torch.from_numpy(
            zarr.open(self.img_folder / f"{tissue_id}.zarr", mode='r')[row:row+self.tile_size, col:col+self.tile_size, :] # type: ignore
        ).float()
        if image_mode == "HWC":
            tile = rearrange(tile, "C H W -> H W C")
        if preprocess:
            tile = self._preprocess(tile)
        return IHCTissue(
            tissue=tile,
            tissue_id=tissue_id,
            channels="RGB",
            kind="tile"
        )
    




# class IHCImagingDataset