from __future__ import annotations

import os
from typing import Tuple
import torch
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from einops import rearrange
from loguru import logger 
from PIL import Image
from pathlib import Path
import zarr

from spatialprot_data.datasets.base import BaseImagingDataset
from spatialprot_data.utils.utils import is_rank0, print_verbose
from spatialprot_data.datasets._types import HETissue, TissueMask, CellMask

max_width = int(os.environ.get("MAX_HE_WIDTH", 50000))
max_height = int(os.environ.get("MAX_HE_HEIGHT", 50000))
Image.MAX_IMAGE_PIXELS = max_width * max_height


class HEImagingDataset(BaseImagingDataset):
    """
    Class for handling H&E stained imaging datasets.
    """
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    HIBOU_MEAN = torch.tensor([0.7068, 0.5755, 0.722])[:, None, None]
    HIBOU_STD = torch.tensor([0.195, 0.2316, 0.1816])[:, None, None]

    def __init__(self,
                 name: str,
                 path: os.PathLike | str,
                 resolution: float | str,
                 crop_size: int,
                 load_cell_metadata: bool = False,
                 verbose: bool = True,
                 mean_std_type: str = "imagenet",
                 **kwargs
    ):
        super().__init__(
            name=name,
            path=path,
            modality="he",
            resolution=resolution,
            load_cell_metadata=load_cell_metadata,
            verbose=verbose,
            **kwargs,
        )
        self.mean_std_type = mean_std_type
        self.crop_size = crop_size

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

        self._try_to_load_crop_coords()

    def _get_tissue_all_channels(self, tissue_id: str, preprocess: bool=False, image_mode: str = "CHW") -> HETissue:
        """
        Get the full tissue image without filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            preprocess (bool): If True, preprocess the image (normalize). Default is False.
        Returns:
            HETissue: The full tissue image as an HETissue instance.
        """
        img_path = self.img_folder / f"{tissue_id}.zarr"
        img = torch.from_numpy(zarr.open(img_path, mode='r')[:]).float()
        if image_mode == "CHW":
            img = rearrange(img, "H W C -> C H W")
        if preprocess:
            img = self._preprocess(img)
        return HETissue(
            tissue=img,
            tissue_id=tissue_id
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
        return img # type: ignore

    def get_tissue(self, tissue_id: str, kind: str = "complete", preprocess: bool = True, image_mode: str = "CHW") -> HETissue:
        """
        Get the normalized tissue image post filtering channels for a given tissue id.
        Args:
            tissue_id (str): The tissue ID to retrieve the image for.
            kind (str): The kind of tissue image to retrieve. Default is "complete". Valid options are "complete", "qc_filtered", and "filtered".
            For H&E datasets, "qc_filtered" and "filtered" will return the same image since there is only one modality channel.
            preprocess (bool): If True, preprocess the image (normalize). Default is True.
        Returns:
            HETissue: The normalized tissue image as an HETissue instance.
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
        if image_mode == "CHW":
            return img.shape[2], img.shape[0], img.shape[1] #type: ignore
        else:
            return img.shape[0], img.shape[1], img.shape[2] #type: ignore

    def get_crop(self, tissue_id: str, crop_id: int,
                 image_mode: str = "CHW",
                 preprocess: bool = True) -> HETissue:
        """
        Get a specific crop based on the tissue id and crop id
        Args:
            tissue_id (str): The tissue ID to retrieve the crop for.
            crop_id (int): The crop ID to retrieve.
        Returns:
            HETissue: The specific crop as an HETissue instance.
        """
        if self.crop_coordinates is None: # fallback
            C, H, W = self._get_tissue_size(tissue_id)
            row = np.random.randint(0, H - self.crop_size)
            col = np.random.randint(0, W - self.crop_size)
        else:
            row, col = self.crop_coordinates[tissue_id][crop_id]
        crop = torch.from_numpy(
            zarr.open(self.img_folder / f"{tissue_id}.zarr", mode='r')[row:row+self.crop_size, col:col+self.crop_size, :] # type: ignore
        ).float()
        if image_mode == "CHW":
            crop = rearrange(crop, "H W C -> C H W")
        if preprocess:
            crop = self._preprocess(crop)
        return HETissue(
            tissue=crop,
            tissue_id=tissue_id,
            crop_id=crop_id,
            channels="RGB",
            kind="crop"
        )
    




