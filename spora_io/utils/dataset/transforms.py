from __future__ import annotations

import numpy as np
import random
import torch
from torchvision.transforms import v2
import math
from einops import rearrange
from loguru import logger
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List

class CustomGaussianBlur(object):

    def __init__(self, kernel_size, sigma):
        self.transform = v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            return self.transform(img).numpy()
        else:
            return self.transform(img)
        


def custom_median_filter(input_tensor: torch.Tensor, kernel_size: int = 3, padding: str = 'reflect') -> torch.Tensor:
    """
    Applies a median filter to a 4D input tensor (batch, channels, height, width).
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        kernel_size (int): Size of the kernel (must be odd, e.g., 3, 5, 7)
        padding (str): Padding mode ('reflect', 'replicate', or 'constant')
    
    Returns:
        torch.Tensor: Filtered tensor of the same shape as input
    """ 
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if missing
        SHAPE_ADDED = True
    else:
        SHAPE_ADDED = False
    # Ensure kernel_size is odd
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    
    # Calculate padding
    pad = kernel_size // 2
    
    # Pad the input tensor
    padded = F.pad(input_tensor, (pad, pad, pad, pad), mode=padding)
    
    # Unfold the tensor to get all patches of size kernel_size x kernel_size
    # Shape after unfold: (B, C, H * W, kernel_size * kernel_size)
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    B, C, H, W, _, _ = patches.shape
    patches = patches.reshape(B, C, H, W, kernel_size * kernel_size)
    
    # Compute median along the last dimension (across the kernel)
    # Shape after median: (B, C, H, W)
    filtered = torch.median(patches, dim=-1).values
    if SHAPE_ADDED:
        filtered = filtered.squeeze(0)  # Remove batch dimension if it was added
    return filtered


class FilterFactory:
    def __init__(
        self,
        filters_to_apply: List[str],
        filter_params: Dict[str, Dict[str, Any]],
    ):
        self.filters_to_apply = filters_to_apply
        self.filter_params = filter_params

        for filter_name in self.filters_to_apply:
            if filter_name == "gaussian_blur":
                params = self.filter_params.get(filter_name, {})
                kernel_size = params.get("kernel_size", 3)
                sigma = params.get("sigma", 1.0)
                setattr(self, filter_name, CustomGaussianBlur(kernel_size=kernel_size, sigma=sigma))
            elif filter_name == "median_filter":
                params = self.filter_params.get(filter_name, {})
                kernel_size = params.get("kernel_size", 3)
                padding = params.get("padding", 'reflect')
                setattr(self, filter_name, lambda x, k=kernel_size, p=padding: custom_median_filter(x, kernel_size=k, padding=p))
            else:
                raise ValueError(f"Unsupported filter {filter_name} provided to FilterFactory.")
    
    def _ensure_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError(f"Input must be a numpy array or a torch tensor, but got {type(x)}.")

    def apply_filters(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        x_t = self._ensure_tensor(x)
        for filter_name in self.filters_to_apply:
            filter_fn = getattr(self, filter_name)
            x_t = filter_fn(x_t)
        return x_t    
    
    def print_filters(self):
        print(f"Filters to apply: {self.filters_to_apply}")