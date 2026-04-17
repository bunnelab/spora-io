from __future__ import annotations

import numpy as np
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from typing import Optional
from numpy.typing import ArrayLike
from spora_io.utils.dataset.transforms import CustomGaussianBlur


def load_tissue_and_mask(dataset, tissue_id: str):
    """
    Load tissue image and mask using the new MultiplexImagingDataset interface.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_id: str, tissue identifier
        method_name: str, normalization method name
        
    Returns:
        img: torch.Tensor or np.ndarray, image data (C, H, W)
        tissue_mask: np.ndarray, boolean mask (H, W)
        measured_mask: np.ndarray, boolean mask indicating which channels were measured
    """
    # Get tissue without preprocessing
    tissue_data = dataset.get_tissue(tissue_id, kind="complete", preprocess=False)
    
    # Get tissue mask
    tissue_mask_data = dataset.get_tissue_mask(tissue_id)
    
    # Convert to numpy if needed
    img = tissue_data.tissue.numpy() if isinstance(tissue_data.tissue, torch.Tensor) else tissue_data.tissue
    tissue_mask = tissue_mask_data.mask.numpy() if isinstance(tissue_mask_data.mask, torch.Tensor) else tissue_mask_data.mask
    
    # Get measured mask
    measured_mask = tissue_data.measured_mask
    
    return img, tissue_mask, measured_mask


def apply_gaussian_blur(img):
    """
    Apply Gaussian blur to image. Replace with your CustomGaussianBlur if available.
    """
    try:
        gaussian_blur = CustomGaussianBlur(3, 1.0)
        return gaussian_blur(img[None, ...])[0]
    except ImportError:
        # Fallback to scipy
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=1.0)


def calculate_image_level_quantiles(
    dataset,
    tissue_ids: ArrayLike,
    channel_names: list,
    upper_quantile: float = 0.99,
    lower_quantile: Optional[float] = None,
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Calculate image-level quantiles for each tissue and channel.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: ArrayLike of tissue IDs
        channel_names: list of channel names
        lower_quantile: lower quantile to calculate (default None)
        upper_quantile: upper quantile to calculate (default 0.99)

    Returns:
        pd.DataFrame with shape (n_tissues, n_channels)
    """
    logger.info(f"Calculating image-level {upper_quantile} quantiles and lower quantiles {lower_quantile} if specified")
    
    upper_quantiles_dict = {}
    if lower_quantile is not None:
        lower_quantiles_dict = {}
    
    for tissue_id in tqdm(tissue_ids, desc="Image-level quantiles"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id)
            
            upper_quantiles_dict[tissue_id] = []
            if lower_quantile is not None:
                lower_quantiles_dict[tissue_id] = []
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in tqdm(channel_names, desc='Looping channels', leave=False):
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]

                    # Apply tissue mask 
                    channel_img = channel_img[tissue_mask]
                    
                    # Calculate quantile
                    q_upper = np.quantile(channel_img, upper_quantile)
                    upper_quantiles_dict[tissue_id].append(q_upper)
                    if lower_quantile is not None:
                        q_lower = np.quantile(channel_img, lower_quantile)
                        lower_quantiles_dict[tissue_id].append(q_lower)
                else:
                    q = np.nan            
                    upper_quantiles_dict[tissue_id].append(q)
                    if lower_quantile is not None:
                        lower_quantiles_dict[tissue_id].append(q) 
                
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            upper_quantiles_dict[tissue_id] = [np.nan] * len(channel_names)
    
    df_upper = pd.DataFrame(upper_quantiles_dict).T
    df_upper.columns = channel_names

    if lower_quantile is not None:
        df_lower = pd.DataFrame(lower_quantiles_dict).T
        df_lower.columns = channel_names
    
    return {
        "upper": df_upper,
        "lower": df_lower if lower_quantile is not None else None
    }


def calculate_global_level_quantiles(
    dataset,
    tissue_ids: ArrayLike,
    channel_names: list,
    lower_quantile: Optional[float] = None,
    upper_quantile: float = 0.99,
) -> dict:
    """
    Calculate global quantiles across all tissues for each channel.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: ArrayLike of tissue IDs
        channel_names: list of channel names
        quantile: quantile to calculate (default 0.99)
        method_name: normalization method name
        
    Returns:
        dict mapping channel_name -> quantile value
    """
    logger.info(f"Calculating global quantiles with upper quantile {upper_quantile} and lower quantile {lower_quantile} if specified")
    
    # Accumulate all pixel values per channel
    channel_pixels = {ch: [] for ch in channel_names}
    
    for tissue_id in tqdm(tissue_ids, desc="Collecting pixels for global quantiles"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id)
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    channel_img = channel_img[tissue_mask]  
                    
                    channel_pixels[channel_name].append(channel_img.flatten())
                    
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            continue
    
    # Calculate global quantile for each channel
    logger.info("Computing quantiles from collected pixels")
    global_upper_quantiles = {}
    if lower_quantile is not None:
        global_lower_quantiles = {}
    for channel_name in tqdm(channel_names, desc="Computing global quantiles"):
        if len(channel_pixels[channel_name]) > 0:
            all_pixels = np.concatenate(channel_pixels[channel_name])
            global_upper_quantiles[channel_name] = np.quantile(all_pixels, upper_quantile)
            if lower_quantile is not None:
                global_lower_quantiles[channel_name] = np.quantile(all_pixels, lower_quantile)
        else:
            global_upper_quantiles[channel_name] = np.nan
            if lower_quantile is not None:
                global_lower_quantiles[channel_name] = np.nan

    return {
        "upper": global_upper_quantiles,
        "lower": global_lower_quantiles if lower_quantile is not None else None
    }

def process_image_for_statistics(
    img: np.ndarray,
    tissue_mask: np.ndarray,
    method_name: str,
    upper_quantile: float,
    lower_quantile: Optional[float] = None,
) -> tuple:
    """
    Process a single channel image and return statistics.
    
    Args:
        img: np.ndarray, single channel image (H, W)
        tissue_mask: np.ndarray, boolean mask (H, W)
        quantile: float, quantile value for normalization
        method_name: str, processing method
        
    Returns:
        tuple: (sum, sum_of_squares, count) for incremental statistics
    """
    if lower_quantile is None:
        lower_quantile = 0.0
    if method_name == 'quantile_clipping':
        # Clip at quantile
        clipped_img = np.clip(img, lower_quantile, upper_quantile)
        clipped_img = (clipped_img - lower_quantile) / (upper_quantile - lower_quantile + 1e-8) # rescale to [0, 1]
        
        masked_pixels = clipped_img[tissue_mask]
        
        return np.sum(masked_pixels), np.sum(masked_pixels**2), len(masked_pixels)
    
    elif method_name == 'quantile_clipping_log1p':
        # first clip at quantile, then log1p, then mean-std of foreground pixels, then standardize
        clipped_img = np.clip(img, lower_quantile, upper_quantile)
        clipped_img = np.log1p(clipped_img)
        # get fg pixels using tissue mask
        fg_pixels = clipped_img[tissue_mask]
        return np.sum(fg_pixels), np.sum(fg_pixels**2), len(fg_pixels)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


def calculate_image_level_statistics(
    dataset,
    tissue_ids: ArrayLike,
    channel_names: list,
    method_name: str,
    quantile_level: str,
    upper_quantiles,  # Can be pd.DataFrame (image-level) or dict (global-level)
    lower_quantiles = None,  # Can be pd.DataFrame (image-level) or dict (global-level)
) -> tuple:
    """
    Calculate image-level means and stds for each tissue.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: ArrayLike of tissue IDs
        channel_names: list of channel names
        method_name: normalization method name
        quantile_level: 'image' or 'global'
        upper_quantiles: pd.DataFrame | dict,  # Can be pd.DataFrame (image-level) or dict (global-level)
        lower_quantiles: pd.DataFrame | dict = None  # Can be pd.DataFrame (image-level) or dict (global-level)
    Returns:
        tuple: (means_df, stds_df) - DataFrames with shape (n_tissues, n_channels)
    """
    logger.info("Calculating image-level statistics")
    
    means_dict = {}
    stds_dict = {}
    
    for tissue_id in tqdm(tissue_ids, desc="Image-level statistics"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id)
            
            means_dict[tissue_id] = []
            stds_dict[tissue_id] = []
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    # Get quantile based on level       
                    
                    if quantile_level == 'image': 
                        upper_q = upper_quantiles.loc[tissue_id, channel_name]
                        if lower_quantiles is not None:
                            lower_q = lower_quantiles.loc[tissue_id, channel_name]
                        else:
                            lower_q = None  
                    else:  # global
                        upper_q = upper_quantiles[channel_name]
                        if lower_quantiles is not None:
                            lower_q = lower_quantiles[channel_name]
                        else:
                            lower_q = None
                    
                    if np.isnan(upper_q):
                        means_dict[tissue_id].append(np.nan)
                        stds_dict[tissue_id].append(np.nan)
                        continue
                    
                    # Process image and calculate statistics
                    _sum, _sum_sq, _count = process_image_for_statistics(
                        channel_img, tissue_mask, method_name, upper_q, lower_q
                    )
                    
                    if _count > 0:
                        mean = _sum / _count
                        std = np.sqrt(_sum_sq / _count - mean**2)
                    else:
                        mean = np.nan
                        std = np.nan
                    
                    means_dict[tissue_id].append(mean)
                    stds_dict[tissue_id].append(std)
                else:
                    means_dict[tissue_id].append(np.nan)
                    stds_dict[tissue_id].append(np.nan)
                    
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            means_dict[tissue_id] = [np.nan] * len(channel_names)
            stds_dict[tissue_id] = [np.nan] * len(channel_names)
    
    means_df = pd.DataFrame(means_dict).T
    means_df.columns = channel_names
    
    stds_df = pd.DataFrame(stds_dict).T
    stds_df.columns = channel_names
    
    return means_df, stds_df


def calculate_global_statistics(
    dataset,
    tissue_ids: ArrayLike,
    channel_names: ArrayLike,
    method_name: str,
    quantile_level: str,
    upper_quantiles,  # Can be pd.DataFrame (image-level) or dict (global-level)
    lower_quantiles=None,  # Can be pd.DataFrame (image-level) or dict (global-level)
) -> tuple:
    """
    Calculate global mean and std using quantiles.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: ArrayLike of tissue IDs
        channel_names: list of channel names
        quantiles: pd.DataFrame (image-level) or dict (global-level) with quantiles
        method_name: normalization method name
        quantile_level: 'image' or 'global'
        
    Returns:
        tuple: (means_dict, stds_dict)
    """
    logger.info("Calculating global-level statistics")
    # Accumulator dictionaries
    channel_sum = {ch: 0.0 for ch in channel_names}
    channel_sum_sq = {ch: 0.0 for ch in channel_names}
    channel_count = {ch: 0 for ch in channel_names}
    
    for tissue_id in tqdm(tissue_ids, desc="Global statistics"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id)
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    # Get quantile based on level
                    if quantile_level == 'image':
                        upper_q = upper_quantiles.loc[tissue_id, channel_name]
                        lower_q = lower_quantiles.loc[tissue_id, channel_name] if lower_quantiles is not None else None
                    else:  # global
                        upper_q = upper_quantiles[channel_name]
                        lower_q = lower_quantiles[channel_name] if lower_quantiles is not None else None
                    
                    if np.isnan(upper_q):
                        continue
                    
                    # Process image and accumulate statistics
                    _sum, _sum_sq, _count = process_image_for_statistics(
                        channel_img, tissue_mask, method_name, upper_q, lower_q
                    )
                    
                    channel_sum[channel_name] += _sum
                    channel_sum_sq[channel_name] += _sum_sq
                    channel_count[channel_name] += _count
                    
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            continue
    
    # Calculate final statistics
    means_dict = {}
    stds_dict = {}
    
    for channel_name in channel_names:
        count = channel_count[channel_name]
        if count > 0:
            mean = channel_sum[channel_name] / count
            std = np.sqrt(channel_sum_sq[channel_name] / count - mean**2)
            means_dict[channel_name] = mean
            stds_dict[channel_name] = std
        else:
            means_dict[channel_name] = np.nan
            stds_dict[channel_name] = np.nan
    
    return means_dict, stds_dict


def save_statistics(
    output_dir: Path,
    means,  # Can be pd.DataFrame (image) or dict (global)
    stds,  # Can be pd.DataFrame (image) or dict (global)
    quantile_level: str,
    stats_level: str,
    upper_quantiles,  # Can be pd.DataFrame (image) or dict (global)
    lower_quantiles = None,  # Can be pd.DataFrame (image) or dict (global)
):
    """
    Save statistics to CSV files.
    
    Args:
        output_dir: Path to output directory
        upper_quantiles: pd.DataFrame or dict with upper quantiles
        lower_quantiles: pd.DataFrame or dict with lower quantiles
        means: pd.DataFrame or dict with means
        stds: pd.DataFrame or dict with stds
        quantile_level: 'image' or 'global'
        stats_level: 'image' or 'global'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save quantiles
    if quantile_level == 'image':
        logger.info(f"Saving image-level quantiles to {output_dir}")
        upper_quantiles.to_parquet(output_dir / 'image_level_upper_quantiles.parquet')
        if lower_quantiles is not None:
            lower_quantiles.to_parquet(output_dir / 'image_level_lower_quantiles.parquet')
    else:  # global
        logger.info(f"Saving global-level quantiles to {output_dir}")
        df_up = pd.DataFrame([upper_quantiles])
        df_up.to_parquet(output_dir / 'global_level_upper_quantiles.parquet', index=False)
        if lower_quantiles is not None:
            df_low = pd.DataFrame([lower_quantiles])
            df_low.to_parquet(output_dir / 'global_level_lower_quantiles.parquet', index=False)
    
    # Save means and stds
    if stats_level == 'image':
        logger.info(f"Saving image-level statistics to {output_dir}")
        means.to_parquet(output_dir / 'image_level_means.parquet')
        stds.to_parquet(output_dir / 'image_level_stds.parquet')
    else:  # global
        logger.info(f"Saving global-level statistics to {output_dir}")
        df_means = pd.DataFrame([means])
        df_stds = pd.DataFrame([stds])
        df_means.to_parquet(output_dir / 'global_level_means.parquet', index=False)
        df_stds.to_parquet(output_dir / 'global_level_stds.parquet', index=False)