import numpy as np
import os
from tqdm import tqdm
from loguru import logger
import pandas as pd
import sys
from pathlib import Path
import zarr
import torch
from spatialprot_data.utils.dataset.transforms import CustomGaussianBlur, custom_median_filter
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset

DATASET_DIR = Path('/mnt/aimm/scratch/datasets_v2/')


def load_tissue_and_mask(dataset, tissue_id: str, method_name: str):
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
    tissue_ids: list,
    channel_names: list,
    quantile: float = 0.99,
    method_name: str = 'clip99'
) -> pd.DataFrame:
    """
    Calculate image-level quantiles for each tissue and channel.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: list of tissue IDs
        channel_names: list of channel names
        quantile: quantile to calculate (default 0.99)
        method_name: normalization method name
        
    Returns:
        pd.DataFrame with shape (n_tissues, n_channels)
    """
    logger.info(f"Calculating image-level {quantile} quantiles")
    
    quantiles_dict = {}
    
    for tissue_id in tqdm(tissue_ids, desc="Image-level quantiles"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id, method_name)
            
            quantiles_dict[tissue_id] = []
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    # Apply tissue mask if using masked methods
                    if method_name.startswith('tm_'):
                        channel_img = channel_img[tissue_mask]
                    
                    # Calculate quantile
                    q = np.quantile(channel_img, quantile)
                else:
                    q = np.nan
                
                quantiles_dict[tissue_id].append(q)
                
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            quantiles_dict[tissue_id] = [np.nan] * len(channel_names)
    
    df = pd.DataFrame(quantiles_dict).T
    df.columns = channel_names
    
    return df


def calculate_global_level_quantiles(
    dataset,
    tissue_ids: list,
    channel_names: list,
    quantile: float = 0.99,
    method_name: str = 'clip99'
) -> dict:
    """
    Calculate global quantiles across all tissues for each channel.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: list of tissue IDs
        channel_names: list of channel names
        quantile: quantile to calculate (default 0.99)
        method_name: normalization method name
        
    Returns:
        dict mapping channel_name -> quantile value
    """
    logger.info(f"Calculating global {quantile} quantiles")
    
    # Accumulate all pixel values per channel
    channel_pixels = {ch: [] for ch in channel_names}
    
    for tissue_id in tqdm(tissue_ids, desc="Collecting pixels for global quantiles"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id, method_name)
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    # Apply tissue mask if using masked methods
                    if method_name.startswith('tm_'):
                        channel_img = channel_img[tissue_mask]
                    
                    channel_pixels[channel_name].append(channel_img.flatten())
                    
        except Exception as e:
            logger.error(f"Error processing {tissue_id}: {e}")
            continue
    
    # Calculate global quantile for each channel
    logger.info("Computing quantiles from collected pixels")
    global_quantiles = {}
    for channel_name in tqdm(channel_names, desc="Computing global quantiles"):
        if len(channel_pixels[channel_name]) > 0:
            all_pixels = np.concatenate(channel_pixels[channel_name])
            global_quantiles[channel_name] = np.quantile(all_pixels, quantile)
        else:
            global_quantiles[channel_name] = np.nan
    
    return global_quantiles


def process_image_for_statistics(
    img: np.ndarray,
    tissue_mask: np.ndarray,
    quantile: float,
    method_name: str
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
    
    if method_name == 'q99_clipping':
        # Clip at quantile
        clipped_img = np.clip(img, 0, quantile)
        # clipped_img = np.log1p(clipped_img)
        # blurred_img = apply_gaussian_blur(clipped_img)
        clipped_img = clipped_img / quantile # [0, 1]
        
        masked_pixels = clipped_img[tissue_mask]
        
        return np.sum(masked_pixels), np.sum(masked_pixels**2), len(masked_pixels)
    
    elif method_name == 'z_standardization':
        # first clip at quantile, then log1p, then mean-std of foreground pixels, then standardize
        clipped_img = np.clip(img, 0, quantile)
        clipped_img = np.log1p(clipped_img)
        # get fg pixels using tissue mask
        fg_pixels = clipped_img[tissue_mask]
        mean = np.mean(fg_pixels)
        std = np.std(fg_pixels)
        if std > 0:
            standardized_img = (clipped_img - mean) / std
        else:
            standardized_img = clipped_img - mean
        masked_pixels = standardized_img[tissue_mask]
        return np.sum(masked_pixels), np.sum(masked_pixels**2), len(masked_pixels)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


def calculate_image_level_statistics(
    dataset,
    tissue_ids: list,
    channel_names: list,
    quantiles,  # Can be pd.DataFrame (image-level) or dict (global-level)
    method_name: str,
    quantile_level: str
) -> tuple:
    """
    Calculate image-level means and stds for each tissue.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: list of tissue IDs
        channel_names: list of channel names
        quantiles: pd.DataFrame (image-level) or dict (global-level) with quantiles
        method_name: normalization method name
        quantile_level: 'image' or 'global'
        
    Returns:
        tuple: (means_df, stds_df) - DataFrames with shape (n_tissues, n_channels)
    """
    logger.info("Calculating image-level statistics")
    
    means_dict = {}
    stds_dict = {}
    
    for tissue_id in tqdm(tissue_ids, desc="Image-level statistics"):
        try:
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id, method_name)
            
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
                        q = quantiles.loc[tissue_id, channel_name]
                    else:  # global
                        q = quantiles[channel_name]
                    
                    if np.isnan(q):
                        means_dict[tissue_id].append(np.nan)
                        stds_dict[tissue_id].append(np.nan)
                        continue
                    
                    # Process image and calculate statistics
                    _sum, _sum_sq, _count = process_image_for_statistics(
                        channel_img, tissue_mask, q, method_name
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
    tissue_ids: list,
    channel_names: list,
    quantiles,  # Can be pd.DataFrame (image-level) or dict (global-level)
    method_name: str,
    quantile_level: str
) -> tuple:
    """
    Calculate global mean and std using quantiles.
    
    Args:
        dataset: MultiplexImagingDataset instance
        tissue_ids: list of tissue IDs
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
            img, tissue_mask, measured_mask = load_tissue_and_mask(dataset, tissue_id, method_name)
            
            # Get channel names for this tissue
            tissue_channel_names = dataset.get_channel_names(tissue_id, kind="complete")
            
            for channel_name in channel_names:
                if channel_name in tissue_channel_names:
                    # Find index in the loaded image
                    channel_idx = np.where(tissue_channel_names == channel_name)[0][0]
                    channel_img = img[channel_idx]
                    
                    # Get quantile based on level
                    if quantile_level == 'image':
                        q = quantiles.loc[tissue_id, channel_name]
                    else:  # global
                        q = quantiles[channel_name]
                    
                    if np.isnan(q):
                        continue
                    
                    # Process image and accumulate statistics
                    _sum, _sum_sq, _count = process_image_for_statistics(
                        channel_img, tissue_mask, q, method_name
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
    quantiles,  # Can be pd.DataFrame (image) or dict (global)
    means,  # Can be pd.DataFrame (image) or dict (global)
    stds,  # Can be pd.DataFrame (image) or dict (global)
    quantile_level: str,
    stats_level: str
):
    """
    Save statistics to CSV files.
    
    Args:
        output_dir: Path to output directory
        quantiles: pd.DataFrame or dict with quantiles
        means: pd.DataFrame or dict with means
        stds: pd.DataFrame or dict with stds
        quantile_level: 'image' or 'global'
        stats_level: 'image' or 'global'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save quantiles
    if quantile_level == 'image':
        logger.info(f"Saving image-level quantiles to {output_dir}")
        quantiles.to_parquet(output_dir / 'image_level_quantiles.parquet')
    else:  # global
        logger.info(f"Saving global-level quantiles to {output_dir}")
        df_q = pd.DataFrame([quantiles])
        df_q.to_parquet(output_dir / 'global_level_quantiles.parquet', index=False)
    
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


def main():
    """
    Main function to calculate normalization statistics.
    
    Usage:
        python calculate_normalization_statistics.py <dataset_name> <method> <quantile_level> <stats_level> <resolution>
        
    Args:
        dataset_name: str, name of the dataset
        method: str, one of ['q99_clipping', 'z_standardization']
        quantile_level: str, 'image' or 'global' 
        stats_level: str, 'image' or 'global'
    """
    
    # Parse arguments
    if len(sys.argv) < 6:
        logger.error("Usage: python std.py <dataset_name> <method> <quantile_level> <stats_level> <resolution>")
        logger.error("Methods: q99_clipping, z_standardization")
        logger.error("Quantile levels: image, global")
        logger.error("Stats levels: image, global")
        logger.error("")
        logger.error("Examples:")
        logger.error("  # Image-level quantiles, global-level stats")
        logger.error("  python std.py dataset q99_clipping image global")
        logger.error("")
        logger.error("  # Both at image-level")
        logger.error("  python std.py dataset q99_clipping image image")
        logger.error("")
        logger.error("  # Both at global-level")
        logger.error("  python std.py dataset q99_clipping global global")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    method = sys.argv[2]
    quantile_level = sys.argv[3]
    stats_level = sys.argv[4]
    resolution = sys.argv[5]
    
    if quantile_level not in ['image', 'global']:
        logger.error(f"Invalid quantile_level: {quantile_level}. Must be 'image' or 'global'")
        sys.exit(1)
    
    if stats_level not in ['image', 'global']:
        logger.error(f"Invalid stats_level: {stats_level}. Must be 'image' or 'global'")
        sys.exit(1)
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Quantile level: {quantile_level}")
    logger.info(f"Stats level: {stats_level}")
    
    
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = MultiplexImagingDataset(
        name=dataset_name,
        path='/mnt/aimm/scratch/datasets_v2/' + dataset_name,
        resolution=float(resolution),
        crop_size=224,
        load_cell_metadata=True,
        modality="codex",
        normalization="identity"
    )
        
    # Determine paths based on method
    method_to_path = {
        # 'log_compress_fg_0.99': 'normalization/log_compress_fg_0.99',
        'tm_q99_clipping': 'normalization/q99_clipping',
        'tm_z_standardization': 'normalization/z_standardization',
        # 'tm_rescale255_clip99': 'normalization/tm_rescale255_clip99'
    }


    
    if method not in method_to_path:
        logger.error(f"Unknown method: {method}")
        logger.error(f"Valid methods: {list(method_to_path.keys())}")
        sys.exit(1)
    
    quantile_path = method_to_path[method]
    
    # Determine modality
    dataset_path = DATASET_DIR / dataset_name / dataset.modality.name / dataset.resolution
    
    logger.info(f"Modality: {dataset.modality.name}")
    
    output_dir = dataset_path / quantile_path
    logger.info(f"Output directory: {output_dir}") 
    
    # Determine which files to check based on levels
    check_files = []
    if quantile_level == 'image':
        check_files.append(output_dir / 'image_level_quantiles.parquet')
    else:
        check_files.append(output_dir / 'global_level_quantiles.parquet')
    
    if stats_level == 'image':
        check_files.append(output_dir / 'image_level_means.parquet')
        check_files.append(output_dir / 'image_level_stds.parquet')
    else:
        check_files.append(output_dir / 'global_level_means.parquet')
        check_files.append(output_dir / 'global_level_stds.parquet')
    
    if all(f.exists() for f in check_files):
        logger.info(f"Statistics already computed for {dataset_name} with method {method}")
        logger.info(f"Quantile level: {quantile_level}, Stats level: {stats_level}")
        sys.exit(0)
    
    # Get all channel names (QC-filtered, no embedding filter)
    channel_names = dataset.channel_list['channel_name'].values
    logger.info(f"Found {len(channel_names)} channels")
    
    # Get tissue IDs
    tissue_metadata = dataset.tissue_modality_metadata
    tissue_ids = dataset.get_tissue_ids()
    logger.info(f"Found {len(tissue_ids)} tissues")
    
    # Calculate quantiles
    logger.info(f"Calculating {quantile_level}-level quantiles...")
    if quantile_level == 'image':
        quantiles = calculate_image_level_quantiles(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            quantile=0.99,
            method_name=method
        )
    else:  # global
        quantiles = calculate_global_level_quantiles(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            quantile=0.99,
            method_name=method
        )
    
    # Calculate statistics
    logger.info(f"Calculating {stats_level}-level statistics...")
    if stats_level == 'image':
        means, stds = calculate_image_level_statistics(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            quantiles=quantiles,
            method_name=method,
            quantile_level=quantile_level
        )
    else:  # global
        means, stds = calculate_global_statistics(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            quantiles=quantiles,
            method_name=method,
            quantile_level=quantile_level
        )
    
    # Save results
    save_statistics(
        output_dir=output_dir,
        quantiles=quantiles,
        means=means,
        stds=stds,
        quantile_level=quantile_level,
        stats_level=stats_level
    )
    
    logger.info("Done!")
    
    # Display sample results
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    if quantile_level == 'image':
        logger.info(f"\nImage-level quantiles shape: {quantiles.shape}")
        logger.info(f"Sample (first 5 rows, 5 columns):")
        logger.info(f"\n{quantiles.iloc[:5, :5]}")
    else:
        logger.info(f"\nGlobal-level quantiles: {len(quantiles)} channels")
        logger.info(f"Sample (first 5 channels):")
        logger.info(f"{dict(list(quantiles.items())[:5])}")
    
    if stats_level == 'image':
        logger.info(f"\nImage-level means shape: {means.shape}")
        logger.info(f"Image-level stds shape: {stds.shape}")
        logger.info(f"Sample means (first 5 rows, 5 columns):")
        logger.info(f"\n{means.iloc[:5, :5]}")
    else:
        logger.info(f"\nGlobal-level statistics: {len(means)} channels")
        logger.info(f"Sample means (first 5 channels):")
        logger.info(f"{dict(list(means.items())[:5])}")
        logger.info(f"Sample stds (first 5 channels):")
        logger.info(f"{dict(list(stds.items())[:5])}")


if __name__ == '__main__':
    main()