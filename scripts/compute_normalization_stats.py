"""Compute normalization statistics (quantiles, means, stds) for a dataset.

Usage:
    python scripts/compute_normalization_stats.py <dataset_name> <method> <quantile_level> <stats_level> <resolution>

Methods: tm_q99_clipping, tm_z_standardization
Quantile levels: image, global
Stats levels: image, global

Examples:
    # Image-level quantiles, global-level stats
    python scripts/compute_normalization_stats.py dataset tm_q99_clipping image global 1.0

    # Both at image-level
    python scripts/compute_normalization_stats.py dataset tm_q99_clipping image image 1.0
"""

import sys

from loguru import logger

from spatialprot_data._config import get_datasets_dir
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset
from spatialprot_data.utils.helpers.std import (
    calculate_global_level_quantiles,
    calculate_global_statistics,
    calculate_image_level_quantiles,
    calculate_image_level_statistics,
    save_statistics,
)


def main():
    if len(sys.argv) < 6:
        logger.error("Usage: python compute_normalization_stats.py <dataset_name> <method> <quantile_level> <stats_level> <resolution>")
        logger.error("Methods: tm_q99_clipping, tm_z_standardization")
        logger.error("Quantile levels: image, global")
        logger.error("Stats levels: image, global")
        sys.exit(1)

    dataset_dir = get_datasets_dir()
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
        path=str(dataset_dir / dataset_name),
        resolution=float(resolution),
        crop_size=224,
        load_cell_metadata=True,
        modality="codex",
        normalization="identity"
    )

    # Determine paths based on method
    method_to_path = {
        'tm_q99_clipping': 'normalization/q99_clipping',
        'tm_z_standardization': 'normalization/z_standardization',
    }

    if method not in method_to_path:
        logger.error(f"Unknown method: {method}")
        logger.error(f"Valid methods: {list(method_to_path.keys())}")
        sys.exit(1)

    quantile_path = method_to_path[method]

    # Determine modality
    dataset_path = dataset_dir / dataset_name / dataset.modality.name / dataset.resolution

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
    else:
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
    else:
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
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

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
