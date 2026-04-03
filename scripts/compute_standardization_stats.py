HELPER_TEXT = """
Compute standardization statistics (quantiles, means, stds) for a dataset.
Currently supports multiplex datasets and broadly two methods:
(1) quantile_clipping: Clips images at a specified upper quantile (e.g. 0.99) and optionally a lower quantile (e.g. 0.01), then rescales to [0,1] range.
(2) quantile_clipping_log1p: Similar as quantile_clipping, but applies log1p transformation after clipping without scaling.

Example:
    python -m scripts.compute_standardization_stats \
        --dataset-name dataset \
        --method quantile_clipping \
        --quantile-level image \
        --stats-level global \
        --resolution 1.0
"""

import argparse
import os

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

VALID_METHODS = ["quantile_clipping", "quantile_clipping_log1p"]
VALID_LEVELS = ["image", "global"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=HELPER_TEXT,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name under the datasets root.",
    )
    parser.add_argument(
        "--modality", 
        required=True,
        choices=["codex", "cycif", "imc"],
        help="Modality to process. Required to disambiguate paths and modality-specific processing.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=VALID_METHODS,
        help="Standardization statistics recipe to compute.",
    )
    parser.add_argument(
        "--quantile-level",
        required=True,
        choices=VALID_LEVELS,
        help="Whether to compute quantiles at image or global level.",
    )
    parser.add_argument(
        "--stats-level",
        required=True,
        choices=VALID_LEVELS,
        help="Whether to compute means/stds at image or global level.",
    )
    parser.add_argument(
        "--lower-quantile",
        "-lq",
        type=float,
        default=None,
        help="Lower quantile level to compute (e.g. 0.01). Only used for quantile clipping method. Defaults to None, which means only the upper quantile (e.g. 0.99) will be computed.",
    )
    parser.add_argument(
        "--upper-quantile",
        "-uq",
        type=float,
        default=0.99,
        help="Upper quantile level to compute (e.g. 0.99). Only used for quantile clipping method. Defaults to 0.99.",  
    )
    parser.add_argument(
        "--resolution",
        required=True,
        type=float,
        help="Resolution in mpp.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = get_datasets_dir()
    dataset_name = args.dataset_name
    base_method = args.method
    quantile_level = args.quantile_level
    stats_level = args.stats_level
    resolution = args.resolution
    upper_quantile = args.upper_quantile
    lower_quantile = args.lower_quantile
    modality = args.modality

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Modality: {modality}")
    logger.info(f"Base Method: {base_method}")
    logger.info(f"Quantile level: {quantile_level}")
    logger.info(f"Stats level: {stats_level}")
    logger.info(f"Resolution: {resolution}")
    logger.info(f"Upper quantile: {upper_quantile}")
    logger.info(f"Lower quantile: {lower_quantile}")



    logger.info("Initializing dataset...")
    dataset = MultiplexImagingDataset(
        name=dataset_name,
        path=str(dataset_dir / dataset_name),
        resolution=resolution,
        crop_size=None,
        load_cell_metadata=False,
        modality=args.modality,
        standardization="identity",
    )

    method_name = f"{base_method}/"


    method_name += f"uq_{upper_quantile}"
    if lower_quantile is not None:
        method_name += f"_lq_{lower_quantile}"
        

    logger.info(f"Full method name for saving stats: {method_name}")


    method_to_path = {
        "quantile_clipping": f"standardization/quantile_clipping/{method_name}",
        "quantile_clipping_log1p": f"standardization/quantile_clipping_log1p/{method_name}",
    }

    quantile_path = method_to_path[base_method]

    dataset_path = dataset_dir / dataset_name / dataset.modality.name / dataset.resolution

    logger.info(f"Modality: {dataset.modality.name}")

    output_dir = dataset_path / quantile_path
    logger.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    check_files = []
    if quantile_level == "image":
        check_files.append(output_dir / "image_level_upper_quantiles.parquet")
        if lower_quantile is not None:
            check_files.append(output_dir / "image_level_lower_quantiles.parquet")
    else:
        check_files.append(output_dir / "global_level_upper_quantiles.parquet")
        if lower_quantile is not None:
            check_files.append(output_dir / "global_level_lower_quantiles.parquet")

    if stats_level == "image":
        check_files.append(output_dir / "image_level_means.parquet")
        check_files.append(output_dir / "image_level_stds.parquet")
    else:
        check_files.append(output_dir / "global_level_means.parquet")
        check_files.append(output_dir / "global_level_stds.parquet")

    if all(f.exists() for f in check_files):
        logger.info(f"Statistics already computed for {dataset_name} with method {base_method} with specific name {method_name} at resolution {resolution} for modality {modality}. Skipping computation.")
        logger.info(f"Quantile level: {quantile_level}, Stats level: {stats_level}")
        return

    channel_names = dataset.channel_list["channel_name"].values
    logger.info(f"Found {len(channel_names)} channels")

    tissue_ids = dataset.get_tissue_ids()
    logger.info(f"Found {len(tissue_ids)} tissues")

    logger.info(f"Calculating {quantile_level}-level quantiles...")
    if quantile_level == "image":
        quantiles = calculate_image_level_quantiles(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )
    else:
        quantiles = calculate_global_level_quantiles(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )

    logger.info(f"Calculating {stats_level}-level statistics...")
    if stats_level == "image":
        means, stds = calculate_image_level_statistics(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            method_name=base_method,
            quantile_level=quantile_level,
            upper_quantiles=quantiles["upper"],
            lower_quantiles=quantiles["lower"],
        )
    else:
        means, stds = calculate_global_statistics(
            dataset=dataset,
            tissue_ids=tissue_ids,
            channel_names=channel_names,
            method_name=base_method,
            quantile_level=quantile_level,
            upper_quantiles=quantiles["upper"],
            lower_quantiles=quantiles["lower"],
        )

    save_statistics(
        output_dir=output_dir,
        means=means,
        stds=stds,
        quantile_level=quantile_level,
        stats_level=stats_level,
        upper_quantiles=quantiles["upper"],
        lower_quantiles=quantiles["lower"],
    )

    logger.info("Done!")

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    if quantile_level == "image":
        logger.info(f"\nImage-level quantiles shape: {quantiles['upper'].shape}")
        logger.info("Sample (first 5 rows, 5 columns):")
        logger.info(f"\n{quantiles['upper'].iloc[:5, :5]}")
    else:
        logger.info(f"\nGlobal-level quantiles: {len(quantiles)} channels")
        logger.info("Sample (first 5 channels):")
        logger.info(f"{dict(list(quantiles.items())[:5])}")

    if stats_level == "image":
        logger.info(f"\nImage-level means shape: {means.shape}")
        logger.info(f"Image-level stds shape: {stds.shape}")
        logger.info("Sample means (first 5 rows, 5 columns):")
        logger.info(f"\n{means.iloc[:5, :5]}")
    else:
        logger.info(f"\nGlobal-level statistics: {len(means)} channels")
        logger.info("Sample means (first 5 channels):")
        logger.info(f"{dict(list(means.items())[:5])}")
        logger.info("Sample stds (first 5 channels):")
        logger.info(f"{dict(list(stds.items())[:5])}")


if __name__ == '__main__':
    main()
