"""Compute tile coordinates from shared tissue masks in the new dataset format."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatialprot_data._config import get_datasets_dir
from spatialprot_data.utils.helpers.crop import best_mask_tiling_try_to_stop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tile coordinates from segmentations/<resolution>/tissue_masks and save them as parquet.",
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name under the datasets root.")
    parser.add_argument("--crop-size", required=True, type=int, help="Square crop size in pixels.")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution in mpp used to read tissue masks and write tiling outputs.",
    )
    parser.add_argument(
        "--tiling-method",
        default="default",
        help="Subdirectory name under tiling/<resolution>/ used for the saved outputs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in pixels. Defaults to crop_size // 2.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.85,
        help="Maximum invalid-pixel fraction allowed within a tile.",
    )
    parser.add_argument(
        "--coverage-goal",
        type=float,
        default=1.0,
        help="Coverage goal passed to best_mask_tiling_try_to_stop.",
    )
    parser.add_argument(
        "--min-gain-ratio",
        type=float,
        default=0.05,
        help="Minimum marginal gain ratio used for adaptive stopping.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet outputs if they already exist.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable per-tissue progress bars inside the tiling helper.",
    )
    return parser.parse_args()


def resolution_to_dir(resolution: float | str) -> str:
    return f"{str(resolution).replace('.', '_')}mpp"


def load_tissue_mask(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path)["mask"]
    return mask.astype(bool, copy=False)


def build_coordinate_rows(tissue_id: str, tiles: list) -> list[dict[str, int | str]]:
    return [
        {
            "tissue_id": tissue_id,
            "crop_id": crop_id,
            "row": int(tile.y),
            "col": int(tile.x),
        }
        for crop_id, tile in enumerate(tiles)
    ]


def build_stats_row(
    tissue_id: str,
    mask: np.ndarray,
    stats: dict,
    crop_size: int,
    stride: int,
    tolerance: float,
    coverage_goal: float,
    min_gain_ratio: float,
) -> dict[str, int | float | str]:
    return {
        "tissue_id": tissue_id,
        "mask_height": int(mask.shape[0]),
        "mask_width": int(mask.shape[1]),
        "tissue_mask_frac": float(mask.mean()),
        "num_tiles": int(stats["num_tiles"]),
        "coverage_ratio": float(stats["coverage_ratio"]),
        "covered_valid_pixels": int(stats["covered_valid_pixels"]),
        "total_valid_pixels": int(stats["total_valid_pixels"]),
        "candidate_count": int(stats["candidate_count"]),
        "stop_reason": str(stats["stop_reason"]),
        "crop_size": int(crop_size),
        "stride": int(stride),
        "tolerance": float(tolerance),
        "coverage_goal": float(coverage_goal),
        "min_gain_ratio": float(min_gain_ratio),
    }


def main():
    args = parse_args()
    dataset_root = get_datasets_dir() / args.dataset_name
    resolution_dir = resolution_to_dir(args.resolution)
    tissue_masks_dir = dataset_root / "segmentations" / resolution_dir / "tissue_masks"
    tiling_dir = dataset_root / "tiling" / resolution_dir / args.tiling_method
    coords_path = tiling_dir / f"{args.crop_size}_tile_coordinates.parquet"
    stats_path = tiling_dir / f"{args.crop_size}_tile_stats.parquet"
    stride = args.stride if args.stride is not None else args.crop_size // 2

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not tissue_masks_dir.exists():
        raise FileNotFoundError(f"Tissue masks directory does not exist: {tissue_masks_dir}")

    if coords_path.exists() and stats_path.exists() and not args.overwrite:
        print(f"Tiling outputs already exist. Skipping: {coords_path} and {stats_path}")
        return

    mask_paths = sorted(path for path in tissue_masks_dir.iterdir() if path.suffix == ".npz")
    if not mask_paths:
        raise FileNotFoundError(f"No tissue mask files found in {tissue_masks_dir}")

    tiling_dir.mkdir(parents=True, exist_ok=True)

    coordinate_rows: list[dict[str, int | str]] = []
    stats_rows: list[dict[str, int | float | str]] = []

    iterator = tqdm(mask_paths, desc=f"Tiling {args.dataset_name}", unit="mask")
    for mask_path in iterator:
        tissue_id = mask_path.stem
        mask = load_tissue_mask(mask_path)
        tiles, stats, _ = best_mask_tiling_try_to_stop(
            mask=mask,
            tile_size=args.crop_size,
            stride=stride,
            tolerance=args.tolerance,
            coverage_goal=args.coverage_goal,
            min_gain_ratio=args.min_gain_ratio,
            allow_overlap=True,
            progress=not args.disable_progress,
            progress_desc=f"Tiling tissue {tissue_id}",
        )
        coordinate_rows.extend(build_coordinate_rows(tissue_id, tiles))
        stats_rows.append(
            build_stats_row(
                tissue_id=tissue_id,
                mask=mask,
                stats=stats,
                crop_size=args.crop_size,
                stride=stride,
                tolerance=args.tolerance,
                coverage_goal=args.coverage_goal,
                min_gain_ratio=args.min_gain_ratio,
            )
        )

    coordinates_df = pd.DataFrame.from_records(
        coordinate_rows,
        columns=["tissue_id", "crop_id", "row", "col"],
    )
    stats_df = pd.DataFrame.from_records(stats_rows).set_index("tissue_id").sort_index()

    coordinates_df.to_parquet(coords_path, index=False)
    stats_df.to_parquet(stats_path)

    print(f"Saved tile coordinates to {coords_path}")
    print(f"Saved tiling stats to {stats_path}")
    if not stats_df.empty:
        print(
            "Summary: "
            f"{len(stats_df)} tissues | "
            f"{int(stats_df['num_tiles'].sum())} tiles | "
            f"coverage {stats_df['coverage_ratio'].mean():.4f} +/- {stats_df['coverage_ratio'].std(ddof=0):.4f}"
        )


if __name__ == '__main__':
    main()
