"""Compute optimal tile coordinates for tissue images."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatialprot_data._config import get_datasets_dir
from spatialprot_data.datasets.he import HEImagingDataset
from spatialprot_data.datasets.ihc import SingleIHCImagingDataset
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset
from spatialprot_data.utils.helpers.crop import best_mask_tiling_try_to_stop

VALID_MODALITIES = ["he", "imc", "codex", "cycif", "ihc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute crop coordinates for tissue images.",
    )
    parser.add_argument("dataset_name", help="Dataset name under the datasets root.")
    parser.add_argument("crop_size", type=int, help="Square crop size in pixels.")
    parser.add_argument(
        "modality",
        nargs="?",
        help=(
            "Optional modality to process. Use one of he/imc/codex/cycif, 'ihc' to process all "
            "IHC markers, or a specific marker like 'ihc_CD3'."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution in mpp used for reading masks/images and writing coordinates.",
    )
    return parser.parse_args()


def discover_modalities(dataset_root: Path, requested: str | None) -> list[str]:
    if requested is not None:
        if requested in {"he", "imc", "codex", "cycif"}:
            return [requested]
        if requested == "ihc":
            ihc_root = dataset_root / "ihc"
            return sorted(child.name for child in ihc_root.iterdir() if child.is_dir() and child.name.startswith("ihc_")) if ihc_root.exists() else []
        if requested.startswith("ihc_"):
            return [requested]
        raise ValueError(
            f"Unsupported modality '{requested}'. Use one of {VALID_MODALITIES} or a specific IHC marker like 'ihc_CD3'."
        )

    modalities = [modality for modality in ["he", "imc", "codex", "cycif"] if (dataset_root / modality).is_dir()]
    ihc_root = dataset_root / "ihc"
    if ihc_root.exists():
        modalities.extend(sorted(child.name for child in ihc_root.iterdir() if child.is_dir() and child.name.startswith("ihc_")))
    return modalities


def segmentations_relpath(modality: str) -> Path:
    if modality.startswith("ihc_"):
        return Path("ihc") / modality
    return Path(modality)


def build_dataset(dataset_root: Path, dataset_name: str, modality: str, resolution: float):
    if modality == "he":
        return HEImagingDataset(
            name=dataset_name,
            path=dataset_root,
            verbose=True,
            resolution=resolution,
            crop_size=0,
        )
    if modality in {"imc", "codex", "cycif"}:
        return MultiplexImagingDataset(
            name=dataset_name,
            modality=modality,
            path=dataset_root,
            verbose=True,
            resolution=resolution,
            crop_size=0,
            standardization="identity",
        )
    if modality.startswith("ihc_"):
        return SingleIHCImagingDataset(
            name=dataset_name,
            path=dataset_root,
            marker_name=modality,
            verbose=True,
            resolution=resolution,
            crop_size=0,
        )
    raise ValueError(f"Unsupported modality '{modality}' for dataset '{dataset_name}'.")


def main():
    args = parse_args()
    dataset_dir = get_datasets_dir()
    dataset_name = args.dataset_name
    crop_size = args.crop_size
    modality = args.modality
    resolution = args.resolution
    resolution_dir = f"{str(resolution).replace('.', '_')}mpp"
    dataset_root = dataset_dir / dataset_name

    valid_modalities = discover_modalities(dataset_root, modality)
    if not valid_modalities:
        raise ValueError(
            f"No valid modalities found for dataset '{dataset_name}' in {dataset_dir}."
        )

    print(f"Found valid modalities for dataset '{dataset_name}': {valid_modalities}")

    for modality in valid_modalities:
        seg_relpath = segmentations_relpath(modality)
        tissue_mask_path = dataset_root / "segmentations" / seg_relpath / "tissue_masks" / resolution_dir
        coords_path = dataset_root / "segmentations" / seg_relpath / "crop_coordinates" / resolution_dir / f"{crop_size}_tiles_coordinates.json"

        if coords_path.exists():
            print(
                f"Crop coordinates already exist for modality '{modality}' in "
                f"dataset '{dataset_name}' at {coords_path}. Skipping.",
            )
            continue

        if os.path.exists(tissue_mask_path) and len(os.listdir(tissue_mask_path)) > 0:
            print(f"Processing modality '{modality}' for dataset '{dataset_name}'...")
        else:
            print(
                f"Warning: No tissue masks found for modality '{modality}' in "
                f"dataset '{dataset_name}' at {tissue_mask_path}. Skipping.",
                file=sys.stderr,
            )
            continue

        dataset = build_dataset(dataset_root, dataset_name, modality, resolution)

        tids = dataset.get_tissue_ids()
        tissue_mask_frac = {}
        tissue_mask_coverage = {}
        tissue_mask_num_tiles = {}
        tile_coordinates = {}
        for tid in tqdm(tids, desc=f"Processing tissues for modality '{modality}'", unit="tissue"):
            tissue_mask = dataset.get_tissue_mask(tid)
            tissue_mask_frac[tid] = float(tissue_mask.mask.mean())
            tiles, stats, covered = best_mask_tiling_try_to_stop(
                mask=tissue_mask.mask,
                tile_size=crop_size,
                stride=crop_size // 2,
                tolerance=0.85,
                coverage_goal=1,
                min_gain_ratio=0.05,
                allow_overlap=True,
                progress=True,
                progress_desc=f"Tiling tissue {tid}",
            )
            tissue_mask_coverage[tid] = stats["coverage_ratio"]
            tissue_mask_num_tiles[tid] = stats["num_tiles"]
            for tile in tiles:
                tile_coordinates.setdefault(tid, []).append((tile.y, tile.x))

        print(f"Summary for modality '{modality}' in dataset '{dataset_name}':")
        print(
            f"  Tissue mask fraction (mean +/- std): "
            f"{np.mean(list(tissue_mask_frac.values())):.4f} +/- "
            f"{np.std(list(tissue_mask_frac.values())):.4f}"
        )
        print(
            f"  Tissue mask coverage after tiling (mean +/- std): "
            f"{np.mean(list(tissue_mask_coverage.values())):.4f} +/- "
            f"{np.std(list(tissue_mask_coverage.values())):.4f}"
        )
        print(
            f"  Tissue mask number of tiles (mean +/- std): "
            f"{np.mean(list(tissue_mask_num_tiles.values())):.2f} +/- "
            f"{np.std(list(tissue_mask_num_tiles.values())):.2f}"
        )

        df_path = dataset_root / "metadata" / "crop_tiling" / f"{dataset.resolution}" / "tiling_stats.parquet"
        if df_path.exists():
            df = pd.read_parquet(df_path)
        else:
            os.makedirs(df_path.parent, exist_ok=True)
            df = pd.DataFrame(index=tids)
            df["tissue_mask_frac"] = pd.Series(tissue_mask_frac)
        df[f"{modality}_coverage_{crop_size}"] = pd.Series(tissue_mask_coverage)
        df[f"{modality}_num_tiles_{crop_size}"] = pd.Series(tissue_mask_num_tiles)
        df.to_parquet(df_path)

        coords_dir = dataset_root / "segmentations" / seg_relpath / "crop_coordinates" / f"{dataset.resolution}"
        os.makedirs(coords_dir, exist_ok=True)
        with open(coords_dir / f"{crop_size}_tiles_coordinates.json", "w") as f:
            json.dump(tile_coordinates, f)


if __name__ == '__main__':
    main()
