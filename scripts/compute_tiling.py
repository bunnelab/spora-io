"""Compute optimal tile coordinates for tissue images.

Usage:
    python scripts/compute_tiling.py <dataset_name> <crop_size> [modality]
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatialprot_data._config import get_datasets_dir
from spatialprot_data.datasets.he import HEImagingDataset
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset
from spatialprot_data.utils.helpers.crop import best_mask_tiling_try_to_stop

VALID_MODALITIES = ["he", "imc", "codex", "cycif"]


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <crop_size> <modality (optional)>", file=sys.stderr)
        sys.exit(1)

    dataset_dir = get_datasets_dir()
    dataset_name = sys.argv[1]
    crop_size = int(sys.argv[2])
    modality = sys.argv[3] if len(sys.argv) > 3 else None

    valid_modalities = [m for m in VALID_MODALITIES if (dataset_dir / dataset_name / m).is_dir()] if modality is None else [modality] if modality in VALID_MODALITIES else []
    if not valid_modalities:
        print(f"Error: No valid modalities found for dataset '{dataset_name}' in {dataset_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found valid modalities for dataset '{dataset_name}': {valid_modalities}")

    for modality in valid_modalities:
        tissue_mask_path = dataset_dir / dataset_name / "segmentations" / modality / "tissue_masks" / "1_0mpp"

        if os.path.exists(tissue_mask_path) and len(os.listdir(tissue_mask_path)) > 0:
            print(f"Processing modality '{modality}' for dataset '{dataset_name}'...")
        else:
            print(f"Warning: No tissue masks found for modality '{modality}' in dataset '{dataset_name}' at {tissue_mask_path}. Skipping.", file=sys.stderr)
            continue

        if modality == "he":
            dataset = HEImagingDataset(
                name=dataset_name,
                path=dataset_dir / dataset_name,
                verbose=True,
                resolution=1.0,
                crop_size=0
            )
        elif modality in ["imc", "codex", "cycif"]:
            dataset = MultiplexImagingDataset(
                name=dataset_name,
                modality=modality,
                path=dataset_dir / dataset_name,
                verbose=True,
                resolution=1.0,
                crop_size=0,
                normalization="identity",
            )
        else:
            raise ValueError(f"Unsupported modality '{modality}' for dataset '{dataset_name}'.")

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
        print(f"  Tissue mask fraction (mean +/- std): {np.mean(list(tissue_mask_frac.values())):.4f} +/- {np.std(list(tissue_mask_frac.values())):.4f}")
        print(f"  Tissue mask coverage after tiling (mean +/- std): {np.mean(list(tissue_mask_coverage.values())):.4f} +/- {np.std(list(tissue_mask_coverage.values())):.4f}")
        print(f"  Tissue mask number of tiles (mean +/- std): {np.mean(list(tissue_mask_num_tiles.values())):.2f} +/- {np.std(list(tissue_mask_num_tiles.values())):.2f}")

        df_path = dataset_dir / dataset_name / "metadata" / "crop_tiling" / f"{dataset.resolution}" / "tiling_stats.parquet"
        if df_path.exists():
            df = pd.read_parquet(df_path)
        else:
            os.makedirs(df_path.parent, exist_ok=True)
            df = pd.DataFrame(index=tids)
            df['tissue_mask_frac'] = pd.Series(tissue_mask_frac)
        df[f"{modality}_coverage_{crop_size}"] = pd.Series(tissue_mask_coverage)
        df[f"{modality}_num_tiles_{crop_size}"] = pd.Series(tissue_mask_num_tiles)
        df.to_parquet(df_path)

        coords_dir = dataset_dir / dataset_name / "segmentations" / modality / "crop_coordinates" / f"{dataset.resolution}"
        os.makedirs(coords_dir, exist_ok=True)
        with open(coords_dir / f"{crop_size}_tiles_coordinates.json", "w") as f:
            json.dump(tile_coordinates, f)


if __name__ == '__main__':
    main()
