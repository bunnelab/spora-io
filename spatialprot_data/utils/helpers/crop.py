from numpy.core.defchararray import mod
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from pathlib import Path
import os
from spatialprot_data.datasets.he import HEImagingDataset
from spatialprot_data.datasets.multiplex import MultiplexImagingDataset


VALID_MODALITIES = ["he", "imc", "codex", "cycif"]
DATASET_DIR = Path("/mnt/aimm/scratch/datasets_v2/")


@dataclass
class Tile:
    y: int
    x: int
    h: int
    w: int
    valid_ratio: float
    gain: int   # newly covered valid pixels when this tile was selected


def _integral_image(arr: np.ndarray) -> np.ndarray:
    """Integral image with one zero-padded row/col at the top-left."""
    arr = arr.astype(np.int64, copy=False)
    return np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def _rect_sums_vec(ii: np.ndarray, ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> np.ndarray:
    """Vectorised rect-sum for arrays of (y, x) positions. Returns int64 array of shape (N,)."""
    y2 = ys + h
    x2 = xs + w
    return ii[y2, x2] - ii[ys, x2] - ii[y2, xs] + ii[ys, xs]


def _candidate_starts(n: int, tile: int, stride: int) -> np.ndarray:
    """Candidate start positions along one dimension."""
    if n <= tile:
        return np.array([0], dtype=np.int32)
    starts = np.arange(0, n - tile + 1, stride, dtype=np.int32)
    if starts[-1] != n - tile:
        starts = np.append(starts, n - tile)
    return starts


def _pad_to_tile(mask: np.ndarray, tile: int) -> np.ndarray:
    h, w = mask.shape
    H, W = max(h, tile), max(w, tile)
    if H == h and W == w:
        return mask
    out = np.zeros((H, W), dtype=mask.dtype)
    out[:h, :w] = mask
    return out


def best_mask_tiling_try_to_stop(
    mask: np.ndarray,
    tile_size: int,
    stride: int = None,
    tolerance: float = 0.2,
    coverage_goal: float = 0.99,
    min_gain_ratio: float = 0.05,
    max_tiles: int = None,
    allow_overlap: bool = True,
    progress: bool = False,
    progress_desc: str = "Tiling",
):
    """
    Find a good tiling of the unmasked region with adaptive stopping.

    Stopping is controlled by TWO criteria that must *both* be true to stop:

      1. covered_valid / total_valid  >= coverage_goal
      2. best_gain / tile_area        <  min_gain_ratio

    This makes the two parameters complementary:

    - ``coverage_goal=0.98, min_gain_ratio=0.05``
        Runs past 0.98 as long as tiles still contribute ≥5 % new pixels,
        potentially reaching near-full coverage for free.

    - ``coverage_goal=1.0, min_gain_ratio=0.05``
        Aims for full coverage but bails early once tiles become mostly
        redundant (< 5 % new pixels), avoiding useless overlap.

    Set ``min_gain_ratio=0.0`` to recover the original hard-cutoff behaviour
    (stops exactly at coverage_goal).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask of shape (H, W), with 1 = valid/unmasked, 0 = masked.
    tile_size : int
        Tile size C, so each tile is C x C.
    stride : int
        Sliding stride. Defaults to tile_size (non-overlapping grid).
    tolerance : float
        Maximum fraction of invalid pixels allowed inside a tile (0 = strict).
    coverage_goal : float
        Soft lower bound on coverage — the loop will not stop *below* this
        unless gains have already hit zero.
    min_gain_ratio : float
        Soft upper bound on marginal efficiency — once the best remaining tile
        covers less than this fraction of its area in new pixels, AND
        coverage_goal has been reached, the loop stops.
        Range [0, 1). Default 0.05.
    max_tiles : int or None
        Hard cap on number of selected tiles.
    allow_overlap : bool
        If False, selected tiles cannot overlap each other.
    progress : bool
        Show a tqdm progress bar on stderr.
    progress_desc : str
        Label prefix on the progress bar.

    Returns
    -------
    tiles : list[Tile]
    stats : dict
    covered_mask : np.ndarray
    """
    if stride is None:
        stride = tile_size

    if not (0.0 <= tolerance < 1.0):
        raise ValueError("tolerance must be in [0, 1).")
    if not (0.0 < coverage_goal <= 1.0):
        raise ValueError("coverage_goal must be in (0, 1].")
    if not (0.0 <= min_gain_ratio < 1.0):
        raise ValueError("min_gain_ratio must be in [0, 1).")

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    mask = _pad_to_tile(mask, tile_size)

    H, W = mask.shape
    total_valid = int(mask.sum())
    tile_area   = tile_size * tile_size
    # Absolute pixel threshold derived from min_gain_ratio.
    min_gain_px = int(np.ceil(min_gain_ratio * tile_area))

    if total_valid == 0:
        return [], {
            "num_tiles": 0,
            "candidate_count": 0,
            "covered_valid_pixels": 0,
            "total_valid_pixels": 0,
            "coverage_ratio": 1.0,
            "stop_reason": "empty_mask",
        }, np.zeros_like(mask, dtype=np.uint8)

    min_valid_ratio = 1.0 - tolerance
    use_progress    = progress 

    # ------------------------------------------------------------------ #
    # Phase 1 – vectorised candidate filtering                            #
    # ------------------------------------------------------------------ #
    ys_starts = _candidate_starts(H, tile_size, stride)
    xs_starts = _candidate_starts(W, tile_size, stride)

    ys_grid, xs_grid = np.meshgrid(ys_starts, xs_starts, indexing="ij")
    all_ys = ys_grid.ravel().astype(np.int32)
    all_xs = xs_grid.ravel().astype(np.int32)

    ii_full    = _integral_image(mask)
    all_valids = _rect_sums_vec(ii_full, all_ys, all_xs, tile_size, tile_size)
    all_ratios = all_valids / float(tile_area)

    keep    = (all_valids > 0) & (all_ratios >= min_valid_ratio)
    cand_y  = all_ys[keep].copy()
    cand_x  = all_xs[keep].copy()
    cand_vr = all_ratios[keep].astype(np.float32)

    total_candidates = int(keep.sum())
    if use_progress:
        tqdm.write(
            f"{progress_desc} – Phase 1 done: "
            f"{total_candidates:,} / {len(all_ys):,} candidates pass tolerance filter",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Phase 2 – greedy selection with incremental gain updates            #
    # ------------------------------------------------------------------ #
    uncovered     = mask.copy()
    selected_tiles = []
    covered_valid  = 0
    stop_reason    = "candidates_exhausted"

    # Compute all initial gains once, upfront.
    ii_unc = _integral_image(uncovered)
    gains  = _rect_sums_vec(ii_unc, cand_y, cand_x, tile_size, tile_size).astype(np.int64)

    coverage_bar: Optional["tqdm"] = None
    if use_progress:
        coverage_bar = tqdm(
            total=total_valid,
            desc=f"{progress_desc}",
            unit="px",
            dynamic_ncols=True,
            file=sys.stderr,
            miniters=1,
            mininterval=0.1,
        )

    try:
        while len(cand_y):
            best_idx  = int(np.argmax(gains))
            best_gain = int(gains[best_idx])

            if best_gain <= 0:
                stop_reason = "no_gain"
                break

            # ---- adaptive dual stopping criterion ----
            # coverage_goal=1.0 means "aim for full coverage" — since exactly
            # 1.0 is nearly unreachable, treat it as no floor and let
            # min_gain_ratio alone govern stopping.  Any other value acts as a
            # minimum floor: we won't stop until coverage has reached it.
            coverage_reached = True if coverage_goal >= 1.0 else (covered_valid / total_valid) >= coverage_goal
            gain_too_low     = best_gain < min_gain_px
            if coverage_reached and gain_too_low:
                stop_reason = "adaptive_stop"
                break

            sy, sx = int(cand_y[best_idx]), int(cand_x[best_idx])
            selected_tiles.append(
                Tile(
                    y=sy, x=sx, h=tile_size, w=tile_size,
                    valid_ratio=float(cand_vr[best_idx]),
                    gain=best_gain,
                )
            )

            # Identify spatially overlapping candidates.
            remove   = np.zeros(len(cand_y), dtype=bool)
            remove[best_idx] = True

            affected = (
                ~remove &
                (cand_y < sy + tile_size) & (cand_y + tile_size > sy) &
                (cand_x < sx + tile_size) & (cand_x + tile_size > sx)
            )

            if not allow_overlap:
                remove |= affected
            elif affected.any():
                # Local integral image over the selected patch — O(tile²).
                local_patch = uncovered[sy: sy + tile_size, sx: sx + tile_size]
                local_ii    = _integral_image(local_patch)

                ay  = cand_y[affected]
                ax  = cand_x[affected]
                iy1 = np.maximum(ay, sy) - sy
                iy2 = np.minimum(ay + tile_size, sy + tile_size) - sy
                ix1 = np.maximum(ax, sx) - sx
                ix2 = np.minimum(ax + tile_size, sx + tile_size) - sx

                decrements = (
                    local_ii[iy2, ix2]
                    - local_ii[iy1, ix2]
                    - local_ii[iy2, ix1]
                    + local_ii[iy1, ix1]
                )
                gains[affected] -= decrements

            # Zero out uncovered AFTER computing decrements.
            uncovered[sy: sy + tile_size, sx: sx + tile_size] = 0

            keep_mask = ~remove
            cand_y    = cand_y[keep_mask]
            cand_x    = cand_x[keep_mask]
            cand_vr   = cand_vr[keep_mask]
            gains     = gains[keep_mask]

            # Incremental coverage (no uncovered.sum() scan needed).
            covered_valid += best_gain

            if coverage_bar is not None:
                bar_delta = covered_valid - coverage_bar.n
                if bar_delta > 0:
                    coverage_bar.update(bar_delta)
                coverage_bar.set_postfix(
                    tiles=len(selected_tiles),
                    cov=f"{covered_valid / total_valid:.1%}",
                    gain=f"{best_gain / tile_area:.1%}",
                    cands=len(cand_y),
                    refresh=True,
                )

            if max_tiles is not None and len(selected_tiles) >= max_tiles:
                stop_reason = "max_tiles"
                break

    finally:
        if coverage_bar is not None:
            coverage_bar.close()

    covered_mask        = (mask > uncovered).astype(np.uint8)
    covered_valid_final = int(covered_mask.sum())

    total_tile_area  = len(selected_tiles) * tile_area
    overlap_pixels   = max(0, total_tile_area - covered_valid_final)
    overlap_ratio    = overlap_pixels / total_tile_area if total_tile_area > 0 else 0.0

    stats = {
        "num_tiles": len(selected_tiles),
        "candidate_count": int(keep.sum()),
        "accepted_candidates": len(selected_tiles),
        "covered_valid_pixels": covered_valid_final,
        "total_valid_pixels": total_valid,
        "coverage_ratio": covered_valid_final / float(total_valid),
        # Intersection: fraction of total tile area that overlaps with already-covered pixels.
        # overlap_ratio=0.0 means every tile was 100% new; 0.5 means half the placed
        # tile area was redundant overlap.
        "total_tile_area": total_tile_area,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio": overlap_ratio,
        "stop_reason": stop_reason,
    }

    return selected_tiles, stats, covered_mask




if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <crop_size> <modality (optional)>", file=sys.stderr)
        sys.exit(1)
    dataset_name = sys.argv[1]
    crop_size = int(sys.argv[2])
    modality = sys.argv[3] if len(sys.argv) > 3 else None

    valid_modalities = [m for m in VALID_MODALITIES if (DATASET_DIR / dataset_name / m).is_dir()] if modality is None else [modality] if modality in VALID_MODALITIES else []
    if not valid_modalities:
        print(f"Error: No valid modalities found for dataset '{dataset_name}' in {DATASET_DIR}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found valid modalities for dataset '{dataset_name}': {valid_modalities}")

    for modality in valid_modalities:
        tissue_mask_path = DATASET_DIR / dataset_name / "segmentations" / modality / "tissue_masks" / "1_0mpp"

        if os.path.exists(tissue_mask_path) and len(os.listdir(tissue_mask_path)) > 0:
            print(f"Processing modality '{modality}' for dataset '{dataset_name}'...")
        else:
            print(f"Warning: No tissue masks found for modality '{modality}' in dataset '{dataset_name}' at {tissue_mask_path}. Skipping.", file=sys.stderr)
            continue            

        if modality == "he":
            dataset = HEImagingDataset(
                name=dataset_name,
                path=DATASET_DIR / dataset_name,
                verbose=True,
                resolution=1.0,
                crop_size=0
            )
        elif modality in ["imc", "codex", "cycif"]:
            dataset = MultiplexImagingDataset(
                name=dataset_name,
                modality=modality,
                path=DATASET_DIR / dataset_name,
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
        print(f"  Tissue mask fraction (mean ± std): {np.mean(list(tissue_mask_frac.values())):.4f} ± {np.std(list(tissue_mask_frac.values())):.4f}")
        print(f"  Tissue mask coverage after tiling (mean ± std): {np.mean(list(tissue_mask_coverage.values())):.4f} ± {np.std(list(tissue_mask_coverage.values())):.4f}")
        print(f"  Tissue mask number of tiles (mean ± std): {np.mean(list(tissue_mask_num_tiles.values())):.2f} ± {np.std(list(tissue_mask_num_tiles.values())):.2f}")

        # dataframe with index: tid, columns: mask_frac_{crop_size}, coverage_{crop_size}, num_tiles_{crop_size}
        # check if it exists, if yes, load it and update it, if no, create it
        import pandas as pd
        df_path = DATASET_DIR / dataset_name / "metadata" / f"crop_tiling" / f"{dataset.resolution}" / "tiling_stats.parquet"
        if df_path.exists():
            df = pd.read_parquet(df_path)
        else:
            os.makedirs(df_path.parent, exist_ok=True)
            df = pd.DataFrame(index=tids)
            df['tissue_mask_frac'] = pd.Series(tissue_mask_frac)
        # add/update columns for this modality and crop size
        df[f"{modality}_coverage_{crop_size}"] = pd.Series(tissue_mask_coverage)
        df[f"{modality}_num_tiles_{crop_size}"] = pd.Series(tissue_mask_num_tiles)
        df.to_parquet(df_path)

        # save coords
        os.makedirs(DATASET_DIR / dataset_name / "segmentations" / modality / "crop_coordinates" / f"{dataset.resolution}", exist_ok=True)
        import json
        with open(DATASET_DIR / dataset_name / "segmentations" / modality / "crop_coordinates" / f"{dataset.resolution}" / f"{crop_size}_tiles_coordinates.json", "w") as f:
            json.dump(tile_coordinates, f)


        







