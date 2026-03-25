from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
import zarr

### Run this file like this:
# requires pytest, pytest-html
# python -m pytest -vv -s --html=DatasetV2-Report.html test_datasets.py

from spatialprot_data._config import get_datasets_dir

DATASETS_ROOT = get_datasets_dir()
SKIP_DATASETS = {
    name.strip()
    for name in ["marker_embeddings"]
    if name.strip()
}

DATASET_NAME_RE = re.compile(r"^[a-z0-9_]+$")
TISSUE_ID_RE = re.compile(r"^[a-z0-9_]+_[a-z]{8}_[0-9]{4}$")
MPP_RE = re.compile(r"^[0-9]+(?:_[0-9]+)?mpp$")
UNIPROT_TOKEN_RE = re.compile(
    r"^(Excluded|Exclude|Not found|(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})(?:-[0-9]+)?)$"
)

RESERVED_ROOT_DIRS = {"metadata", "segmentations", ".setup", "tmp", "imc_new_processed"}
SP_MODALITIES = {"imc", "cycif", "codex"}
RGB_MODALITIES = {"he"}


@dataclass(frozen=True)
class ImageEntry:
    dataset_name: str
    tissue_id: str
    modality_key: str
    base_modality: str
    mpp: str | None
    path: Path


def _dataset_dirs() -> list[Path]:
    if not DATASETS_ROOT.exists():
        return []
    dirs = [
        p
        for p in sorted(DATASETS_ROOT.iterdir())
        if p.is_dir() and p.name not in SKIP_DATASETS and (p / "metadata").is_dir()
    ]
    return dirs


DATASET_DIRS = _dataset_dirs()


def _parse_uniprot_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[;,|\s]+", text) if token.strip()]


def _is_bool_like_series(series: pd.Series) -> bool:
    if series.empty:
        return True
    lowered = series.astype(str).str.strip().str.lower()
    valid = {"0", "1", "true", "false"}
    return lowered.isin(valid).all()


def _discover_image_entries(dataset_dir: Path) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    dataset_name = dataset_dir.name

    for top in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name not in RESERVED_ROOT_DIRS):
        if top.name == "ihc":
            zarr_paths = sorted(top.glob("ihc*/*mpp/*.zarr"))
            for zpath in zarr_paths:
                rel_parts = zpath.relative_to(top).parts
                marker = next((p for p in rel_parts if p.startswith("ihc_")), "ihc")
                mpp = next((p for p in rel_parts if MPP_RE.match(p)), None)
                entries.append(
                    ImageEntry(
                        dataset_name=dataset_name,
                        tissue_id=zpath.stem,
                        modality_key=marker,
                        base_modality="ihc",
                        mpp=mpp,
                        path=zpath,
                    )
                )
            continue

        zarr_paths = sorted(top.glob("*mpp/*.zarr"))
        for zpath in zarr_paths:
            rel_parts = zpath.relative_to(top).parts
            mpp = next((p for p in rel_parts if MPP_RE.match(p)), None)
            entries.append(
                ImageEntry(
                    dataset_name=dataset_name,
                    tissue_id=zpath.stem,
                    modality_key=top.name,
                    base_modality=top.name,
                    mpp=mpp,
                    path=zpath,
                )
            )

    return entries


def _zarr_channel_count(zarr_path: Path, modality_key: str) -> int:
    arr = zarr.open(str(zarr_path), mode="r")
    shape = tuple(int(s) for s in arr.shape)

    assert len(shape) in (2, 3), f"Unexpected array rank at {zarr_path}: shape={shape}"

    if modality_key in RGB_MODALITIES or modality_key.startswith("ihc_"):
        assert len(shape) == 3, f"RGB modality expects rank-3 array at {zarr_path}, got shape={shape}"
        assert shape[-1] == 3, f"RGB modality expects 3 channels in last axis at {zarr_path}, got shape={shape}"
        _ = arr[0, 0, 0]
        return int(shape[-1])

    if len(shape) == 2:
        _ = arr[0, 0]
        return 1

    _ = arr[0, 0, 0]
    return int(shape[0])


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str], file_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns in {file_path}: {missing}"


@pytest.mark.skipif(not DATASETS_ROOT.exists(), reason="DATASETS_V2_ROOT does not exist")
def test_dataset_root_discovery() -> None:
    assert DATASET_DIRS, f"No dataset directories found in {DATASETS_ROOT}"


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_dataset_name_convention(dataset_dir: Path) -> None:
    assert DATASET_NAME_RE.match(dataset_dir.name), (
        f"Dataset folder name violates convention: {dataset_dir.name}"
    )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_metadata_files_and_tissues_schema(dataset_dir: Path) -> None:
    metadata_dir = dataset_dir / "metadata"
    tissues_file = metadata_dir / "tissues.parquet"

    assert tissues_file.exists(), f"Missing required file: {tissues_file}"
    tissues = pd.read_parquet(tissues_file)
    _ensure_required_columns(
        tissues,
        required=["tissue_id", "patient_id", "alignment", "modality"],
        file_path=tissues_file,
    )

    assert len(tissues) > 0, f"tissues.parquet is empty: {tissues_file}"
    assert tissues["tissue_id"].notna().all(), f"Null tissue_id values found in {tissues_file}"
    assert tissues["modality"].notna().all(), f"Null modality values found in {tissues_file}"

    invalid_tids = [tid for tid in tissues["tissue_id"].astype(str).unique() if not TISSUE_ID_RE.match(tid)]
    assert not invalid_tids, (
        f"Invalid tissue_id format in {tissues_file}. First invalid IDs: {invalid_tids[:10]}"
    )

    duplicated = tissues.duplicated(subset=["tissue_id", "modality"]).sum()
    assert duplicated == 0, (
        f"Found duplicated (tissue_id, modality) rows in {tissues_file}: {duplicated}"
    )

    modality_values = tissues["modality"].astype(str)
    invalid_modalities = [
        m for m in sorted(modality_values.unique()) if not (m in {"he", "imc", "codex", "cycif"} or m.startswith("ihc_"))
    ]
    assert not invalid_modalities, (
        f"Unexpected modality values in {tissues_file}: {invalid_modalities}"
    )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_images_exist_and_no_untracked_images(dataset_dir: Path) -> None:
    tissues = pd.read_parquet(dataset_dir / "metadata" / "tissues.parquet")
    image_entries = _discover_image_entries(dataset_dir)
    assert image_entries, f"No .zarr images found for dataset: {dataset_dir.name}"

    row_pairs = set(zip(tissues["tissue_id"].astype(str), tissues["modality"].astype(str)))
    image_pairs = {(e.tissue_id, e.modality_key) for e in image_entries}

    missing_images = sorted(row_pairs - image_pairs)
    assert not missing_images, (
        f"Rows in tissues.parquet without matching .zarr images in {dataset_dir.name}. "
        f"First missing pairs: {missing_images[:20]}"
    )

    extra_images = sorted(image_pairs - row_pairs)
    assert not extra_images, (
        f"Found .zarr images with no matching row in tissues.parquet in {dataset_dir.name}. "
        f"First extra pairs: {extra_images[:20]}"
    )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_zarr_integrity_and_channel_axis(dataset_dir: Path) -> None:
    image_entries = _discover_image_entries(dataset_dir)
    assert image_entries, f"No .zarr images found for dataset: {dataset_dir.name}"

    failures: list[str] = []
    for entry in image_entries:
        try:
            channels = _zarr_channel_count(entry.path, entry.modality_key)
            assert channels > 0, f"Non-positive channel count in {entry.path}"
        except Exception as exc:
            failures.append(f"{entry.path}: {exc}")

    assert not failures, "Corrupted or invalid zarr images detected:\n" + "\n".join(failures[:100])


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_spatial_proteomics_channel_tables(dataset_dir: Path) -> None:
    tissues = pd.read_parquet(dataset_dir / "metadata" / "tissues.parquet")
    image_entries = _discover_image_entries(dataset_dir)

    for modality in SP_MODALITIES:
        modality_dir = dataset_dir / modality
        if not modality_dir.exists():
            continue

        channels_file = modality_dir / "channels.parquet"
        channels_per_tissue_file = modality_dir / "channels_per_tissue.parquet"

        assert channels_file.exists(), f"Missing channels.parquet for {dataset_dir.name}/{modality}"
        assert channels_per_tissue_file.exists(), (
            f"Missing channels_per_tissue.parquet for {dataset_dir.name}/{modality}"
        )

        channels_df = pd.read_parquet(channels_file)
        _ensure_required_columns(
            channels_df,
            required=["channel_name", "index", "qc_pass", "uniprot_id"],
            file_path=channels_file,
        )

        assert channels_df["channel_name"].notna().all(), f"Null channel names in {channels_file}"
        assert channels_df["channel_name"].astype(str).str.len().gt(0).all(), (
            f"Empty channel names in {channels_file}"
        )
        assert channels_df["channel_name"].astype(str).is_unique, (
            f"Duplicate channel names in {channels_file}"
        )
        assert channels_df["index"].notna().all(), f"Null index values in {channels_file}"

        index_values = channels_df["index"].astype(int).tolist()
        assert len(set(index_values)) == len(index_values), f"Duplicate channel indices in {channels_file}"
        assert min(index_values) == 0, f"Channel indices must start at 0 in {channels_file}"
        assert max(index_values) == len(index_values) - 1, (
            f"Channel indices must be contiguous in {channels_file}"
        )

        assert channels_df["qc_pass"].notna().all(), f"Null qc_pass values in {channels_file}"
        assert _is_bool_like_series(channels_df["qc_pass"]), (
            f"qc_pass must be boolean-like (True/False/0/1) in {channels_file}"
        )

        bad_uniprot_rows: list[tuple[int, str]] = []
        for idx, value in channels_df["uniprot_id"].items():
            tokens = _parse_uniprot_tokens(value)
            if not tokens:
                bad_uniprot_rows.append((int(idx), str(value)))
                continue
            if not all(UNIPROT_TOKEN_RE.match(token) for token in tokens):
                bad_uniprot_rows.append((int(idx), str(value)))

        assert not bad_uniprot_rows, (
            f"Invalid uniprot_id entries in {channels_file}. First invalid rows: {bad_uniprot_rows[:20]}"
        )

        cpt_df = pd.read_parquet(channels_per_tissue_file)
        _ensure_required_columns(cpt_df, required=["tissue_id"], file_path=channels_per_tissue_file)
        assert cpt_df["tissue_id"].is_unique, f"Duplicate tissue_id rows in {channels_per_tissue_file}"

        expected_channels = set(channels_df["channel_name"].astype(str).tolist())
        cpt_channels = set(cpt_df.columns) - {"tissue_id"}
        assert expected_channels == cpt_channels, (
            f"Channel mismatch between {channels_file} and {channels_per_tissue_file}. "
            f"Only in channels: {sorted(expected_channels - cpt_channels)[:15]}, "
            f"only in channels_per_tissue: {sorted(cpt_channels - expected_channels)[:15]}"
        )

        if cpt_channels:
            for col in sorted(cpt_channels):
                assert cpt_df[col].notna().all(), f"Null values in channels_per_tissue column '{col}'"
                assert _is_bool_like_series(cpt_df[col]), (
                    f"Non-binary values in channels_per_tissue column '{col}'"
                )

        modality_tissues = set(
            tissues.loc[tissues["modality"].astype(str) == modality, "tissue_id"].astype(str).tolist()
        )
        cpt_tissues = set(cpt_df["tissue_id"].astype(str).tolist())

        missing_cpt_rows = sorted(modality_tissues - cpt_tissues)
        assert not missing_cpt_rows, (
            f"Missing channels_per_tissue rows for modality {modality} in {dataset_dir.name}. "
            f"First missing tissue IDs: {missing_cpt_rows[:20]}"
        )

        extra_cpt_rows = sorted(cpt_tissues - modality_tissues)
        assert not extra_cpt_rows, (
            f"Extra channels_per_tissue rows for modality {modality} in {dataset_dir.name}. "
            f"First extra tissue IDs: {extra_cpt_rows[:20]}"
        )

        zarr_by_tissue: dict[str, int] = {}
        for entry in image_entries:
            if entry.modality_key != modality:
                continue
            zarr_by_tissue[entry.tissue_id] = _zarr_channel_count(entry.path, entry.modality_key)

        channel_total = len(expected_channels)
        for _, row in cpt_df.iterrows():
            tissue_id = str(row["tissue_id"])
            if tissue_id not in zarr_by_tissue:
                continue

            available = int(pd.Series([row[c] for c in expected_channels]).astype(bool).sum())
            observed = zarr_by_tissue[tissue_id]
            valid_counts = {available, channel_total}
            assert observed in valid_counts, (
                f"Channel-count mismatch for {dataset_dir.name}/{modality}/{tissue_id}: "
                f"zarr has {observed}, channels_per_tissue has {available}, channels table has {channel_total}"
            )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_cells_parquet_if_present(dataset_dir: Path) -> None:
    cells_file = dataset_dir / "metadata" / "cells.parquet"
    if not cells_file.exists():
        return

    tissues = pd.read_parquet(dataset_dir / "metadata" / "tissues.parquet")
    cells = pd.read_parquet(cells_file)
    _ensure_required_columns(cells, required=["tissue_id", "cell_id"], file_path=cells_file)

    assert cells["tissue_id"].notna().all(), f"Null tissue_id in {cells_file}"
    assert cells["cell_id"].notna().all(), f"Null cell_id in {cells_file}"

    unknown_tissues = sorted(set(cells["tissue_id"].astype(str)) - set(tissues["tissue_id"].astype(str)))
    assert not unknown_tissues, (
        f"cells.parquet references unknown tissue_id values in {dataset_dir.name}. "
        f"First unknown IDs: {unknown_tissues[:20]}"
    )

    duplicated = cells.duplicated(subset=["tissue_id", "cell_id"]).sum()
    assert duplicated == 0, (
        f"Duplicate (tissue_id, cell_id) combinations in {cells_file}: {duplicated}"
    )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_segmentation_mask_file_conventions(dataset_dir: Path) -> None:
    seg_dir = dataset_dir / "segmentations"
    if not seg_dir.exists():
        pytest.fail(f"Missing segmentations directory: {seg_dir}")

    npz_files = list(seg_dir.glob("**/*.npz"))
    if not npz_files:
        return

    bad_names = [p for p in npz_files if not TISSUE_ID_RE.match(p.stem)]
    assert not bad_names, (
        f"Found segmentation masks with invalid file names in {dataset_dir.name}. "
        f"First invalid files: {[str(p) for p in bad_names[:20]]}"
    )


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda p: p.name)
def test_all_tissues_have_tissue_masks(dataset_dir: Path) -> None:
    tissues = pd.read_parquet(dataset_dir / "metadata" / "tissues.parquet")
    _ensure_required_columns(
        tissues,
        required=["tissue_id", "modality"],
        file_path=dataset_dir / "metadata" / "tissues.parquet",
    )

    image_entries = _discover_image_entries(dataset_dir)
    by_pair: dict[tuple[str, str], list[ImageEntry]] = {}
    for entry in image_entries:
        by_pair.setdefault((entry.tissue_id, entry.modality_key), []).append(entry)

    missing_masks: list[str] = []
    invalid_masks: list[str] = []

    for _, row in tissues[["tissue_id", "modality"]].drop_duplicates().iterrows():
        tissue_id = str(row["tissue_id"])
        modality = str(row["modality"])
        entries = by_pair.get((tissue_id, modality), [])

        if entries:
            expected_mpps = sorted({e.mpp for e in entries if e.mpp})
            base_modality = entries[0].base_modality
        else:
            # If no image exists for this row, keep expectations conservative and check any mpp folder.
            expected_mpps = []
            base_modality = "ihc" if modality.startswith("ihc_") else modality

        mask_root = dataset_dir / "segmentations" / base_modality / "tissue_masks"

        if expected_mpps:
            for mpp in expected_mpps:
                mask_file = mask_root / mpp / f"{tissue_id}.npz"
                if not mask_file.exists():
                    missing_masks.append(str(mask_file))
                    continue
                try:
                    with np.load(mask_file) as data:
                        if "mask" not in data:
                            invalid_masks.append(f"{mask_file} (missing key 'mask')")
                except Exception as exc:
                    invalid_masks.append(f"{mask_file} ({exc})")
        else:
            # Fallback: require a mask at any available mpp for the modality.
            candidates = list(mask_root.glob(f"*/{tissue_id}.npz"))
            if not candidates:
                missing_masks.append(f"{mask_root}/*/{tissue_id}.npz")
                continue
            for mask_file in candidates:
                try:
                    with np.load(mask_file) as data:
                        if "mask" not in data:
                            invalid_masks.append(f"{mask_file} (missing key 'mask')")
                except Exception as exc:
                    invalid_masks.append(f"{mask_file} ({exc})")

    assert not missing_masks, (
        "Missing tissue mask files for tissue IDs. "
        f"First missing paths: {missing_masks[:30]}"
    )
    assert not invalid_masks, (
        "Invalid tissue mask files detected. "
        f"First invalid files: {invalid_masks[:30]}"
    )
