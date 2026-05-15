from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from spora_io._config import get_datasets_dir
from spora_io.datasets.compose import ComposedImagingDataset
from spora_io.datasets.multiplex import MultiplexImagingDataset


SamplingUnit = Literal["tissues", "tiles"]

SIMPLE_MODALITIES = ("he", *tuple(sorted(MultiplexImagingDataset.VALID_MODALITIES)))


def _resolution_to_dir(resolution: float | str) -> str:
    return f"{str(resolution).replace('.', '_')}mpp"


def _as_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _discover_modalities(dataset_root: Path, requested: list[str] | None) -> list[str]:
    available = [modality for modality in SIMPLE_MODALITIES if (dataset_root / modality).is_dir()]

    ihc_root = dataset_root / "ihc"
    ihc_markers = (
        sorted(path.name for path in ihc_root.iterdir() if path.is_dir() and path.name.startswith("ihc_"))
        if ihc_root.exists()
        else []
    )
    available.extend(ihc_markers)

    if requested is None:
        return available

    expanded: list[str] = []
    for modality in requested:
        if modality == "ihc":
            expanded.extend(ihc_markers)
        else:
            expanded.append(modality)

    return [modality for modality in expanded if modality in available]


def _kind_for_modality(modality: str, kind: str) -> str:
    if modality == "he" or modality.startswith("ihc_"):
        return "complete"
    return kind


class SporaDataset:
    """Dataset-of-datasets wrapper for sampling tissues or tiles across cohorts.

    `SporaDataset` instantiates one :class:`ComposedImagingDataset` per dataset
    name, then builds either a tissue index or a concatenated tile-coordinate
    index. Samples are returned with a dataset name, tissue id, optional tile id,
    and a modality-to-tissue/tile mapping.
    """

    def __init__(
        self,
        dataset_names: str | Iterable[str],
        *,
        datasets_dir: str | Path | None = None,
        modalities: str | Iterable[str] | None = None,
        resolution: float | str = 1.0,
        tile_size: int | None = None,
        tile_strategy: str = "default",
        sampling_unit: SamplingUnit | None = None,
        verbose: bool = True,
        load_cell_metadata: bool = False,
        split: str | None = None,
        modality_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        dataset_modality_kwargs: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.dataset_names = _as_list(dataset_names)
        if not self.dataset_names:
            raise ValueError("dataset_names must contain at least one dataset name.")

        self.datasets_dir = Path(datasets_dir) if datasets_dir is not None else get_datasets_dir()
        self.requested_modalities = None if modalities is None else _as_list(modalities)
        self.resolution = resolution
        self.resolution_dir = _resolution_to_dir(resolution)
        self.tile_size = tile_size
        self.tile_strategy = tile_strategy
        self.verbose = verbose
        self.load_cell_metadata = load_cell_metadata
        self.split = split
        self.modality_kwargs = {k: dict(v) for k, v in (modality_kwargs or {}).items()}
        self.dataset_modality_kwargs = {
            dname: {mod: dict(kwargs) for mod, kwargs in per_dataset.items()}
            for dname, per_dataset in (dataset_modality_kwargs or {}).items()
        }
        self.rng = np.random.default_rng(seed)

        if sampling_unit is None:
            sampling_unit = "tiles" if tile_size is not None else "tissues"
        if sampling_unit == "tiles" and tile_size is None:
            raise ValueError("sampling_unit='tiles' requires tile_size.")
        if sampling_unit not in {"tissues", "tiles"}:
            raise ValueError("sampling_unit must be 'tissues' or 'tiles'.")
        self.sampling_unit: SamplingUnit = sampling_unit

        self.datasets: dict[str, ComposedImagingDataset] = {}
        self.modalities_by_dataset: dict[str, list[str]] = {}
        for dataset_name in self.dataset_names:
            dataset_root = self.datasets_dir / dataset_name
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

            dataset_modalities = _discover_modalities(dataset_root, self.requested_modalities)
            if not dataset_modalities:
                continue

            kwargs = self._merged_modality_kwargs(dataset_name)
            self.datasets[dataset_name] = ComposedImagingDataset(
                name=dataset_name,
                path=dataset_root,
                modalities=dataset_modalities,
                tile_size=tile_size,
                resolution=resolution,
                verbose=verbose,
                load_cell_metadata=load_cell_metadata,
                tile_strategy=tile_strategy,
                split=split,
                modality_kwargs=kwargs,
            )
            self.modalities_by_dataset[dataset_name] = dataset_modalities

        if not self.datasets:
            raise ValueError("No datasets with matching modalities were loaded.")

        self.tissue_index = self._build_tissue_index()
        self.tile_index = self._build_tile_index() if self.sampling_unit == "tiles" else None

    def _merged_modality_kwargs(self, dataset_name: str) -> dict[str, dict[str, Any]]:
        merged = {mod: dict(kwargs) for mod, kwargs in self.modality_kwargs.items()}
        for mod, kwargs in self.dataset_modality_kwargs.get(dataset_name, {}).items():
            merged.setdefault(mod, {}).update(kwargs)
        return merged

    def _build_tissue_index(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for dataset_name, dataset in self.datasets.items():
            for tissue_id in dataset.get_tissue_ids():
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "tissue_id": str(tissue_id),
                        "modalities": tuple(dataset.get_modalities_of_tissue(str(tissue_id))),
                    }
                )
        return pd.DataFrame(rows, columns=["dataset_name", "tissue_id", "modalities"])

    def _build_tile_index(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for dataset_name, dataset in self.datasets.items():
            coords_path = (
                self.datasets_dir
                / dataset_name
                / "tiling"
                / self.resolution_dir
                / self.tile_strategy
                / f"{self.tile_size}_tile_coordinates.parquet"
            )
            if not coords_path.exists():
                if self.verbose:
                    print(f"Skipping missing tile coordinates: {coords_path}")
                continue

            coords = pd.read_parquet(coords_path)
            required = {"tissue_id", "tile_id", "row", "col"}
            if not required.issubset(coords.columns):
                raise ValueError(f"Tile coordinates at {coords_path} are missing columns {sorted(required)}.")

            tissue_ids = set(dataset.get_tissue_ids())
            coords = coords[coords["tissue_id"].astype(str).isin(tissue_ids)].copy()
            if coords.empty:
                continue
            coords.insert(0, "dataset_name", dataset_name)
            frames.append(coords[["dataset_name", "tissue_id", "tile_id", "row", "col"]])

        if not frames:
            raise FileNotFoundError(
                f"No tile coordinate parquet files found for tile_size={self.tile_size}, "
                f"resolution={self.resolution_dir}, strategy={self.tile_strategy}."
            )

        tile_index = pd.concat(frames, ignore_index=True)
        tile_index.insert(0, "global_tile_id", np.arange(len(tile_index), dtype=np.int64))
        return tile_index

    def __len__(self) -> int:
        if self.sampling_unit == "tiles":
            assert self.tile_index is not None
            return len(self.tile_index)
        return len(self.tissue_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sampling_unit == "tiles":
            assert self.tile_index is not None
            row = self.tile_index.iloc[int(index)]
            return self.get_tile_sample(
                dataset_name=str(row["dataset_name"]),
                tissue_id=str(row["tissue_id"]),
                tile_id=int(row["tile_id"]),
            )

        row = self.tissue_index.iloc[int(index)]
        return self.get_tissue_sample(dataset_name=str(row["dataset_name"]), tissue_id=str(row["tissue_id"]))

    def get_dataset(self, dataset_name: str) -> ComposedImagingDataset:
        return self.datasets[dataset_name]

    def get_tissue_ids(self, dataset_name: str | None = None) -> list[str]:
        if dataset_name is not None:
            return [str(tissue_id) for tissue_id in self.datasets[dataset_name].get_tissue_ids()]
        return [str(row.tissue_id) for row in self.tissue_index.itertuples()]

    def get_tissue_sample(
        self,
        *,
        dataset_name: str,
        tissue_id: str,
        kind: str = "uniprot_filtered",
        preprocess: bool = True,
        image_mode: str = "CHW",
    ) -> dict[str, Any]:
        dataset = self.datasets[dataset_name]
        modalities = dataset.get_modalities_of_tissue(tissue_id)
        return {
            "dataset_name": dataset_name,
            "tissue_id": tissue_id,
            "modalities": {
                modality: dataset.get_unimodal_tissue(
                    tissue_id,
                    modality=modality,
                    kind=_kind_for_modality(modality, kind),
                    preprocess=preprocess,
                    image_mode=image_mode,
                )
                for modality in modalities
            },
        }

    def get_tile_sample(
        self,
        *,
        dataset_name: str,
        tissue_id: str,
        tile_id: int,
        kind: str = "uniprot_filtered",
        preprocess: bool = True,
        image_mode: str = "CHW",
    ) -> dict[str, Any]:
        dataset = self.datasets[dataset_name]
        modalities = dataset.get_modalities_of_tissue(tissue_id)
        return {
            "dataset_name": dataset_name,
            "tissue_id": tissue_id,
            "tile_id": int(tile_id),
            "modalities": {
                modality: dataset.get_unimodal_tile(
                    tissue_id,
                    tile_id,
                    modality=modality,
                    kind=_kind_for_modality(modality, kind),
                    preprocess=preprocess,
                    image_mode=image_mode,
                )
                for modality in modalities
            },
        }

    def sample_random_tissue(self, **kwargs: Any) -> dict[str, Any]:
        idx = int(self.rng.integers(0, len(self.tissue_index)))
        row = self.tissue_index.iloc[idx]
        return self.get_tissue_sample(
            dataset_name=str(row["dataset_name"]),
            tissue_id=str(row["tissue_id"]),
            **kwargs,
        )

    def sample_random_tile(self, **kwargs: Any) -> dict[str, Any]:
        if self.tile_index is None:
            raise ValueError("Tile sampling is unavailable because sampling_unit != 'tiles'.")
        idx = int(self.rng.integers(0, len(self.tile_index)))
        row = self.tile_index.iloc[idx]
        return self.get_tile_sample(
            dataset_name=str(row["dataset_name"]),
            tissue_id=str(row["tissue_id"]),
            tile_id=int(row["tile_id"]),
            **kwargs,
        )

    def sample_random(self, **kwargs: Any) -> dict[str, Any]:
        if self.sampling_unit == "tiles":
            return self.sample_random_tile(**kwargs)
        return self.sample_random_tissue(**kwargs)

    def __repr__(self) -> str:
        # also to repr add how many tissues and tiles per dataset
        dataset_summaries = []
        for dataset_name, dataset in self.datasets.items():
            n_tissues = len(dataset.get_tissue_ids())
            n_tiles = len(self.tile_index[self.tile_index["dataset_name"] == dataset_name]) if self.tile_index is not None else "N/A"
            dataset_summaries.append(f"{dataset_name} (tissues: {n_tissues}, tiles: {n_tiles})")
        return (
            f"SporaDataset(datasets=[{', '.join(dataset_summaries)}], "
            f"sampling_unit={self.sampling_unit!r}, resolution={self.resolution!r}, "
            f"tile_size={self.tile_size!r}, split={self.split!r}, n={len(self)})"
        )
