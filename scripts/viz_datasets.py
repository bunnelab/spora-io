#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, Input, Label, Select, Static, TabbedContent, TabPane

from spora_io._config import get_datasets_dir

SUPPORTED_ROOT_MODALITIES = ("he", "imc", "codex", "cycif", "ihc")
SIMPLE_MODALITIES = ("he", "imc", "codex", "cycif")
MULTIPLEX_MODALITIES = {"imc", "codex", "cycif"}


def is_blank_select_value(value: object) -> bool:
    return value in (None, False, "") or value == Select.BLANK


def expected_root(modality: str) -> str | None:
    if modality in SIMPLE_MODALITIES:
        return modality
    if modality.startswith("ihc_"):
        return "ihc"
    return None


def discover_loader_modalities(dataset_dir: Path, metadata_modalities: list[str]) -> list[str]:
    modalities: set[str] = set(m for m in metadata_modalities if expected_root(m) is not None)

    for modality in SIMPLE_MODALITIES:
        if (dataset_dir / modality).is_dir():
            modalities.add(modality)

    ihc_root = dataset_dir / "ihc"
    if ihc_root.exists():
        for child in ihc_root.iterdir():
            if child.is_dir() and child.name.startswith("ihc_"):
                modalities.add(child.name)

    return sorted(modalities)


def summarize_tile_parquet(parquet_path: Path) -> tuple[int, int]:
    try:
        df = pd.read_parquet(parquet_path, columns=["tissue_id"])
    except Exception:
        return 0, 0
    tissue_count = df["tissue_id"].astype(str).nunique() if "tissue_id" in df.columns else 0
    crop_count = len(df)
    return int(tissue_count), int(crop_count)


def discover_standardization_specs(modality_dir: Path) -> list[str]:
    std_dir = modality_dir / "standardization"
    if not std_dir.exists():
        return []

    specs: list[str] = []
    for parquet_path in std_dir.rglob("*.parquet"):
        rel_parent = parquet_path.parent.relative_to(std_dir)
        spec = str(rel_parent)
        if spec == ".":
            spec = parquet_path.stem
        specs.append(spec)
    return sorted(set(specs))


def parse_crop_count_map(value: object) -> dict[str, int]:
    if value is None or value is pd.NA:
        return {}
    text = str(value).strip()
    if not text or text == "-":
        return {}

    counts: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        size, count = item.split(":", 1)
        size = size.strip()
        try:
            counts[size] = int(count.strip())
        except ValueError:
            continue
    return counts


def load_inventory() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets_dir = get_datasets_dir()
    dataset_rows: list[dict[str, object]] = []
    modality_rows: list[dict[str, object]] = []
    crop_rows: list[dict[str, object]] = []
    std_rows: list[dict[str, object]] = []

    for dataset_dir in sorted(path for path in datasets_dir.iterdir() if path.is_dir()):
        metadata_path = dataset_dir / "metadata" / "tissues.parquet"
        if not metadata_path.exists():
            continue

        try:
            metadata = pd.read_parquet(metadata_path)
        except Exception:
            continue

        metadata_modalities = sorted(metadata["modality"].dropna().astype(str).unique().tolist()) if "modality" in metadata.columns else []
        loader_modalities = discover_loader_modalities(dataset_dir, metadata_modalities)
        on_disk_roots = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir() and path.name in SUPPORTED_ROOT_MODALITIES)

        dataset_std_specs = 0
        dataset_image_files = 0
        dataset_crop_counts_by_size: dict[str, int] = {}
        resolution_union: set[str] = set()
        shared_crop_specs: list[str] = []

        tiling_root = dataset_dir / "tiling"
        if tiling_root.exists():
            for resolution_dir in sorted(path for path in tiling_root.iterdir() if path.is_dir() and path.name.endswith("mpp")):
                for method_dir in sorted(path for path in resolution_dir.iterdir() if path.is_dir()):
                    for parquet_path in sorted(method_dir.glob("*_tile_coordinates.parquet")):
                        tissues_with_crops, crop_count = summarize_tile_parquet(parquet_path)
                        crop_size = parquet_path.stem.removesuffix("_tile_coordinates")
                        crop_spec = f"{resolution_dir.name}/{method_dir.name}/{crop_size}"
                        shared_crop_specs.append(crop_spec)
                        dataset_crop_counts_by_size[crop_size] = dataset_crop_counts_by_size.get(crop_size, 0) + crop_count
                        crop_rows.append(
                            {
                                "dataset": dataset_dir.name,
                                "resolution": resolution_dir.name,
                                "tiling_method": method_dir.name,
                                "crop_file": parquet_path.name,
                                "crop_size": crop_size,
                                "tissues_with_crops": tissues_with_crops,
                                "num_crops": crop_count,
                                "path": str(parquet_path),
                            }
                        )

        dataset_total_crops = sum(dataset_crop_counts_by_size.values())
        dataset_crop_specs = len(shared_crop_specs)

        for modality in loader_modalities:
            root = expected_root(modality)
            if root is None:
                continue

            modality_dir = dataset_dir / root / modality if modality.startswith("ihc_") else dataset_dir / modality
            modality_meta = metadata.loc[metadata["modality"].astype(str) == modality].copy() if "modality" in metadata.columns else pd.DataFrame()
            modality_tissues = modality_meta["tissue_id"].astype(str).tolist() if "tissue_id" in modality_meta.columns else []
            metadata_rows = len(modality_meta)
            unique_tissues = len(set(modality_tissues))

            resolution_dirs = sorted(path for path in modality_dir.iterdir() if path.is_dir() and path.name.endswith("mpp")) if modality_dir.exists() else []
            resolution_names = [path.name for path in resolution_dirs]
            resolution_union.update(resolution_names)

            image_counts: list[tuple[str, int]] = []
            for resolution_dir in resolution_dirs:
                zarr_count = len(list(resolution_dir.glob("*.zarr")))
                image_counts.append((resolution_dir.name, zarr_count))
            images_total_files = sum(count for _, count in image_counts)
            images_best_resolution = max((count for _, count in image_counts), default=0)
            dataset_image_files += images_best_resolution

            channels_path = modality_dir / "channels.parquet"
            channels_per_tissue_path = modality_dir / "channels_per_tissue.parquet"
            channel_count = pd.NA
            qc_pass_count = pd.NA
            has_nuclear_marker_col = pd.NA
            if root in MULTIPLEX_MODALITIES and channels_path.exists():
                try:
                    channels_df = pd.read_parquet(channels_path)
                    channel_count = len(channels_df)
                    if "qc_pass" in channels_df.columns:
                        qc_pass_count = int(channels_df["qc_pass"].fillna(False).astype(bool).sum())
                    has_nuclear_marker_col = "is_nuclear_marker" in channels_df.columns
                except Exception:
                    pass

            crop_specs = shared_crop_specs
            modality_crop_total = dataset_total_crops
            modality_crop_counts_by_size = dataset_crop_counts_by_size

            std_specs = discover_standardization_specs(modality_dir) if root in MULTIPLEX_MODALITIES else []
            dataset_std_specs += len(std_specs)
            for spec in std_specs:
                std_rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "modality": modality,
                        "standardization_spec": spec,
                        "path": str(modality_dir / "standardization" / spec),
                    }
                )

            modality_rows.append(
                {
                    "dataset": dataset_dir.name,
                    "modality": modality,
                    "root": root,
                    "metadata_rows": metadata_rows,
                    "unique_tissues": unique_tissues,
                    "resolutions": ", ".join(resolution_names),
                    "num_resolutions": len(resolution_names),
                    "images_best_resolution": images_best_resolution,
                    "images_total_files": images_total_files,
                    "image_counts_by_resolution": ", ".join(f"{res}:{count}" for res, count in image_counts),
                    "has_channels_parquet": channels_path.exists(),
                    "has_channels_per_tissue": channels_per_tissue_path.exists(),
                    "channel_count": channel_count,
                    "qc_pass_count": qc_pass_count,
                    "has_is_nuclear_marker": has_nuclear_marker_col,
                    "num_crop_specs": len(crop_specs),
                    "total_crops": modality_crop_total,
                    "crop_specs": ", ".join(crop_specs),
                    "crop_counts_by_size": ", ".join(f"{size}:{count}" for size, count in sorted(modality_crop_counts_by_size.items())) or "-",
                    "num_standardization_specs": len(std_specs),
                    "standardization_specs": ", ".join(std_specs),
                    "modality_path": str(modality_dir),
                }
            )

        dataset_rows.append(
            {
                "dataset": dataset_dir.name,
                "metadata_rows": len(metadata),
                "unique_metadata_tissues": metadata["tissue_id"].astype(str).nunique() if "tissue_id" in metadata.columns else 0,
                "metadata_modalities": ", ".join(metadata_modalities),
                "disk_roots": ", ".join(on_disk_roots),
                "num_modalities": len(loader_modalities),
                "loader_modalities": ", ".join(loader_modalities),
                "resolutions": ", ".join(sorted(resolution_union)),
                "images_best_resolution_total": dataset_image_files,
                "num_crop_specs": dataset_crop_specs,
                "total_crops": dataset_total_crops,
                "crop_counts_by_size": ", ".join(f"{size}:{count}" for size, count in sorted(dataset_crop_counts_by_size.items())) or "-",
                "num_standardization_specs": dataset_std_specs,
                "metadata_path": str(metadata_path),
            }
        )

    dataset_df = pd.DataFrame(dataset_rows)
    if not dataset_df.empty:
        dataset_df = dataset_df.sort_values("dataset").reset_index(drop=True)

    modality_df = pd.DataFrame(modality_rows)
    if not modality_df.empty:
        modality_df = modality_df.sort_values(["dataset", "modality"]).reset_index(drop=True)

    crop_df = pd.DataFrame(crop_rows)
    if not crop_df.empty:
        crop_df = crop_df.sort_values(["dataset", "resolution", "tiling_method", "crop_size"]).reset_index(drop=True)

    std_df = pd.DataFrame(std_rows)
    if not std_df.empty:
        std_df = std_df.sort_values(["dataset", "modality", "standardization_spec"]).reset_index(drop=True)

    return dataset_df, modality_df, crop_df, std_df


class FiltersChanged(Message):
    pass


class FilterPanel(Widget):
    DEFAULT_CSS = """
    FilterPanel {
        width: 32;
        height: 100%;
        border-right: solid $primary-darken-2;
        background: $surface-darken-1;
        overflow-y: auto;
        padding: 0 1;
    }
    FilterPanel .section-label {
        color: $warning;
        text-style: bold;
        margin-top: 1;
    }
    FilterPanel Select {
        margin-bottom: 1;
    }
    FilterPanel Input {
        margin-bottom: 1;
    }
    """

    def __init__(self, datasets: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets

    def compose(self) -> ComposeResult:
        yield Label("▸ DATASET", classes="section-label")
        yield Select([("All datasets", "__all__"), *[(dataset, dataset) for dataset in self._datasets]], value="__all__", id="dataset-select")
        yield Label("▸ SEARCH", classes="section-label")
        yield Input(placeholder="dataset / modality / resolution / spec…", id="search-input")

    @on(Select.Changed)
    @on(Input.Changed)
    def _changed(self, _: object = None) -> None:
        self.post_message(FiltersChanged())

    @property
    def selected_dataset(self) -> str | None:
        value = self.query_one("#dataset-select", Select).value
        if value in (None, "__all__"):
            return None
        return str(value)

    @property
    def search(self) -> str:
        return self.query_one("#search-input", Input).value.strip()


class SummaryView(Widget):
    DEFAULT_CSS = """
    SummaryView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    SummaryView .metrics-row {
        height: 5;
        margin-bottom: 1;
    }
    SummaryView .metric-box {
        border: solid $primary-darken-1;
        width: 1fr;
        height: 5;
        content-align: center middle;
        margin: 0 1;
        background: $surface;
    }
    SummaryView DataTable {
        height: 1fr;
    }
    """

    def __init__(self, crop_sizes: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._crop_sizes = crop_sizes

    def compose(self) -> ComposeResult:
        with Horizontal(classes="metrics-row"):
            yield Static("", id="m-datasets", classes="metric-box")
            yield Static("", id="m-modalities", classes="metric-box")
            yield Static("", id="m-images", classes="metric-box")
            yield Static("", id="m-crops", classes="metric-box")
            yield Static("", id="m-std", classes="metric-box")
            for crop_size in self._crop_sizes:
                yield Static("", id=f"m-crop-size-{crop_size}", classes="metric-box")
        yield DataTable(id="summary-table", cursor_type="row")

    def refresh_data(self, dataset_df: pd.DataFrame, modality_df: pd.DataFrame) -> None:
        def metric(value: int, label: str) -> str:
            return f"[bold white]{value}[/]\n[dim]{label}[/]"

        crop_totals_by_size = {crop_size: 0 for crop_size in self._crop_sizes}
        if not dataset_df.empty and "crop_counts_by_size" in dataset_df.columns:
            for value in dataset_df["crop_counts_by_size"]:
                for crop_size, count in parse_crop_count_map(value).items():
                    crop_totals_by_size[crop_size] = crop_totals_by_size.get(crop_size, 0) + count

        self.query_one("#m-datasets", Static).update(metric(int(len(dataset_df)), "datasets"))
        self.query_one("#m-modalities", Static).update(metric(int(len(modality_df)), "modalities"))
        self.query_one("#m-images", Static).update(metric(int(dataset_df["images_best_resolution_total"].fillna(0).sum()) if not dataset_df.empty else 0, "images"))
        self.query_one("#m-crops", Static).update(metric(int(dataset_df["num_crop_specs"].fillna(0).sum()) if not dataset_df.empty else 0, "tile files"))
        self.query_one("#m-std", Static).update(metric(int(dataset_df["num_standardization_specs"].fillna(0).sum()) if not dataset_df.empty else 0, "std specs"))
        for crop_size in self._crop_sizes:
            self.query_one(f"#m-crop-size-{crop_size}", Static).update(
                metric(crop_totals_by_size.get(crop_size, 0), f"{crop_size} crops")
            )

        table = self.query_one("#summary-table", DataTable)
        table.clear(columns=True)
        if dataset_df.empty:
            return

        columns = [
            "dataset",
            "num_modalities",
            "images_best_resolution_total",
            "num_crop_specs",
            "crop_counts_by_size",
            "num_standardization_specs",
            "resolutions",
            "loader_modalities",
        ]
        labels = [
            "Dataset",
            "Modalities",
            "Images",
            "Tile files",
            "Crop counts by size",
            "Std specs",
            "Resolutions",
            "Loader modalities",
        ]
        table.add_columns(*labels)
        for _, row in dataset_df[columns].iterrows():
            table.add_row(*[str(row[col]) for col in columns])


class ExplorerView(Widget):
    DEFAULT_CSS = """
    ExplorerView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    ExplorerView #explorer-help {
        margin-bottom: 1;
        color: $text-muted;
    }
    ExplorerView #picker-row {
        height: auto;
        margin-bottom: 1;
    }
    ExplorerView Select {
        width: 1fr;
        margin-right: 1;
    }
    ExplorerView #explorer-body {
        height: 1fr;
    }
    ExplorerView #modality-table {
        width: 3fr;
        height: 100%;
    }
    ExplorerView #dataset-side {
        width: 2fr;
        height: 100%;
        border-left: solid $primary-darken-2;
        padding: 0 1;
    }
    ExplorerView #dataset-side Static {
        height: auto;
        margin-bottom: 1;
    }
    ExplorerView DataTable {
        height: 1fr;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Inspect one dataset at a time. The left table compares modalities; the right panel shows the selected modality, including shared tile coordinate files split by resolution, method, and crop size.",
            id="explorer-help",
        )
        with Horizontal(id="picker-row"):
            yield Select([], id="dataset-picker", prompt="Select dataset…")
            yield Select([], id="modality-picker", prompt="Select modality…")
        with Horizontal(id="explorer-body"):
            yield DataTable(id="modality-table", cursor_type="row")
            with Vertical(id="dataset-side"):
                yield Label("[bold]Dataset summary[/]")
                yield Static("", id="dataset-stats")
                yield Label("[bold]Selected modality[/]")
                yield Static("", id="modality-stats")
                yield Label("[bold]Shared Tile Files[/]")
                yield DataTable(id="crop-table", cursor_type="row")
                yield Label("[bold]Standardization specs[/]")
                yield DataTable(id="std-table", cursor_type="row")

    def refresh_data(
        self,
        dataset_df: pd.DataFrame,
        modality_df: pd.DataFrame,
        crop_df: pd.DataFrame,
        std_df: pd.DataFrame,
        selected_dataset: str | None,
    ) -> None:
        self._dataset_df = dataset_df
        self._modality_df = modality_df
        self._crop_df = crop_df
        self._std_df = std_df

        datasets = dataset_df["dataset"].tolist()
        dataset_picker = self.query_one("#dataset-picker", Select)
        dataset_picker.set_options([(dataset, dataset) for dataset in datasets])

        if not datasets:
            dataset_picker.clear()
            self._set_modality_options([])
            self._render_empty_state("No datasets match the current filters.")
            return

        target_dataset = selected_dataset if selected_dataset in datasets else datasets[0]
        dataset_picker.value = target_dataset
        self._render_dataset(target_dataset)

    @on(Select.Changed, "#dataset-picker")
    def _dataset_changed(self, event: Select.Changed) -> None:
        if not is_blank_select_value(event.value):
            self._render_dataset(str(event.value))

    @on(Select.Changed, "#modality-picker")
    def _modality_changed(self, event: Select.Changed) -> None:
        dataset_picker = self.query_one("#dataset-picker", Select)
        dataset = dataset_picker.value
        if not is_blank_select_value(dataset) and not is_blank_select_value(event.value):
            self._render_modality(str(dataset), str(event.value))

    def _set_modality_options(self, modalities: list[str], preferred: str | None = None) -> str | None:
        picker = self.query_one("#modality-picker", Select)
        picker.set_options([(modality, modality) for modality in modalities])
        if not modalities:
            picker.clear()
            return None
        target = preferred if preferred in modalities else modalities[0]
        picker.value = target
        return target

    def _render_empty_state(self, message: str) -> None:
        self.query_one("#dataset-stats", Static).update(message)
        self.query_one("#modality-stats", Static).update(message)

        modality_table = self.query_one("#modality-table", DataTable)
        modality_table.clear(columns=True)
        modality_table.add_columns("Status")
        modality_table.add_row(message)

        crop_table = self.query_one("#crop-table", DataTable)
        crop_table.clear(columns=True)
        crop_table.add_columns("Status")
        crop_table.add_row(message)

        std_table = self.query_one("#std-table", DataTable)
        std_table.clear(columns=True)
        std_table.add_columns("Status")
        std_table.add_row(message)

    def _render_dataset(self, dataset: str) -> None:
        dataset_match = self._dataset_df.loc[self._dataset_df["dataset"] == dataset]
        if dataset_match.empty:
            self._set_modality_options([])
            self._render_empty_state(f"Dataset '{dataset}' is not available under the current filters.")
            return

        dataset_row = dataset_match.iloc[0]
        modality_rows = self._modality_df.loc[self._modality_df["dataset"] == dataset].copy()
        crop_rows = self._crop_df.loc[self._crop_df["dataset"] == dataset].copy() if not self._crop_df.empty else pd.DataFrame()
        std_rows = self._std_df.loc[self._std_df["dataset"] == dataset].copy() if not self._std_df.empty else pd.DataFrame()

        stats = (
            f"Metadata rows: [bold]{dataset_row['metadata_rows']}[/]\n"
            f"Unique tissue IDs: [bold]{dataset_row['unique_metadata_tissues']}[/]\n"
            f"Metadata modalities: [bold]{dataset_row['metadata_modalities'] or '-'}[/]\n"
            f"Disk roots: [bold]{dataset_row['disk_roots'] or '-'}[/]\n"
            f"Loader modalities: [bold]{dataset_row['loader_modalities'] or '-'}[/]\n"
            f"Resolutions: [bold]{dataset_row['resolutions'] or '-'}[/]\n"
            f"Images: [bold]{dataset_row['images_best_resolution_total']}[/]\n"
            f"Tile files: [bold]{dataset_row['num_crop_specs']}[/]\n"
            f"Total crops: [bold]{dataset_row['total_crops']}[/]\n"
            f"Standardization specs: [bold]{dataset_row['num_standardization_specs']}[/]\n\n"
            f"[dim]{dataset_row['metadata_path']}[/]"
        )
        self.query_one("#dataset-stats", Static).update(stats)

        modality_table = self.query_one("#modality-table", DataTable)
        modality_table.clear(columns=True)
        modality_table.add_columns(
            "Modality", "Tissues", "Images by res", "Tile files", "Tile specs", "Std specs", "Channels"
        )
        for _, row in modality_rows.iterrows():
            crop_sizes = str(row["crop_specs"]) if row["crop_specs"] else "-"
            modality_table.add_row(
                str(row["modality"]),
                str(row["unique_tissues"]),
                str(row["image_counts_by_resolution"] or "-"),
                str(row["num_crop_specs"]),
                crop_sizes,
                str(row["num_standardization_specs"]),
                str(row["channel_count"]),
            )

        modalities = modality_rows["modality"].astype(str).tolist()
        selected_modality = self._set_modality_options(modalities)
        if selected_modality is None:
            self.query_one("#modality-stats", Static).update("No loader-visible modalities for this dataset.")
            crop_table = self.query_one("#crop-table", DataTable)
            crop_table.clear(columns=True)
            crop_table.add_columns("Status")
            crop_table.add_row("No tile coordinate files found")
            std_table = self.query_one("#std-table", DataTable)
            std_table.clear(columns=True)
            std_table.add_columns("Status")
            std_table.add_row("No standardization specs found")
            return

        self._render_modality(dataset, selected_modality, crop_rows=crop_rows, std_rows=std_rows)

    def _render_modality(
        self,
        dataset: str,
        modality: str,
        *,
        crop_rows: pd.DataFrame | None = None,
        std_rows: pd.DataFrame | None = None,
    ) -> None:
        modality_match = self._modality_df.loc[
            (self._modality_df["dataset"] == dataset) & (self._modality_df["modality"] == modality)
        ]
        if modality_match.empty:
            self.query_one("#modality-stats", Static).update(
                f"Modality '{modality}' is not available for dataset '{dataset}'."
            )
            return

        row = modality_match.iloc[0]
        if crop_rows is None:
            crop_rows = self._crop_df.loc[self._crop_df["dataset"] == dataset].copy() if not self._crop_df.empty else pd.DataFrame()
        else:
            crop_rows = crop_rows.copy() if not crop_rows.empty else pd.DataFrame()
        if std_rows is None:
            std_rows = self._std_df.loc[
                (self._std_df["dataset"] == dataset) & (self._std_df["modality"] == modality)
            ].copy() if not self._std_df.empty else pd.DataFrame()
        else:
            std_rows = std_rows.loc[std_rows["modality"] == modality].copy() if not std_rows.empty else pd.DataFrame()

        modality_stats = (
            f"Modality path: [bold]{row['modality_path']}[/]\n"
            f"Root: [bold]{row['root']}[/]\n"
            f"Metadata rows: [bold]{row['metadata_rows']}[/]\n"
            f"Unique tissue IDs: [bold]{row['unique_tissues']}[/]\n"
            f"Resolutions: [bold]{row['resolutions'] or '-'}[/]\n"
            f"Images by resolution: [bold]{row['image_counts_by_resolution'] or '-'}[/]\n"
            f"Channels parquet: [bold]{row['has_channels_parquet']}[/]\n"
            f"Channels-per-tissue: [bold]{row['has_channels_per_tissue']}[/]\n"
            f"Channel count: [bold]{row['channel_count']}[/]\n"
            f"QC-pass channels: [bold]{row['qc_pass_count']}[/]\n"
            f"is_nuclear_marker column: [bold]{row['has_is_nuclear_marker']}[/]\n"
            f"Shared tile files: [bold]{row['num_crop_specs']}[/]\n"
            f"Tile specs: [bold]{row['crop_specs'] or '-'}[/]\n"
            f"Total crops: [bold]{row['total_crops']}[/]\n"
            f"Standardization specs: [bold]{row['standardization_specs'] or '-'}[/]"
        )
        self.query_one("#modality-stats", Static).update(modality_stats)

        crop_table = self.query_one("#crop-table", DataTable)
        crop_table.clear(columns=True)
        if crop_rows.empty:
            crop_table.add_columns("Status")
            crop_table.add_row("No tile coordinate files found")
        else:
            crop_table.add_columns("Resolution", "Method", "Crop size", "Tissues", "Crops", "File")
            for _, crop_row in crop_rows.iterrows():
                crop_table.add_row(
                    str(crop_row["resolution"]),
                    str(crop_row["tiling_method"]),
                    str(crop_row["crop_size"]),
                    str(crop_row["tissues_with_crops"]),
                    str(crop_row["num_crops"]),
                    str(crop_row["crop_file"]),
                )

        std_table = self.query_one("#std-table", DataTable)
        std_table.clear(columns=True)
        if std_rows.empty:
            std_table.add_columns("Status")
            std_table.add_row("No standardization specs found")
        else:
            std_table.add_columns("Spec")
            for _, std_row in std_rows.iterrows():
                std_table.add_row(str(std_row["standardization_spec"]))


class DatasetInventoryApp(App):
    TITLE = "Dataset Inventory"
    CSS = """
    Screen {
        layout: horizontal;
    }
    #main-content {
        width: 1fr;
        height: 100%;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        layout: vertical;
        width: 1fr;
        height: 1fr;
        padding: 0;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "reload", "Reload data"),
        Binding("1", "switch_tab('summary')", "Summary", show=False),
        Binding("2", "switch_tab('explorer')", "Explorer", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._dataset_df = pd.DataFrame()
        self._modality_df = pd.DataFrame()
        self._crop_df = pd.DataFrame()
        self._std_df = pd.DataFrame()
        self._filtered_dataset_df = pd.DataFrame()
        self._filtered_modality_df = pd.DataFrame()
        self._filtered_crop_df = pd.DataFrame()
        self._filtered_std_df = pd.DataFrame()
        self._crop_sizes: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

        self._dataset_df, self._modality_df, self._crop_df, self._std_df = load_inventory()
        self._crop_sizes = sorted(self._crop_df["crop_size"].astype(str).unique().tolist(), key=lambda value: (len(value), value)) if not self._crop_df.empty else []
        datasets = self._dataset_df["dataset"].tolist() if not self._dataset_df.empty else []

        with Horizontal():
            yield FilterPanel(datasets=datasets, id="filter-panel")
            with Vertical(id="main-content"):
                with TabbedContent(id="tabs"):
                    with TabPane("📊 Summary", id="summary"):
                        yield SummaryView(crop_sizes=self._crop_sizes, id="summary-view")
                    with TabPane("🔎 Explorer", id="explorer"):
                        yield ExplorerView(id="explorer-view")

    def on_mount(self) -> None:
        if self._dataset_df.empty:
            self.notify("No dataset metadata found under the configured datasets directory.", severity="error")
            return
        self._refresh_all()

    @on(FiltersChanged)
    def _filters_changed(self, _: FiltersChanged) -> None:
        self._refresh_all()

    def _refresh_all(self) -> None:
        filter_panel = self.query_one("#filter-panel", FilterPanel)
        selected_dataset = filter_panel.selected_dataset
        search = filter_panel.search.lower()

        dataset_df = self._dataset_df.copy()
        modality_df = self._modality_df.copy()
        crop_df = self._crop_df.copy()
        std_df = self._std_df.copy()

        if selected_dataset is not None:
            dataset_df = dataset_df[dataset_df["dataset"] == selected_dataset]
            modality_df = modality_df[modality_df["dataset"] == selected_dataset]
            if not crop_df.empty:
                crop_df = crop_df[crop_df["dataset"] == selected_dataset]
            if not std_df.empty:
                std_df = std_df[std_df["dataset"] == selected_dataset]

        if search:
            dataset_mask = dataset_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            dataset_df = dataset_df[dataset_mask]
            keep_datasets = set(dataset_df["dataset"].tolist())
            modality_df = modality_df[modality_df["dataset"].isin(keep_datasets)]
            if not crop_df.empty:
                crop_df = crop_df[crop_df["dataset"].isin(keep_datasets)]
            if not std_df.empty:
                std_df = std_df[std_df["dataset"].isin(keep_datasets)]

        self._filtered_dataset_df = dataset_df.reset_index(drop=True)
        self._filtered_modality_df = modality_df.reset_index(drop=True)
        self._filtered_crop_df = crop_df.reset_index(drop=True)
        self._filtered_std_df = std_df.reset_index(drop=True)

        self.query_one("#summary-view", SummaryView).refresh_data(self._filtered_dataset_df, self._filtered_modality_df)
        explorer_dataset = selected_dataset
        if explorer_dataset is None and not self._filtered_dataset_df.empty:
            explorer_dataset = str(self._filtered_dataset_df.iloc[0]["dataset"])
        self.query_one("#explorer-view", ExplorerView).refresh_data(
            self._filtered_dataset_df,
            self._filtered_modality_df,
            self._filtered_crop_df,
            self._filtered_std_df,
            explorer_dataset,
        )

    def action_reload(self) -> None:
        self.notify("Reloading data…")
        self._dataset_df, self._modality_df, self._crop_df, self._std_df = load_inventory()
        self._refresh_all()
        self.notify("Data reloaded.")

    def action_switch_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab


if __name__ == "__main__":
    DatasetInventoryApp().run()
