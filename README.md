# spatialprot-data

Data structures and loaders for spatial proteomics datasets. Provides a unified interface for loading and working with multi-modal imaging data including H&E, immunohistochemistry (IHC), as well as spatial proteomics technologies such as IMC, CODEX, Orion, and CycIF.

## Installation

```bash
pip install spatialprot-data
```

For development:

```bash
git clone https://github.com/bunnelab/spatialprot-data.git
cd spatialprot-data
pip install -e ".[dev]"
```

## Quick Start

```python
from spora_io import HEImagingDataset, MultiplexImagingDataset, ComposedImagingDataset

# Load an H&E dataset
he_dataset = HEImagingDataset(
    name="my_dataset",
    path="/path/to/dataset",
    resolution=1.0,
    crop_size=224,
)
tissue = he_dataset.get_tissue("tissue_id")

# Load a multiplex (IMC/CODEX/CycIF) dataset
mp_dataset = MultiplexImagingDataset(
    name="my_dataset",
    path="/path/to/dataset",
    modality="imc",
    normalization="q99_clipping",
    resolution=1.0,
    crop_size=224,
)
tissue = mp_dataset.get_tissue("tissue_id", kind="filtered")

# Compose multiple modalities into a single dataset
composed = ComposedImagingDataset(
    name="my_dataset",
    path="/path/to/dataset",
    modalities=["he", "imc"],
    resolution=1.0,
    crop_size=224,
    modality_kwargs={"imc": {"normalization": "q99_clipping"}},
)
composed_tissue = composed.get_composed_tissue("tissue_id")
```

## Supported Modalities

| Modality | Dataset Class | Description |
|----------|--------------|-------------|
| H&E | `HEImagingDataset` | Hematoxylin & Eosin stained images |
| IHC | `SingleIHCImagingDataset` | Single-marker immunohistochemistry |
| IMC | `MultiplexImagingDataset` | Imaging Mass Cytometry |
| CODEX | `MultiplexImagingDataset` | CO-Detection by indEXing |
| CycIF | `MultiplexImagingDataset` | Cyclic Immunofluorescence |
| Multi-modal | `ComposedImagingDataset` | Combines any of the above |

## Expected Dataset Structure

Each dataset must follow this directory layout:

```
dataset_name/
├── metadata/
│   ├── tissues.parquet          # Required
│   └── cells.parquet            # Optional
├── he/                          # H&E images
│   └── {resolution}mpp/
│       └── {tissue_id}.zarr     # Shape: (H, W, 3), dtype: uint8
├── cycif/ | imc/ | codex/       # Multiplex images
│   ├── {resolution}mpp/
│   │   └── {tissue_id}.zarr     # Shape: (C, H, W), dtype: float32
│   ├── channels.parquet         # Channel metadata
│   └── channels_per_tissue.parquet
├── ihc/                         # IHC images
│   └── ihc_{marker}/
│       └── {resolution}mpp/
│           └── {tissue_id}.zarr # Shape: (H, W, 3), dtype: uint8
└── segmentations/
    └── {modality}/
        └── tissue_masks/
            └── {resolution}mpp/
                └── {tissue_id}.npz  # Key: "mask", dtype: bool
```

### Metadata Files

**`tissues.parquet`** (required) — one row per (tissue, modality) pair:

| Column | Type | Description |
|--------|------|-------------|
| `tissue_id` | str | Unique tissue identifier, format: `{name}_{8char}_{4digit}` |
| `patient_id` | str | Patient identifier |
| `modality` | str | One of: `he`, `imc`, `codex`, `cycif`, `ihc_{marker}` |
| `alignment` | str | Alignment group for cross-modal registration |

**`channels.parquet`** (required per multiplex modality):

| Column | Type | Description |
|--------|------|-------------|
| `channel_name` | str | Unique channel/marker name |
| `qc_pass` | bool | Whether the channel passes quality control |
| `uniprot_id` | str | UniProt protein identifier for marker embeddings |
| `is_nuclear_marker` | bool | Indicates nucleus segmentation markers |

**`channels_per_tissue.parquet`** — boolean matrix indicating channel availability per tissue. Index: `tissue_id`. Columns: One column per channel name following order of `channels.parquet`.

**`cells.parquet`** (optional) — cell-level annotations with at least `tissue_id` and `cell_id` columns.

## Configuration

Set the `SPATIALPROT_DATASETS_DIR` environment variable to point to your datasets root directory:

```bash
export SPATIALPROT_DATASETS_DIR=/path/to/datasets
```

## Data Preparation Scripts

The `scripts/` directory contains utilities for preparing dataset files:

- `compute_tiling.py` - Generate optimal tile coordinates for tissue images
- `compute_normalization_stats.py` - Compute normalization statistics (quantiles, means, stds)

## Documentation

Build the API docs locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```
