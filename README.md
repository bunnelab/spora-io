<p align="center">
  <img src=".github/spora_io.png" alt="spora io" width="45%">
</p>

# Introduction

**spora [io]** is the Python loading layer for the spora ecosystem. Building upon the unified dataset of **[spora [data]](TBD)** and serving as the data interface for the benchmark suite **[spora [bench]](https://github.com/bunnelab/spora-bench)**, it provides typed dataset classes and composed loaders that turn the harmonized data layout of spora [data] into ready-to-use training and evaluation inputs.

At the *modality-level*, dedicated dataset classes handle H&E, marker-specific IHC, and multiplex spatial proteomics (IMC, CODEX, CyCIF, MIBI), each with consistent access to images, masks, channel annotations, and standardization statistics. At the *sample-level*, composed multi-modal loaders align co-registered modalities for the same tissue. At the *cohort-level*, multi-cohort tissue and tile samplers enable balanced sampling across studies, sites, and acquisition protocols for large-scale pretraining and benchmarking.

For more information about **spora**, please also refer to the following ressources:

1. Our paper: *To be announced*
2. Our project website: *To be announced*
3. Documentation: *To be announced*

## Table of Contents
1. ⚙️ [Installation](#️-installation)
2. 🏃 [Quick Start](#-quick-start)
3. 🗂️ [Dataset Classes](#️-dataset-classes)
4. 📋 [Expected Dataset Layout](#-expected-dataset-layout)
5. 🛠️ [Utility Scripts](#️-utility-scripts)
6. ⚖️ [Licence and Terms of Use](#️-license-and-terms-of-use)
7. 📝 [Citation](#-citation)

# ⚙️ Installation
To install spora [io], please run:
```bash
git clone https://github.com/bunnelab/spora-io.git
cd spora-io
pip install -e .
```
Optionally for automatic detection of available datasets, you may set the datasets root directory via:
```bash
export SPORA_DATASETS_DIR=/path/to/datasets
```

# 🏃 Quick Start

The following minimal code example demonstrates how to instantiate four different data loading classes in order to load a single tissue or tile. Note that the example requires the environmental variable `SPORA_DATASETS_DIR` to be set, otherwise you need to specify the dataset paths explicitly when initializing the dataset classes. 

```python
from spora_io import (
    HEImagingDataset,
    MultiplexImagingDataset,
    ComposedImagingDataset,
    SporaDataset,
)

he = HEImagingDataset(
    name="my_dataset",
    resolution=1.0,
    tile_size=224,
)
tissue = he.get_tissue("tissue_id")

multiplex = MultiplexImagingDataset(
    name="my_dataset",
    modality="imc",
    resolution=1.0,
    tile_size=224,
    standardization="quantile_clipping/uq_0.99_image",
    replace_nuclear_uniprot_ids=True,
)
tissue = multiplex.get_tissue("tissue_id", kind="uniprot_filtered")

composed = ComposedImagingDataset(
    name="my_dataset",
    modalities=["he", "imc"],
    resolution=1.0,
    tile_size=224,
    split="train",
    modality_kwargs={"imc": {"standardization": "quantile_clipping/uq_0.99_image"}},
)
sample = composed.get_composed_tissue("tissue_id")

spora = SporaDataset(
    ["dataset_a", "dataset_b"],
    modalities=["he", "imc"],
    resolution=1.0,
    tile_size=224,
    sampling_unit="tiles",
    split="train",
    modality_kwargs={"imc": {"standardization": "identity"}},
)
tile_sample = spora.sample_random_tile()
```

# 🗂️ Dataset Classes

In the table below, we list the core data loading classes of spora [io] as well as their intended use cases. For more information about the available functionality of each class, please refer to the [documentation](TBD).

| Class | Use Case |
| --- | --- |
| `HEImagingDataset` | Access a single H&E dataset and load H&E data  |
| `SingleIHCImagingDataset` | Access a single marker-specific IHC dataset and load single-marker IHC data. |
| `MultiplexImagingDataset` | Access a single multiplexed imaging dataset and load multiplex data including channel metadata and standardization. |
| `ComposedImagingDataset` | Access a single multi-modal dataset (H&E, IHC, SP) and load aligned or un-aligned multi-modal tissue data through shared tissue IDs. |
| `SporaDataset` | Sample tissues or tiles across multiple uni- or multi-modal datasets. |

# 📋 Expected Dataset Layout

spora [io] requires all datasets to be stored in the standardized dataset format of spora [data], that we outline below. For more information about this dataset format, please refer to the [documentation](TBD).

```text
dataset_name/
├── metadata/
│   ├── tissues.parquet
│   └── cells.parquet      # optional
├── he/
│   └── 1_0mpp/
│       └── images/
│           └── <tissue_id>.ome.zarr
├── imc/ | codex/ | cycif/ | mibi/
│   ├── channels.parquet
│   ├── channels_per_tissue.parquet
│   └── 1_0mpp/
│       ├── images/
│       │   └── <tissue_id>.ome.zarr
│       └── standardization/
│           ├── quantile_clipping/
│           │   └── uq_0.99_image/
│           └── quantile_clipping_log1p/
│               └── uq_0.99_image/
├── ihc/
│   └── ihc_<marker>/
│       └── 1_0mpp/
│           └── images/
│               └── <tissue_id>.ome.zarr
├── segmentations/
│   └── 1_0mpp/
│       ├── tissue_masks/
│       │   └── <tissue_id>.npz      # key: mask
│       └── cell_masks/
│           └── instances/
│               └── <tissue_id>.npz      # optional
└── tiling/
    └── 1_0mpp/
        └── default/
            ├── 224_tile_coordinates.parquet
            └── 224_tile_stats.parquet
```

`tissues.parquet` is the metadata source of truth. It should expose
`tissue_id` either as a column or as the index and should include modality
information. `channels_per_tissue.parquet` is indexed by `tissue_id` and stores
per-channel availability for multiplex images.

## Standardization Statistics

For multiplexed imaging data, spora [io] provides various standardization pipelines to preprocess images.

Dataset- und image-level statistics precomputed and used for these pipelines are stored at:

```text
<dataset>/<modality>/<resolution>/standardization/<method>/uq_<quantile>_<quantile_level>/
```

See for example:

```text
quantile_clipping/uq_0.99_image
quantile_clipping/uq_0.99_global
quantile_clipping_log1p/uq_0.99_image
```

The suffix (`_image` or `_global`) records the quantile level used for
clipping. Means and standard deviations are computed after that clipping
transform, so they can differ between `uq_0.99_image` and `uq_0.99_global` even
with the same `stats_level`.

# 🛠️ Utility scripts

With spora [io], we provide multiple utility scripts to facilitate the setup of a new dataset following the structure of spora [data]:
1. `scripts.compute_tissue_masks` to compute tissue masks.
2. `scripts.compute_tiling` to compute tilings of the images.
3. `scripts.compute_standardization_stats` to compute dataset and image-level statistics (means, standard deviations and quantiles) required for image preprocessing.

These scripts should be executed from the repository root. Please find below examples of how to execute these scripts:

```bash
# Computes tissue masks
python -m scripts.compute_tissue_masks --dataset_name my_dataset

# Default tiling method optimizing coverage.
python -m scripts.compute_tiling --dataset-name my_dataset --tile-size 224 --resolution 1.0

# Strided grid tiling within tissue area
python -m scripts.compute_tiling \
  --dataset-name my_dataset \
  --tile-size 224 \
  --resolution 1.0 \
  --tiling-method grid_stride224 \
  --grid \
  --stride 224 \
  --tolerance 0.85

# Computes image-level quantiles and dataset-level statistics (mean and standard deviation) after quantile clipping.
python -m scripts.compute_standardization_stats \
  --dataset-name my_dataset \
  --modality imc \
  --method quantile_clipping \
  --quantile-level image \
  --stats-level global \
  --upper-quantile 0.99 \
  --resolution 1.0
```

Further we provide useful CLI inspection tools to inspect the available spora datasets:

```bash
PYTHONPATH=. python scripts/marker_viz.py
PYTHONPATH=. python scripts/datasets_viz.py
PYTHONPATH=. python scripts/tile_viz.py my_dataset 224 he --mask-only
```

# ⚖️ License and Terms of Use
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Laboratory of Artificial Intelligence in Molecular Medicine, 2026

This repository and associated code are released under the MIT Licence. See LICENSE.md for details.

**Note:**  This licence does not cover the datasets themselves. Please refer to the dataset root directories for more information about the licence of each dataset.

# 📝 Citation
*To be announced*
