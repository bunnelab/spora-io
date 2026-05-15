<img src=".github/spora_io.png" alt="spora io" width="45%" style="float:left;" />
<div style="clear:both;"></div>

# Introduction

**spora [io]** is the Python loading layer for the spora ecosystem. Building upon the unified dataset of __spora [data]__ and serving as the data interface for the benchmark suite __[spora [bench]](https://github.com/bunnelab/spora-bench)__, it provides typed dataset classes and composed loaders that turn the harmonized layout into ready-to-use training and evaluation inputs across scales:

At the modality-level, dedicated dataset classes handle H&E, marker-specific IHC, and multiplex spatial proteomics (IMC, CODEX, CyCIF, MIBI), each with consistent access to images, masks, channel annotations, and standardization statistics. At the sample-level, composed multi-modal loaders align co-registered modalities for the same tissue. At the cohort-level, multi-cohort tissue and tile samplers enable balanced sampling across studies, sites, and acquisition protocols for large-scale pretraining and benchmarking.

For more information about spora, please also refer to the following ressources:

1. Our paper: To be announced
2. Our project website: To be announced
3. Documentation: To be announced

# ⚙️ Installation
To install spora [io] run
```bash
git clone https://github.com/bunnelab/spora-io.git
cd spora-io
pip install -e .
```
and set the datasets root:
```bash
export SPORA_DATASETS_DIR=/path/to/datasets_v2
```

# 🏃 Quick Start

```python
from spora_io import (
    HEImagingDataset,
    MultiplexImagingDataset,
    ComposedImagingDataset,
    SporaDataset,
)

he = HEImagingDataset(
    name="my_dataset",
    path="/path/to/datasets_v2/my_dataset",
    resolution=1.0,
    tile_size=224,
)
tissue = he.get_tissue("tissue_id")

multiplex = MultiplexImagingDataset(
    name="my_dataset",
    path="/path/to/datasets_v2/my_dataset",
    modality="imc",
    resolution=1.0,
    tile_size=224,
    standardization="quantile_clipping/uq_0.99_image",
)
tissue = multiplex.get_tissue("tissue_id", kind="uniprot_filtered")

composed = ComposedImagingDataset(
    name="my_dataset",
    path="/path/to/datasets_v2/my_dataset",
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

| Class | Use case |
| --- | --- |
| `HEImagingDataset` | Load one H&E modality from one dataset. |
| `SingleIHCImagingDataset` | Load one marker-specific IHC modality. |
| `MultiplexImagingDataset` | Load one multiplex modality with channel metadata and standardization. |
| `ComposedImagingDataset` | Load several modalities from one dataset through shared tissue IDs. |
| `SporaDataset` | Sample tissues or tiles across multiple datasets. |

# 📋 Expected Dataset Layout

```text
dataset_name/
├── metadata/
│   ├── tissues.parquet
│   └── cells.parquet                         # optional
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
│       │   └── <tissue_id>.npz               # key: mask
│       └── cell_masks/
│           └── instances/
│               └── <tissue_id>.npz           # optional
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

# 💾 Standardization Specs

Multiplex standardization stats live under:

```text
<dataset>/<modality>/<resolution>/standardization/<method>/uq_<quantile>_<quantile_level>/
```

Examples:

```text
quantile_clipping/uq_0.99_image
quantile_clipping/uq_0.99_global
quantile_clipping_log1p/uq_0.99_image
```

The suffix (`_image` or `_global`) records the quantile level used for
clipping. Means and standard deviations are computed after that clipping
transform, so they can differ between `uq_0.99_image` and `uq_0.99_global` even
with the same `stats_level`.

# 🛠️ Utility Scripts

Run scripts from the repository root.

```bash
python -m scripts.compute_tiling --dataset-name my_dataset --tile-size 224 --resolution 1.0

python -m scripts.compute_tiling \
  --dataset-name my_dataset \
  --tile-size 224 \
  --resolution 1.0 \
  --tiling-method grid_stride224 \
  --grid \
  --stride 224 \
  --tolerance 0.85

python -m scripts.compute_standardization_stats \
  --dataset-name my_dataset \
  --modality imc \
  --method quantile_clipping \
  --quantile-level image \
  --stats-level global \
  --upper-quantile 0.99 \
  --resolution 1.0
```

Useful inspection tools:

```bash
PYTHONPATH=. python scripts/marker_viz.py
PYTHONPATH=. python scripts/datasets_viz.py
PYTHONPATH=. python scripts/tile_viz.py my_dataset 224 he --mask-only
```

# 📝 Citation
*To be announced*
