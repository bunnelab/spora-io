# Spora Dataset Creation

This guide explains how to convert a spatial proteomics or histopathology imaging dataset from any file format into the spora data format.

In short, every image is first converted to `.zarr`, with each channel saved separately, using the tools provided in spora. The remaining components (tissue masks, the `.ome.zarr` conversion, standardization statistics and crop coordinates) are then generated with dedicated scripts.

| Step | Output | Tool |
| --- | --- | --- |
| 1 | `.zarr` images and channel metadata | Your own conversion script |
| 2 | Tissue masks | [`compute_tissue_masks.py`](scripts/compute_tissue_masks.py) |
| 3 | `.ome.zarr` images | [`convert_dataset_to_omezarr.py`](scripts/convert_dataset_to_omezarr.py) |
| 4 | Standardization statistics | [`compute_standardization_stats.py`](scripts/compute_standardization_stats.py) |
| 5 | Crop coordinates | [`compute_tiling.py`](scripts/compute_tiling.py) |
| 6 | *(Optional)* Sanity check | spora-io dataset loader |

> **Running example:** each step ends with the commands used to create the `hoefflin2026spatial` dataset.

## 1. Convert the dataset images

- Create the `.zarr` files with the conversion helpers in [`scripts/dataset_creation_utils/`](scripts/dataset_creation_utils/). Use them wherever possible, e.g., for converting NumPy arrays to `.zarr`, hashing patient IDs or creating the folder structure.
- Save the channel names and their corresponding UniProt IDs in `channels.parquet`.
- Save which channels are present in each image as a boolean mask in `channels_per_tissue.parquet`.
- See [`create_hoefflin2026spatial.py`](examples/create_hoefflin2026spatial.py) for a complete example.

**Example:**

```bash
python examples/create_hoefflin2026spatial.py
```

## 2. Create tissue masks

[`compute_tissue_masks.py`](scripts/compute_tissue_masks.py) segments the tissue from the background. It automatically picks the appropriate method: a segmentation model for IHC and H&E images, or Otsu-based background detection for spatial proteomics images.

**Example:**

```bash
python scripts/compute_tissue_masks.py --dataset_name hoefflin2026spatial
```

## 3. Convert images from `.zarr` to `.ome.zarr`

[`convert_dataset_to_omezarr.py`](scripts/convert_dataset_to_omezarr.py) adds the channel names to the image metadata and saves each image in an `.ome.zarr` v3-compliant structure. This step therefore requires that step 1 was done correctly and that the channel mapping in `channels_per_tissue.parquet` is correct.

**Example:**

```bash
python scripts/convert_dataset_to_omezarr.py --dataset_name hoefflin2026spatial
```

## 4. Compute standardization statistics (spatial proteomics only)

Pass the modality and standardization method to [`compute_standardization_stats.py`](scripts/compute_standardization_stats.py). Compute the statistics twice: once at the global level (`--stats-level global`) and once at the image level (`--stats-level image`). Note that different methods might require different standardization procedures and different quantiles.

**Example: VirTues standardization**

```bash
python -m scripts.compute_standardization_stats --dataset-name hoefflin2026spatial --modality codex \
    --method quantile_clipping_log1p --quantile-level image --stats-level global --resolution 1.0

python -m scripts.compute_standardization_stats --dataset-name hoefflin2026spatial --modality codex \
    --method quantile_clipping_log1p --quantile-level image --stats-level image --resolution 1.0
```

**Example: VirTues-M2 standardization**

```bash
python -m scripts.compute_standardization_stats --dataset-name hoefflin2026spatial --modality codex \
    --method quantile_clipping --quantile-level image --stats-level global --resolution 1.0

python -m scripts.compute_standardization_stats --dataset-name hoefflin2026spatial --modality codex \
    --method quantile_clipping --quantile-level image --stats-level image --resolution 1.0
```

## 5. Generate crop coordinates

[`compute_tiling.py`](scripts/compute_tiling.py) computes crop coordinates with a tiling algorithm optimized for maximally diverse tissue coverage.

**Example:**

```bash
python scripts/compute_tiling.py --dataset_name hoefflin2026spatial
```

## 6. *(Optional)* Check the dataset

Load the dataset with the dedicated spora-io dataset loader (e.g., in a notebook) and check that the channel mapping is correct.

---

Your dataset is now in the spora format. To use a resolution other than 1 µm/px or a different standardization approach, pass the corresponding flags (e.g., `--resolution`, `--method`) to steps 4 and 5. Run a script with `--help` to see all of its options.
