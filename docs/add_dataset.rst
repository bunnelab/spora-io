Adding a Dataset
================

This page describes the expected SPORA dataset layout, the commands needed to
generate tile coordinates and multiplex standardization statistics, and the
extension point for custom standardization classes.

Dataset Layout
--------------

Each dataset lives under ``SPORA_DATASETS_DIR``:

.. code-block:: text

   <datasets_dir>/
     <dataset_name>/
       metadata/
         tissues.parquet
         cells.parquet                  # optional
       segmentations/
         1_0mpp/
           tissue_masks/
             <tissue_id>.npz
           cell_masks/
             instances/
               <tissue_id>.npz          # optional
       tiling/
         1_0mpp/
           default/
             224_tile_coordinates.parquet
             224_tile_stats.parquet
             128_tile_coordinates.parquet
             128_tile_stats.parquet
       he/
         1_0mpp/
           images/
             <tissue_id>.ome.zarr
       imc/
         channels.parquet
         channels_per_tissue.parquet
         1_0mpp/
           images/
             <tissue_id>.ome.zarr
           standardization/
             quantile_clipping/
               uq_0.99_image/
                 image_level_upper_quantiles.parquet
                 global_level_means.parquet
                 global_level_stds.parquet
       ihc/
         ihc_<marker>/
           1_0mpp/
             images/
               <tissue_id>.ome.zarr

The multiplex modality directory can be ``imc``, ``codex``, ``cycif``, or
``mibi``. IHC is marker-specific, so each marker is stored as
``ihc/ihc_<marker>``.

Metadata
--------

``metadata/tissues.parquet`` is the source of truth for image-level metadata.
It should contain one row per image or tissue-modality entry. The required
identifier is ``tissue_id``. If the table has ``tissue_id`` as its index, the
loaders use that index directly.

Recommended columns are:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``tissue_id``
     - Unique image identifier, either as a column or index.
   * - ``dataset``
     - Dataset or cohort name.
   * - ``patient_id``
     - Harmonized patient identifier.
   * - ``specimen_id``
     - Harmonized specimen or slide identifier.
   * - ``modality``
     - Image modality, such as ``he``, ``imc``, ``codex``, ``cycif``, ``mibi``, or ``ihc_<marker>``.
   * - ``split``
     - Optional split label used by dataset constructors that accept ``split``.

For multiplex modalities, add channel metadata:

.. list-table::
   :header-rows: 1

   * - File
     - Required content
   * - ``<modality>/channels.parquet``
     - Cohort-level channel table with marker names, channel indices, optional UniProt IDs, QC flags, and nuclear-marker flags.
   * - ``<modality>/channels_per_tissue.parquet``
     - Tissue-by-channel availability table. ``tissue_id`` should be the index.

Useful channel columns are ``channel_name``, ``index``, ``uniprot_id``,
``qc_pass``, and ``is_nuclear_marker``. Nuclear stains such as DAPI may have
missing UniProt IDs, but should be flagged with ``is_nuclear_marker`` when
available.

Images And Masks
----------------

Images are stored as OME-Zarr directories under the modality's
``<resolution>/images`` directory. The current loaders read the image scale from
the OME-Zarr store and return channel-first tensors.

Tissue masks are shared across modalities:

.. code-block:: text

   <dataset_name>/segmentations/1_0mpp/tissue_masks/<tissue_id>.npz

The ``.npz`` file must contain a ``mask`` array. Its height and width should
match the image at the same resolution. Cell instance masks are optional and
use the same shared ``segmentations`` tree.

Validate Basic Loading
----------------------

Before computing derived files, validate that the dataset can be opened:

.. code-block:: python

   from spora_io.datasets import HEImagingDataset, MultiplexImagingDataset

   he = HEImagingDataset(
       name="my_dataset",
       path="/path/to/my_dataset",
       resolution=1.0,
       tile_size=None,
   )
   print(len(he.get_tissue_ids()))

   imc = MultiplexImagingDataset(
       name="my_dataset",
       path="/path/to/my_dataset",
       modality="imc",
       resolution=1.0,
       tile_size=None,
       standardization="identity",
   )
   tissue = imc.get_tissue(imc.get_tissue_ids()[0], preprocess=False)
   print(tissue.image.shape)

Generate Tiling
---------------

Tile coordinates are generated from the shared tissue masks. They are saved as
parquet files with one row per tile:

.. code-block:: text

   tissue_id | tile_id | row | col

Run tiling once per tile size:

.. code-block:: bash

    python -m scripts.compute_tiling \
     --dataset-name my_dataset \
     --resolution 1.0 \
     --tiling-method default \
     --tile-size 224 \
     --stride 112 \
     --tolerance 0.85 \
     --coverage-goal 1.0 \
     --min-gain-ratio 0.05

    python -m scripts.compute_tiling \
     --dataset-name my_dataset \
     --resolution 1.0 \
     --tiling-method default \
     --tile-size 128 \
     --stride 64 \
     --tolerance 0.85 \
     --coverage-goal 1.0 \
     --min-gain-ratio 0.05

Outputs are written to:

.. code-block:: text

   <dataset_name>/tiling/1_0mpp/default/224_tile_coordinates.parquet
   <dataset_name>/tiling/1_0mpp/default/224_tile_stats.parquet

If both output files already exist, the script skips computation unless
``--overwrite`` is passed.

Generate Standardization Statistics
-----------------------------------

Standardization statistics are computed per multiplex modality. The current
supported methods are ``quantile_clipping`` and ``quantile_clipping_log1p``.

Typical commands:

.. code-block:: bash

  python -m scripts.compute_standardization_stats \
     --dataset-name my_dataset \
     --modality imc \
     --method quantile_clipping \
     --quantile-level image \
     --stats-level global \
     --upper-quantile 0.99 \
     --resolution 1.0

  python -m scripts.compute_standardization_stats \
     --dataset-name my_dataset \
     --modality imc \
     --method quantile_clipping_log1p \
     --quantile-level image \
     --stats-level global \
     --upper-quantile 0.99 \
     --resolution 1.0

The first command writes files under:

.. code-block:: text

   <dataset_name>/imc/1_0mpp/standardization/quantile_clipping/uq_0.99_image/

The ``_image`` suffix comes from ``--quantile-level image``. If
``--quantile-level global`` is used, the directory is ``uq_0.99_global``.
This suffix is required because means and standard deviations are computed
after clipping and therefore depend on the quantile level.

Load the generated statistics by passing the same spec to the dataset:

.. code-block:: python

   ds = MultiplexImagingDataset(
       name="my_dataset",
       path="/path/to/my_dataset",
       modality="imc",
       resolution=1.0,
       tile_size=224,
       standardization="quantile_clipping/uq_0.99_image",
       quantile_level="image",
       stats_level="global",
   )

Implement A Custom Standardizer
-------------------------------

Custom standardization classes live in
``spora_io/utils/dataset/standardize.py``.

Use ``BaseStandardizer`` if the transform does not need saved parquet
statistics. Use ``StatsBackedStandardizer`` if the transform should load
quantiles, means, or standard deviations from
``<modality>/<resolution>/standardization/<spec>``.

Example stats-backed standardizer:

.. code-block:: python

   class SqrtQuantileClippingStandardizer(StatsBackedStandardizer):
       def _transform(
           self,
           x_t: torch.Tensor,
           upper_t: torch.Tensor,
           lower_t: torch.Tensor,
       ) -> torch.Tensor:
           x_t = torch.clamp(x_t, min=lower_t, max=upper_t)
           x_t = (x_t - lower_t) / (upper_t - lower_t + 1e-8)
           return torch.sqrt(torch.clamp(x_t, min=0.0))

Register the class in ``build_standardizer``:

.. code-block:: python

   if method == "quantile_clipping_sqrt":
       return SqrtQuantileClippingStandardizer(
           spec=spec,
           use_mean_std=use_mean_std,
           **kwargs_common,
       )

If the method needs the standard statistics script to generate files, also add
the method name to ``VALID_METHODS`` in
``scripts/compute_standardization_stats.py``. The directory name written by the
script must match the method name used in ``build_standardizer``.

After registering the class, test it by loading one tissue manually:

.. code-block:: python

   ds = MultiplexImagingDataset(
       name="my_dataset",
       path="/path/to/my_dataset",
       modality="imc",
       resolution=1.0,
       tile_size=None,
       standardization="quantile_clipping_sqrt/uq_0.99_image",
   )
   tissue = ds.get_tissue(ds.get_tissue_ids()[0])
   print(tissue.image.shape)

Final Checklist
---------------

For a new dataset, verify the following before using it in training:

- ``metadata/tissues.parquet`` loads and exposes the expected tissue IDs.
- Every image has a matching tissue mask at the same resolution.
- Multiplex modalities have ``channels.parquet`` and
  ``channels_per_tissue.parquet``.
- ``224_tile_coordinates.parquet`` and ``128_tile_coordinates.parquet`` exist
  under ``tiling/1_0mpp/default``.
- ``quantile_clipping/uq_0.99_image`` and
  ``quantile_clipping_log1p/uq_0.99_image`` exist
  for every multiplex modality that will be standardized.
- A small ``HEImagingDataset``, ``MultiplexImagingDataset``, or
  ``SporaDataset`` instance can load one tissue or tile without errors.
