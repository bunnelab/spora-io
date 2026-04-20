Concepts
========

This page explains the current abstractions and on-disk layout used by
``spora_io``.

Dataset Format
--------------

The current dataset format separates modality-specific image data from shared
segmentations and shared tiling:

.. code-block:: text

   dataset/
   ├── metadata/
   │   ├── tissues.parquet
   │   └── cells.parquet                # optional
   ├── he/
   │   └── 1_0mpp/
   │       └── images/
   ├── imc/
   │   ├── channels.parquet
   │   ├── channels_per_tissue.parquet
   │   └── 1_0mpp/
   │       ├── images/
   │       └── standardization/
   │           └── quantile_clipping/
   │               └── uq_0.99/
   ├── ihc/
   │   └── ihc_CD3/
   │       └── 1_0mpp/
   │           └── images/
   ├── segmentations/
   │   └── 1_0mpp/
   │       ├── tissue_masks/
   │       └── cell_masks/
   └── tiling/
       └── 1_0mpp/
           └── default/
               ├── 224_tile_coordinates.parquet
               └── 224_tile_stats.parquet

Dataset Hierarchy
-----------------

All unimodal dataset classes inherit from
:class:`~spora_io.datasets.base.BaseImagingDataset`, which handles metadata,
shared tissue masks, shared cell masks, and tile coordinate loading.

- :class:`~spora_io.datasets.he.HEImagingDataset`
  loads RGB H&E images.
- :class:`~spora_io.datasets.multiplex.MultiplexImagingDataset`
  loads multiplex images and aligned channel metadata.
- :class:`~spora_io.datasets.ihc.SingleIHCImagingDataset`
  loads single-marker RGB IHC images.
- :class:`~spora_io.datasets.compose.ComposedImagingDataset`
  wraps several unimodal datasets into a single multi-modal handle.

Modality Types
--------------

Each imaging modality is represented by a dataclass such as
:class:`~spora_io.datasets._types.HEModality` or
:class:`~spora_io.datasets._types.CycIFModality`. These store the modality name
and canonical directory used on disk.

Tissue and Data Types
---------------------

Loading a tissue returns a typed dataclass:

- ``HETissue`` -- contains a ``torch.Tensor`` of shape ``(3, H, W)`` or
  ``(H, W, 3)`` depending on ``image_mode``.
- ``MultiplexTissue`` -- tensor of shape ``(C, H, W)`` plus ``channel_names``,
  ``measured_mask``, ``image_loading_mask``, and optional ``uniprot_ids``.
- ``IHCTissue`` -- RGB IHC tissue image plus the marker name.
- ``ComposedTissue`` -- a dictionary mapping modality names to their respective
  tissue objects.
- ``TissueMask`` / ``CellMask`` -- binary or integer segmentation masks.

Channel Filtering Pipeline
--------------------------

For multiplex datasets, channels are progressively filtered:

1. **complete** -- all channels measured for the tissue (from
   ``channels_per_tissue.parquet``).
2. **qc_filtered** -- only channels that pass quality control (``qc_pass``
   column in ``channels.parquet``).
3. **uniprot_filtered** -- further restricted to channels with valid UniProt
   IDs, enabling aligned downstream protein-centric analysis.

Pass ``kind="complete"``, ``kind="qc_filtered"``, or
``kind="uniprot_filtered"`` to
:meth:`~spora_io.datasets.multiplex.MultiplexImagingDataset.get_tissue`.

Standardization
---------------

Multiplex preprocessing is configured via the ``standardization`` argument when
constructing :class:`~spora_io.datasets.multiplex.MultiplexImagingDataset`.

The active implementation is in
:mod:`spora_io.utils.dataset.standardize`. The standardizer reads parquet stats
from:

.. code-block:: text

   <modality>/<resolution>/standardization/<spec>/

Typical specs are:

- ``identity`` -- no transform other than tensor conversion
- ``quantile_clipping/uq_0.99``
- ``quantile_clipping_log1p/uq_0.99``

The factory function
:func:`~spora_io.utils.dataset.standardize.build_standardizer` resolves the
requested spec to the appropriate standardizer class.

For H&E images, normalization uses ImageNet or HIBOU mean/std presets
(controlled by the ``mean_std_type`` argument).

Tiling
------

The function
:func:`~spora_io.utils.helpers.crop.best_mask_tiling_try_to_stop`
computes an optimised set of tile coordinates from a binary tissue mask. The
result is typically persisted as a parquet file with one row per tile:

- ``tissue_id``
- ``crop_id``
- ``row``
- ``col``

Two criteria must *both* be met to stop early:

1. Coverage has reached ``coverage_goal``.
2. The best remaining tile contributes fewer than ``min_gain_ratio`` new pixels.

Resolution Convention
---------------------

Resolutions are expressed in microns per pixel (MPP) and encoded as directory
names with underscores replacing dots: ``1_0mpp`` for 1.0 MPP, ``0_65mpp`` for
0.65 MPP. Pass a ``float`` (e.g. ``1.0``) or the string directly.
