Concepts
========

This page explains the key abstractions in ``spatialprot-data``.

Dataset Hierarchy
-----------------

All dataset classes inherit from :class:`~spatialprot_data.BaseImagingDataset`,
which handles tissue metadata, label filtering, tissue masks, and cell masks.

- :class:`~spatialprot_data.HEImagingDataset` -- RGB H&E stained images stored
  as ``(H, W, 3)`` Zarr arrays, returned as ``(3, H, W)`` tensors.
- :class:`~spatialprot_data.MultiplexImagingDataset` -- multi-channel images
  (IMC, CODEX, CycIF) stored as ``(C, H, W)`` Zarr arrays.
- ``SingleIHCImagingDataset`` -- single-marker IHC images (RGB, like H&E).
- :class:`~spatialprot_data.ComposedImagingDataset` -- wraps multiple unimodal
  datasets into a single interface, enabling joint loading across modalities.

Modality Types
--------------

Each imaging modality is represented by a dataclass (e.g.
:class:`~spatialprot_data.HEModality`, :class:`~spatialprot_data.CycIFModality`)
that stores the modality name and the canonical directory name used on disk.

Tissue and Data Types
---------------------

Loading a tissue returns a typed dataclass:

- ``HETissue`` -- contains a ``torch.Tensor`` of shape ``(3, H, W)`` or
  ``(H, W, 3)`` depending on ``image_mode``.
- ``MultiplexTissue`` -- tensor of shape ``(C, H, W)`` plus ``channel_names``,
  ``measured_mask``, ``image_loading_mask``, and ``channel_idxs``.
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
3. **filtered** -- further restricted to channels that have protein embeddings
   (e.g. ESM) available, enabling downstream embedding-based analysis.

Pass ``kind="complete"``, ``kind="qc_filtered"``, or ``kind="filtered"`` to
:meth:`~spatialprot_data.MultiplexImagingDataset.get_tissue`.

Normalization
-------------

Normalization is configured via the ``normalization`` argument when constructing a
:class:`~spatialprot_data.MultiplexImagingDataset`. Available strategies:

- ``"identity"`` -- no transformation (just tensor conversion).
- ``"q99_clipping"`` -- clip to per-image 99th percentile, then scale to [0, 1].

The factory function
:func:`~spatialprot_data.utils.dataset.normalize.build_normalizer` resolves the
string to the appropriate normalizer class.

For H&E images, normalization uses ImageNet or HIBOU mean/std presets
(controlled by the ``mean_std_type`` argument).

Tiling
------

The function
:func:`~spatialprot_data.utils.helpers.crop.best_mask_tiling_try_to_stop`
computes an optimised set of tile coordinates from a binary tissue mask using a
greedy algorithm with adaptive stopping. It maximises coverage of valid tissue
while avoiding excessive overlap.

Two criteria must *both* be met to stop early:

1. Coverage has reached ``coverage_goal``.
2. The best remaining tile contributes fewer than ``min_gain_ratio`` new pixels.

Resolution Convention
---------------------

Resolutions are expressed in microns per pixel (MPP) and encoded as directory
names with underscores replacing dots: ``1_0mpp`` for 1.0 MPP, ``0_65mpp`` for
0.65 MPP. Pass a ``float`` (e.g. ``1.0``) or the string directly.
