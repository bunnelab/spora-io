Changelog
=========

v0.1.0 (2025)
--------------

Initial release.

- Dataset classes for H&E, IHC, IMC, CODEX, and CycIF modalities
- ``ComposedImagingDataset`` for multi-modal loading
- Normalization framework (identity, Q99 clipping)
- Channel selection transforms for data augmentation
- Greedy tiling algorithm with adaptive stopping
- ESM protein embedding integration for marker filtering
- Tissue and cell mask support
- Label filtering by metadata columns
