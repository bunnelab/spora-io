Changelog
=========

Current Snapshot
----------------

This documentation reflects the current codebase state rather than a published
release tag.

- Dataset classes for H&E, IHC, IMC, CODEX, and CycIF modalities
- ``ComposedImagingDataset`` for multi-modal loading
- Stats-backed multiplex standardization framework
- Channel selection transforms for data augmentation
- Greedy tiling algorithm with adaptive stopping
- Shared tissue masks and cell masks under ``segmentations/<resolution>/``
- Shared parquet-backed tiling under ``tiling/<resolution>/<strategy>/``
- Label filtering by metadata columns
