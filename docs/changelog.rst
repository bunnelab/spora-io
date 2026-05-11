Changelog
=========

Current Snapshot
----------------

This documentation reflects the current codebase state rather than a published
release tag.

- Dataset classes for H&E, IHC, IMC, CODEX, CyCIF, and MIBI modalities
- ``ComposedImagingDataset`` for multi-modal loading
- ``SporaDataset`` for multi-cohort tissue and tile sampling
- Stats-backed multiplex standardization framework
- Greedy tiling algorithm with adaptive stopping
- Shared tissue masks and cell masks under ``segmentations/<resolution>/``
- Shared parquet-backed tiling under ``tiling/<resolution>/<strategy>/``
- Label filtering by metadata columns
