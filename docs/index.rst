spora-io
========

``spora_io`` provides typed dataset loaders and utility code for spatial
proteomics datasets in the current on-disk format.

The library supports:

- H&E datasets
- single-marker IHC datasets
- multiplex datasets such as IMC, CODEX, and CycIF
- composed multi-modal loading across multiple unimodal datasets
- shared tissue masks, cell masks, tiling, and multiplex standardization stats

The current dataset format uses:

- modality-specific image roots under ``<modality>/<resolution>/images``
- shared segmentations under ``segmentations/<resolution>/...``
- shared tiling under ``tiling/<resolution>/<strategy>/...``
- multiplex standardization stats under
  ``<modality>/<resolution>/standardization/<spec>/...``

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional

   examples
   changelog
