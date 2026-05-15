spora-io
========

``spora_io`` is the data-loading layer for the current SPORA spatial
proteomics dataset format. It turns curated on-disk datasets into typed Python
objects for model training, visualization, and benchmarking.

The library is built around a few explicit contracts:

- **Modality-specific loaders** for H&E, single-marker IHC, and multiplex
  proteomics images such as IMC, CODEX, CyCIF, and MIBI.
- **Composed multimodal loading** for tissues that have aligned views across
  multiple modalities.
- **Multi-cohort sampling** through ``SporaDataset``, which builds a global
  tissue or tile index across several dataset folders.
- **Shared segmentation and tiling** from
  ``segmentations/<resolution>/...`` and
  ``tiling/<resolution>/<strategy>/...``.
- **Stats-backed multiplex standardization** from parquet files under
  ``<modality>/<resolution>/standardization/<spec>/...``.

Minimal Example
---------------

.. code-block:: python

   from spora_io import SporaDataset

   dataset = SporaDataset(
       ["schurch2020coordinated", "lin2022multiplexed"],
       modalities=["codex", "imc"],
       resolution=1.0,
       tile_size=224,
       sampling_unit="tiles",
       split="train",
       modality_kwargs={
           "codex": {"standardization": "quantile_clipping/uq_0.99_image"},
           "imc": {"standardization": "quantile_clipping/uq_0.99_image"},
       },
   )

   sample = dataset.sample_random_tile()
   sample["dataset_name"]
   sample["tissue_id"]
   sample["tile_id"]
   sample["modalities"]

When To Use Which Class
-----------------------

.. list-table::
   :header-rows: 1

   * - Class
     - Use case
   * - ``HEImagingDataset``
     - Load one H&E modality from one dataset.
   * - ``SingleIHCImagingDataset``
     - Load one marker-specific IHC modality from one dataset.
   * - ``MultiplexImagingDataset``
     - Load one multiplex modality with channel metadata and standardization.
   * - ``ComposedImagingDataset``
     - Load multiple modalities from one dataset using shared tissue IDs.
   * - ``SporaDataset``
     - Sample tissues or tiles across multiple datasets.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   add_dataset
   tools
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
