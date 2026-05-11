Examples
========

Common Workflows
----------------

The examples directory can be used as a starting point for common workflows:

- loading H&E and multiplex datasets from the current shared dataset layout
- visualising tissue images and multiplex marker composites
- retrieving tiles from parquet-backed tiling coordinates
- inspecting ``channels.parquet`` and ``channels_per_tissue.parquet``
- using ``ComposedImagingDataset`` for aligned multi-modal access
- using ``SporaDataset`` for multi-cohort sampling
- working with shared tissue masks and cell masks

Minimal Multiplex Example
-------------------------

.. code-block:: python

   from spora_io import MultiplexImagingDataset

   ds = MultiplexImagingDataset(
       name="schurch2020coordinated",
       path="/mnt/aimm/scratch/datasets_v2/schurch2020coordinated",
       modality="codex",
       standardization="quantile_clipping/uq_0.99_image",
       resolution=1.0,
       tile_size=224,
   )

   tissue_id = ds.get_tissue_ids()[0]
   tissue = ds.get_tissue(tissue_id, kind="uniprot_filtered", preprocess=True)
   tile = ds.get_tile(tissue_id, tile_id=0, kind="complete", preprocess=False)

Multi-cohort Tile Sampling
--------------------------

.. code-block:: python

   from spora_io import SporaDataset

   ds = SporaDataset(
       ["schurch2020coordinated", "lin2022multiplexed"],
       modalities=["codex", "imc"],
       resolution=1.0,
       tile_size=224,
       sampling_unit="tiles",
       modality_kwargs={
           "codex": {"standardization": "quantile_clipping/uq_0.99_image"},
           "imc": {"standardization": "quantile_clipping/uq_0.99_image"},
       },
   )

   sample = ds.sample_random_tile()
   sample["dataset_name"]
   sample["tissue_id"]
   sample["tile_id"]
   sample["modalities"]

Inspecting Shared Tiling
------------------------

The current tiling format is dataset-level rather than modality-level:

.. code-block:: text

   dataset/tiling/1_0mpp/default/224_tile_coordinates.parquet

Each row stores one tile:

.. code-block:: text

   tissue_id | tile_id | row | col

Inspecting Standardization Stats
--------------------------------

Multiplex standardization stats live under the modality resolution directory:

.. code-block:: text

   dataset/codex/1_0mpp/standardization/quantile_clipping/uq_0.99_image/

Typical files include:

- ``image_level_upper_quantiles.parquet``
- ``global_level_means.parquet``
- ``global_level_stds.parquet``
