Quick Start
===========

Loading an H&E dataset
----------------------

.. code-block:: python

   from spora_io import HEImagingDataset

   dataset = HEImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       resolution=1.0,
       tile_size=224,
   )

   tissue_ids = dataset.get_tissue_ids()
   tissue = dataset.get_tissue(tissue_ids[0])
   # tissue.image is a torch.Tensor of shape (3, H, W)

Loading a multiplex dataset
----------------------------

.. code-block:: python

   from spora_io import MultiplexImagingDataset

   dataset = MultiplexImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       modality="cycif",            # or "imc", "codex", "mibi" etc.
       standardization="identity",  # or "quantile_clipping/uq_0.99_image"
       resolution=1.0,
       tile_size=224,
   )

   tissue_ids = dataset.get_tissue_ids()
   tissue = dataset.get_tissue(tissue_ids[0], kind="uniprot_filtered")
   # tissue.image is a torch.Tensor of shape (C, H, W)
   # tissue.channel_names contains the marker names
   # tissue.uniprot_ids contains the aligned UniProt IDs when available

Composing multiple modalities
------------------------------

.. code-block:: python

   from spora_io import ComposedImagingDataset

   dataset = ComposedImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       modalities=["he", "cycif"],
       resolution=1.0,
       tile_size=224,
       split="train",  # optional; filters tissues.parquet["split"]
       modality_kwargs={"cycif": {"standardization": "identity"}},
   )

   composed = dataset.get_composed_tissue(tissue_ids[0])
   # composed.modalities["he"] -> HETissue
   # composed.modalities["cycif"] -> MultiplexTissue

Loading across cohorts
----------------------

Use :class:`~spora_io.datasets.spora.SporaDataset` when training over multiple
datasets. It builds one composed dataset per cohort and samples either tissues
or tiles from a global index.

.. code-block:: python

   from spora_io import SporaDataset

   dataset = SporaDataset(
       ["dataset_a", "dataset_b"],
       modalities=["he", "codex"],
       resolution=1.0,
       tile_size=224,
       sampling_unit="tiles",
       modality_kwargs={"codex": {"standardization": "identity"}},
   )

   sample = dataset.sample_random_tile(preprocess=False)
   # sample["dataset_name"], sample["tissue_id"], sample["tile_id"]
   # sample["modalities"]["he"] -> HETissue tile
   # sample["modalities"]["codex"] -> MultiplexTissue tile

Retrieving tiles
----------------

Tiles are fixed-size image regions precomputed from shared tissue masks and stored in
``tiling/<resolution>/<strategy>/<size>_tile_coordinates.parquet``.

Pass a ``tissue_id`` and ``tile_id`` to retrieve a single tile:

.. code-block:: python

   # H&E tile
   he_tile = he_dataset.get_tile(tissue_ids[0], tile_id=0)
   # he_tile.image shape: (3, 224, 224)

   # Multiplex tile
   mx_tile = mx_dataset.get_tile(tissue_ids[0], tile_id=0, kind="complete")
   # mx_tile.image shape: (C, 224, 224)

Inspecting channel metadata
----------------------------

The multiplex dataset exposes channel-level metadata:

.. code-block:: python

   # Full channel list (DataFrame)
   dataset.channel_list[["channel_name", "qc_pass", "uniprot_id", "is_nuclear_marker"]]

   # Per-tissue channel availability matrix
   dataset.image_channel_map.head()

Working with masks
------------------

.. code-block:: python

   # Tissue mask (background vs tissue)
   mask = dataset.get_tissue_mask(tissue_ids[0])
   # mask.mask is a boolean NDArray of shape (H, W)

   # Cell instance mask (integer labels per cell)
   cell_mask = dataset.get_cell_instance_mask(tissue_ids[0])
   # cell_mask.mask is an integer NDArray of shape (H, W)

   # Cell task masks (e.g. cell type annotations)
   mask_types = dataset.get_cell_task_mask_types()
   task_mask = dataset.get_cell_task_mask(tissue_ids[0], mask_types[0])
   # task_mask.mapping maps integer IDs to label strings

Label filtering
---------------

Filter tissues by metadata columns at dataset construction time:

.. code-block:: python

   dataset = HEImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       resolution=1.0,
       tile_size=224,
       label="Histology",
       labels_to_keep=["Adenocarcinoma", "Mucinous"],
       label_type="classification",
   )

   # Only tissues matching the filter are loaded
   print(dataset.unique_labels)    # array of kept label values
   print(dataset.label_encoder)    # {label: int} mapping

Using The Shared Dataset Layout
-------------------------------

The library expects the current dataset format. The most important pieces are:

.. code-block:: text

   my_dataset/
   ├── metadata/
   │   └── tissues.parquet
   ├── he/
   │   └── 1_0mpp/
   │       └── images/
   ├── codex/
   │   ├── channels.parquet
   │   ├── channels_per_tissue.parquet
   │   └── 1_0mpp/
   │       ├── images/
   │       └── standardization/
   │           └── quantile_clipping/
   │               └── uq_0.99_image/
   ├── segmentations/
   │   └── 1_0mpp/
   │       ├── tissue_masks/
   │       └── cell_masks/
   └── tiling/
       └── 1_0mpp/
           └── default/
               ├── 224_tile_coordinates.parquet
               └── 224_tile_stats.parquet
