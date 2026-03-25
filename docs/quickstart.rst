Quick Start
===========

Loading an H&E dataset
----------------------

.. code-block:: python

   from spatialprot_data import HEImagingDataset

   dataset = HEImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       resolution=1.0,
       crop_size=224,
   )

   tissue_ids = dataset.get_tissue_ids()
   tissue = dataset.get_tissue(tissue_ids[0])
   # tissue.tissue is a torch.Tensor of shape (3, H, W)

Loading a multiplex dataset
----------------------------

.. code-block:: python

   from spatialprot_data import MultiplexImagingDataset

   dataset = MultiplexImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       modality="cycif",        # or "imc", "codex"
       normalization="identity", # or "q99_clipping"
       resolution=1.0,
       crop_size=224,
   )

   tissue = dataset.get_tissue(tissue_ids[0], kind="filtered")
   # tissue.tissue is a torch.Tensor of shape (C, H, W)
   # tissue.channel_names contains the marker names

Composing multiple modalities
------------------------------

.. code-block:: python

   from spatialprot_data import ComposedImagingDataset

   dataset = ComposedImagingDataset(
       name="my_dataset",
       path="/path/to/dataset",
       modalities=["he", "cycif"],
       resolution=1.0,
       crop_size=224,
       modality_kwargs={"cycif": {"normalization": "identity"}},
   )

   composed = dataset.get_composed_tissue(tissue_ids[0])
   # composed.modalities["he"] -> HETissue
   # composed.modalities["cycif"] -> MultiplexTissue

Working with masks
------------------

.. code-block:: python

   # Tissue mask (background vs tissue)
   mask = dataset.get_tissue_mask(tissue_ids[0])
   # mask.mask is a boolean NDArray of shape (H, W)

   # Cell instance mask
   cell_mask = dataset.get_cell_instance_mask(tissue_ids[0])
   # cell_mask.mask is an integer NDArray of shape (H, W)
