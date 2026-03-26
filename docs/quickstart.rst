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

Retrieving crops
----------------

Crops are fixed-size tiles precomputed from tissue masks. Pass a ``tissue_id``
and ``crop_id`` to retrieve a single tile:

.. code-block:: python

   # H&E crop
   he_crop = he_dataset.get_crop(tissue_ids[0], crop_id=0)
   # he_crop.tissue shape: (3, 224, 224)

   # Multiplex crop
   mx_crop = mx_dataset.get_crop(tissue_ids[0], crop_id=0, kind="complete")
   # mx_crop.tissue shape: (C, 224, 224)

Inspecting channel metadata
----------------------------

The multiplex dataset exposes channel-level metadata:

.. code-block:: python

   # Full channel list (DataFrame)
   dataset.channel_list[["channel_name", "qc_pass", "uniprot_id"]]

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
       crop_size=224,
       label="Histology",
       labels_to_keep=["Adenocarcinoma", "Mucinous"],
       label_type="classification",
   )

   # Only tissues matching the filter are loaded
   print(dataset.unique_labels)    # array of kept label values
   print(dataset.label_encoder)    # {label: int} mapping
