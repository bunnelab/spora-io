Utilities
=========

Normalization
-------------

.. autofunction:: spatialprot_data.utils.dataset.build_normalizer

.. autoclass:: spatialprot_data.utils.dataset.normalize.BaseNormalizer
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.normalize.IdentityNormalizer
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.normalize.Q99Normalizer
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.normalize.Q99MeanStdNormalizer
   :members:
   :show-inheritance:

Image Transforms
----------------

.. autoclass:: spatialprot_data.utils.dataset.FilterFactory
   :members:

Tiling
------

.. autofunction:: spatialprot_data.utils.helpers.crop.best_mask_tiling_try_to_stop

.. autoclass:: spatialprot_data.utils.helpers.crop.Tile
   :members:

Collation
---------

.. autofunction:: spatialprot_data.utils.dataset.abstract_collate_fn

General
-------

.. autofunction:: spatialprot_data.utils.utils.is_rank0

.. autofunction:: spatialprot_data.utils.utils.print_verbose

.. autofunction:: spatialprot_data.utils.utils.set_seed

.. autofunction:: spatialprot_data.utils.utils.get_modalities_of_dataset
