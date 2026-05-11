Utilities
=========

Standardization
---------------

The multiplex standardization stack currently lives in
``spora_io.utils.dataset.standardize``.

Important entry points:

- ``build_standardizer``
- ``BaseStandardizer``
- ``IdentityStandardizer``
- ``StatsBackedStandardizer``
- ``QuantileClippingStandardizer``
- ``QuantileClippingLog1PStandardizer``

These classes operate on the parquet-backed standardization layout under
``<modality>/<resolution>/standardization/<spec>/``.

Image Transforms
----------------

The filter factory used by multiplex datasets lives in
``spora_io.utils.dataset.transforms.FilterFactory``.

.. autoclass:: spora_io.utils.dataset.transforms.FilterFactory
   :members:


Tiling
------

.. autofunction:: spora_io.utils.helpers.tile.best_mask_tiling_try_to_stop

.. autoclass:: spora_io.utils.helpers.tile.Tile
   :members:

General
-------

.. autofunction:: spora_io.utils.utils.get_modalities_of_dataset

.. autofunction:: spora_io._config.get_datasets_dir
