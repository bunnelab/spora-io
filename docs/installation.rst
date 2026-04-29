Installation
============

This repository currently provides the ``spora_io`` Python package from source.
Install it in editable mode from the repository root:

.. code-block:: bash

   git clone https://github.com/bunnelab/spora-io.git
   cd spora-io
   pip install -e .

Required Data Configuration
---------------------------

The loaders need a root directory containing one or more curated datasets.
Set ``SPATIALPROT_DATASETS_DIR`` to that root:

.. code-block:: bash

   export SPATIALPROT_DATASETS_DIR=/path/to/datasets_folder

If the variable is not set, ``spora_io`` errs out.

Runtime Dependencies
--------------------

The core loaders expect common scientific Python packages, including
``numpy``, ``pandas``, ``torch``, ``zarr``, and ``einops``. Multiplex
standardization and parquet-backed metadata also require parquet support
through pandas, typically provided by ``pyarrow``.

Verify Install
--------------

.. code-block:: bash

   from spora_io import HEImagingDataset, MultiplexImagingDataset, ComposedImagingDataset, SporaDataset
   print("spora_io import OK")
