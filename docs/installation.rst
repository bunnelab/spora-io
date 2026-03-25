Installation
============

From PyPI
---------

.. code-block:: bash

   pip install spatialprot-data

From source
-----------

.. code-block:: bash

   git clone https://github.com/bunnelab/spatialprot-data.git
   cd spatialprot-data
   pip install -e ".[dev]"

Configuration
-------------

Set the ``SPATIALPROT_DATASETS_DIR`` environment variable to point to your
datasets root directory:

.. code-block:: bash

   export SPATIALPROT_DATASETS_DIR=/path/to/datasets

If not set, the default path ``/mnt/aimm/scratch/datasets_v2`` is used.
