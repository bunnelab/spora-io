Tools and Scripts
=================

This page documents the repository-level utilities that operate on the current
dataset format. All commands should be run from the repository root with
``PYTHONPATH=.`` unless the package is installed in the active environment.

Before running these tools, point ``spora_io`` to the datasets root:

.. code-block:: bash

   export SPATIALPROT_DATASETS_DIR=/path/to/datasets_v2

Channel TUI
-----------

``scripts/cli_marker.py`` opens a terminal UI for browsing multiplex channel
metadata across datasets. It reads ``channels.parquet`` files for multiplex
modalities, shows marker names and UniProt IDs, and highlights likely nuclear
channels such as DAPI, Hoechst, Iridium, DNA stains, and Histone H3.

Install the UI dependencies if needed:

.. code-block:: bash

   pip install textual rich pandas pyarrow

Run:

.. code-block:: bash

   PYTHONPATH=. python scripts/cli_marker.py

Use this when checking marker naming consistency, UniProt coverage,
quality-control flags, or nuclear marker annotations across cohorts.

Dataset Inventory TUI
---------------------

``scripts/viz_datasets.py`` opens a terminal UI for dataset-level inspection.
It summarizes available modalities, metadata rows, image counts, tile counts
by tile size, available resolutions, and multiplex standardization specs.

Install dependencies:

.. code-block:: bash

   pip install textual rich pandas pyarrow

Run:

.. code-block:: bash

   PYTHONPATH=. python scripts/viz_datasets.py

Use this after curation or migration to quickly inspect whether datasets expose
the expected modalities, tiling files, and standardization outputs.

Optional Streamlit Channel Viewer
---------------------------------

For a browser-based channel viewer, run:

.. code-block:: bash

   pip install streamlit pandas pyarrow
   PYTHONPATH=. python -m streamlit run scripts/visualize_multiplex_channels.py

This presents the same channel inventory in a Streamlit interface with filters
for dataset, modality, QC status, nuclear channels, UniProt mapping, and marker
search.

Compute Tiling
--------------

``scripts/compute_tiling.py`` computes tile coordinates from shared tissue
masks and writes parquet outputs under:

.. code-block:: text

   <dataset>/tiling/<resolution>/<tiling_method>/<tile_size>_tile_coordinates.parquet
   <dataset>/tiling/<resolution>/<tiling_method>/<tile_size>_tile_stats.parquet

Minimal command:

.. code-block:: bash

   PYTHONPATH=. python -m scripts.compute_tiling \
       --dataset-name schurch2020coordinated \
       --tile-size 224 \
       --resolution 1.0

Common production-style command:

.. code-block:: bash

   PYTHONPATH=. python -m scripts.compute_tiling \
       --dataset-name schurch2020coordinated \
       --tile-size 224 \
       --resolution 1.0 \
       --tiling-method default \
       --stride 224 \
       --tolerance 0.95 \
       --coverage-goal 0.95 \
       --min-gain-ratio 0.1 \
       --overwrite

Important arguments:

- ``--dataset-name``: dataset folder under ``SPATIALPROT_DATASETS_DIR``.
- ``--tile-size``: square tile size in pixels.
- ``--resolution``: mask resolution in microns per pixel; ``1.0`` maps to
  ``1_0mpp``.
- ``--tiling-method``: output subdirectory under
  ``tiling/<resolution>/``.
- ``--stride``: candidate lattice stride in pixels. If omitted, defaults to
  ``tile_size // 2``.
- ``--tolerance``: maximum invalid/background fraction allowed inside a tile.
- ``--coverage-goal``: target foreground coverage before adaptive stopping can
  trigger.
- ``--min-gain-ratio``: minimum marginal new foreground area required after
  ``coverage_goal`` has been reached.
- ``--overwrite``: recompute even if output parquet files already exist.

Compute Multiplex Standardization Stats
---------------------------------------

``scripts/compute_standardization_stats.py`` computes parquet-backed
standardization statistics for multiplex modalities. Outputs are written under:

.. code-block:: text

   <dataset>/<modality>/<resolution>/standardization/<method>/uq_<upper_quantile>/

For example, ``--method quantile_clipping --upper-quantile 0.99`` writes to:

.. code-block:: text

   <dataset>/<modality>/1_0mpp/standardization/quantile_clipping/uq_0.99/

Minimal command:

.. code-block:: bash

   PYTHONPATH=. python -m scripts.compute_standardization_stats \
       --dataset-name schurch2020coordinated \
       --modality codex \
       --method quantile_clipping \
       --quantile-level image \
       --stats-level global \
       --upper-quantile 0.99 \
       --resolution 1.0

Log-compressed standardization:

.. code-block:: bash

   PYTHONPATH=. python -m scripts.compute_standardization_stats \
       --dataset-name schurch2020coordinated \
       --modality codex \
       --method quantile_clipping_log1p \
       --quantile-level image \
       --stats-level global \
       --upper-quantile 0.99 \
       --resolution 1.0 \
       --overwrite

Important arguments:

- ``--dataset-name``: dataset folder under ``SPATIALPROT_DATASETS_DIR``.
- ``--modality``: one of ``codex``, ``cycif``, ``imc``, or ``mibi``.
- ``--method``: ``quantile_clipping`` or ``quantile_clipping_log1p``.
- ``--quantile-level``: ``image`` or ``global`` for quantile computation.
- ``--stats-level``: ``image`` or ``global`` for mean/std computation.
- ``--upper-quantile``: upper clipping quantile, usually ``0.99``.
- ``--lower-quantile``: optional lower clipping quantile.
- ``--resolution``: image resolution in microns per pixel.
- ``--overwrite``: recompute even if all expected files already exist.

After statistics are computed, use the spec with
:class:`~spora_io.datasets.multiplex.MultiplexImagingDataset`:

.. code-block:: python

   from spora_io import MultiplexImagingDataset

   dataset = MultiplexImagingDataset(
       name="schurch2020coordinated",
       path="/path/to/datasets_v2/schurch2020coordinated",
       modality="codex",
       standardization="quantile_clipping/uq_0.99",
       resolution=1.0,
       tile_size=224,
   )
