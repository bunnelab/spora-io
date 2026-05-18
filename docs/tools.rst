Tools and Scripts
=================

This page documents the repository-level utilities that operate on the current
dataset format. All commands should be run from the repository root with
``PYTHONPATH=.`` unless the package is installed in the active environment.

Before running these tools, point ``spora_io`` to the datasets root:

.. code-block:: bash

   export SPORA_DATASETS_DIR=/path/to/datasets_v2

Channel TUI
-----------

``scripts/marker_viz.py`` opens a terminal UI for browsing multiplex channel
metadata across datasets. It reads ``channels.parquet`` files for multiplex
modalities, shows marker names and UniProt IDs, and highlights likely nuclear
channels such as DAPI, Hoechst, Iridium, DNA stains, and Histone H3.

Install the UI dependencies if needed:

.. code-block:: bash

   pip install textual rich pandas pyarrow

Run:

.. code-block:: bash

   PYTHONPATH=. python scripts/marker_viz.py

Use this when checking marker naming consistency, UniProt coverage,
quality-control flags, or nuclear marker annotations across cohorts.

Dataset Inventory TUI
---------------------

``scripts/datasets_viz.py`` opens a terminal UI for dataset-level inspection.
It summarizes available modalities, metadata rows, image counts, tile counts
by tile size, available resolutions, and multiplex standardization specs.

Install dependencies:

.. code-block:: bash

   pip install textual rich pandas pyarrow

Run:

.. code-block:: bash

   PYTHONPATH=. python scripts/datasets_viz.py

Use this after curation or migration to quickly inspect whether datasets expose
the expected modalities, tiling files, and standardization outputs.


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

Fixed-grid tiling is also available. It lays a regular grid over the padded
mask extent and keeps tiles whose tissue fraction is at least
``1 - tolerance``. Because padded grid tiles can extend past the image edge,
grid outputs must be saved under a non-default strategy name; ``default`` is
reserved for the fast no-padding tile-loading path.

.. code-block:: bash

   PYTHONPATH=. python -m scripts.compute_tiling \
       --dataset-name schurch2020coordinated \
       --tile-size 224 \
       --resolution 1.0 \
       --tiling-method grid_stride224 \
       --grid \
       --stride 224 \
       --tolerance 0.85 \
       --overwrite

This writes:

.. code-block:: text

   <dataset>/tiling/1_0mpp/grid_stride224/224_tile_coordinates.parquet
   <dataset>/tiling/1_0mpp/grid_stride224/224_tile_stats.parquet

Important arguments:

- ``--dataset-name``: dataset folder under ``SPORA_DATASETS_DIR``.
- ``--tile-size``: square tile size in pixels.
- ``--resolution``: mask resolution in microns per pixel; ``1.0`` maps to
  ``1_0mpp``.
- ``--tiling-method``: output subdirectory under
  ``tiling/<resolution>/``. When ``--grid`` is used this must not be
  ``default``.
- ``--grid``: use padded fixed-grid tiling instead of adaptive greedy tiling.
  Edge grid tiles that cross the image boundary are represented by their
  original top-left ``row``/``col`` and are padded with zeros during dataset
  loading.
- ``--stride``: candidate lattice stride in pixels. If omitted, defaults to
  ``tile_size // 2`` for adaptive tiling and ``tile_size`` for grid tiling.
- ``--tolerance``: maximum invalid/background fraction allowed inside a tile.
  For grid tiling, this means a tile is kept when its tissue fraction is at
  least ``1 - tolerance``.
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

   <dataset>/<modality>/<resolution>/standardization/<method>/uq_<upper_quantile>_<quantile_level>/

For example, ``--method quantile_clipping --upper-quantile 0.99
--quantile-level image`` writes to:

.. code-block:: text

   <dataset>/<modality>/1_0mpp/standardization/quantile_clipping/uq_0.99_image/

The suffix encodes the quantile level, not the mean/std level. This is
intentional: means and standard deviations are computed after the selected
quantile clipping transform, so ``uq_0.99_image`` and ``uq_0.99_global`` can
have different ``*_means.parquet`` and ``*_stds.parquet`` values even when
``--stats-level`` is the same.

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

- ``--dataset-name``: dataset folder under ``SPORA_DATASETS_DIR``.
- ``--modality``: one of ``codex``, ``cycif``, ``imc``, or ``mibi``.
- ``--method``: ``quantile_clipping`` or ``quantile_clipping_log1p``.
- ``--quantile-level``: ``image`` or ``global`` for quantile computation.
  This value is encoded in the output spec directory. For example,
  ``--quantile-level image`` writes ``uq_0.99_image`` while
  ``--quantile-level global`` writes ``uq_0.99_global``.
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
       modality="codex",
       standardization="quantile_clipping/uq_0.99_image",
       resolution=1.0,
       tile_size=224,
   )
