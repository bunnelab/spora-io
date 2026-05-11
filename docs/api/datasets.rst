Dataset Classes
===============

These classes define the main loading interfaces for the current on-disk
dataset format.

Base
----

.. autoclass:: spora_io.datasets.base.BaseImagingDataset
   :members:
   :show-inheritance:

H&E
---

.. autoclass:: spora_io.datasets.he.HEImagingDataset
   :members:
   :show-inheritance:

Multiplex (IMC, CODEX, CycIF, etc.)
-----------------------------------

.. autoclass:: spora_io.datasets.multiplex.MultiplexImagingDataset
   :members:
   :show-inheritance:

IHC
---

.. autoclass:: spora_io.datasets.ihc.SingleIHCImagingDataset
   :members:
   :show-inheritance:

Composed (Multi-modal)
-----------------------

.. autoclass:: spora_io.datasets.compose.ComposedImagingDataset
   :members:
   :show-inheritance:

SporaDataset (Multi-cohort)
---------------------------

.. autoclass:: spora_io.datasets.spora.SporaDataset
   :members:
   :show-inheritance:
