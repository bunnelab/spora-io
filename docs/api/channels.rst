Channel Selection
=================

Channel selection transforms randomly drop or subsample channels from multiplex
images during training. All selectors inherit from
:class:`~spatialprot_data.utils.dataset.channels.BaseChannelSelector` and are
applied by calling the instance as a function.

Base Class
----------

.. autoclass:: spatialprot_data.utils.dataset.channels.BaseChannelSelector
   :members:
   :show-inheritance:

Selectors
---------

.. autoclass:: spatialprot_data.utils.dataset.DropChannelsFraction
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.DropChannelsFixedNumber
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.DropChannelsFixedNumberRange
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.DropChannelsNuclearKnown
   :members:
   :show-inheritance:

.. autoclass:: spatialprot_data.utils.dataset.HierarchicalChannelSampling
   :members:
   :show-inheritance:
