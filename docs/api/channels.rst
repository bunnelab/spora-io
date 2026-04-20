Channel Selection
=================

Channel selection transforms randomly drop or subsample channels from multiplex
images during training. The implementations live in
``spora_io.utils.dataset.channels``.

Available selectors
-------------------

- ``BaseChannelSelector``
- ``DropChannelsFraction``
- ``DropChannelsFixedNumber``
- ``DropChannelsFixedNumberRange``
- ``DropChannelsNuclearKnown``
- ``HierarchicalChannelSampling``

These transforms are designed for multiplex image tensors and can be used in
training-time augmentation pipelines when channel subsets should vary between
samples.
