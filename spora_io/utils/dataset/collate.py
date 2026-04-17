from __future__ import annotations

from typing import Any, List, Tuple, TypeAlias

import torch

from spora_io.datasets._types import ComposedTissue, Tissue

Image = Tissue | ComposedTissue
ChannelMetadata: TypeAlias = Any
CollatedImage: TypeAlias = torch.Tensor | dict[str, torch.Tensor]
CollatedChannels: TypeAlias = Any


def _to_tensor(x: torch.Tensor | Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _get_channel_metadata(item: Tissue) -> ChannelMetadata:
    if hasattr(item, "uniprot_ids") and item.uniprot_ids is not None:
        return item.uniprot_ids
    if hasattr(item, "channel_names") and item.channel_names is not None:
        return item.channel_names
    if hasattr(item, "channels"):
        return item.channels
    return None


def _get_item_image(item: Image) -> CollatedImage:
    if isinstance(item, ComposedTissue):
        return {mod: _to_tensor(tissue.tissue) for mod, tissue in item.modalities.items()}
    return _to_tensor(item.tissue)


def _get_item_channels(item: Image) -> CollatedChannels:
    if isinstance(item, ComposedTissue):
        return {mod: _get_channel_metadata(tissue) for mod, tissue in item.modalities.items()}
    return _get_channel_metadata(item)


def abstract_collate_fn(
    batch: List[Image],
    image_type: str = "crop",
) -> Tuple[List[CollatedImage], List[str], List[CollatedChannels]]:
    """
    Collate tissue dataclasses into parallel lists.

    Notes:
    - Multiplex tissues/crops use the `tissue` field for image data, even when `kind == "crop"`.
    - Composed tissues are returned as modality -> tensor dictionaries.
    - Channel metadata prefers `uniprot_ids`, then `channel_names`, then `channels`.
    - Images are returned as a list instead of stacked because channel counts and shapes can vary.
    """
    if not batch:
        return [], [], []

    if image_type not in {"crop", "tissue"}:
        raise ValueError(f"Invalid image_type {image_type!r}. Valid options are 'crop' and 'tissue'.")

    images: List[CollatedImage] = []
    tissue_ids: List[str] = []
    channels: List[CollatedChannels] = []

    for item in batch:
        if image_type == "crop" and isinstance(item, ComposedTissue):
            raise ValueError("ComposedTissue does not support image_type='crop'.")
        images.append(_get_item_image(item))
        tissue_ids.append(item.tissue_id)
        channels.append(_get_item_channels(item))

    return images, tissue_ids, channels
