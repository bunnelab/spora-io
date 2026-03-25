import torch
from typing import List, Union, Tuple
from spatialprot_data.datasets._types import Tissue

Image = Union[Tissue]

def abstract_collate_fn(batch: List[Image], image_type: str = 'crop') -> Tuple[List[torch.Tensor], List[str], List[torch.Tensor] | List[str]]:
    """
    A generic collate function that can be used for any dataset.
    It assumes that each item in the batch is a dictionary with the same keys.
    It will stack tensors and keep other types as lists.

    Args:
        batch (list): A list of items, where each item is a dictionary.
        image_type (str): Type of image data, either 'crop' or 'tissue'.

    Returns:
        dict: A dictionary with the same keys as the input items, where tensor values are stacked and others are lists.
    """
    if not batch:
        return {}
    
    images = []
    tissue_ids = []
    channels = []
    if image_type == 'crop':
        # we get a Crop object, which has image in the key crop, and tissue_id, crop_id in metadata
        # we return list of crops, list of tissue_ids, list of crop_ids
        for item in batch:
            images.append(item.crop)
            tissue_ids.append(item.tissue_id)
            channels.append(item.channels)
        
    elif image_type == 'tissue':
        # we get a Tissue object, which has image in the key tissue, and tissue_id in metadata
        # we return list of tissues, list of tissue_ids
        for item in batch:
            images.append(item.tissue)
            tissue_ids.append(item.tissue_id)
            channels.append(item.channels)

    return (
        images,
        tissue_ids,
        channels,
    )
