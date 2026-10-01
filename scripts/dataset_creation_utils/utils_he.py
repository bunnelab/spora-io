import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.ndimage import binary_dilation, binary_fill_holes
from torch import nn
from torchvision import transforms
from skimage.transform import rescale
from PIL import Image

from . import ZARR_COMPRESSOR

# --- HE ---

class SegmentationModel(torch.nn.Module):
    def __init__(self, freeze=True, **build_kwargs):
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = 0.5

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
    def forward(self, batch):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(batch)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class HESTSegmenter(SegmentationModel):

    def _build(self, checkpoint_dir, device):

        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
        model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=2,
            kernel_size=1,
            stride=1
        )

        # download is not prev. downloaded. 
        if not os.path.isfile(os.path.join(checkpoint_dir, 'deeplabv3_seg_v4.ckpt')):
            try:
                from huggingface_hub import snapshot_download
            except:
                raise Exception("Please install huggingface_hub (`pip install huggingface_hub`) to use this model")
            snapshot_download(repo_id="MahmoodLab/hest-tissue-seg", repo_type='model', local_dir=checkpoint_dir, cache_dir=checkpoint_dir)

        checkpoint = torch.load(os.path.join(checkpoint_dir, 'deeplabv3_seg_v4.ckpt'), map_location=device, weights_only=False)
            
        clean_state_dict = {}
        for key in checkpoint['state_dict']:
            if 'aux' in key:
                continue
            new_key = key.replace('model.', '')
            clean_state_dict[new_key] = checkpoint['state_dict'][key]
        model.load_state_dict(clean_state_dict)
        model.to(device)
        self.device = device
        self.input_size = 512

        eval_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        return model, eval_transforms
    

def segmentation_model_factory(model_name, device, freeze=True):
    '''
    Build a slide encoder based on model name.
    '''
    if model_name == 'hest':
        return HESTSegmenter(freeze, device = device)
    else:
        raise ValueError(f"Model type {model_name} not supported")


def get_mask(img:np.ndarray, segmentation_model: SegmentationModel, fill_holes=True, return_largest_mask_only=True, do_dilation=True):
    from skimage.measure import label
    with torch.no_grad():
        img = segmentation_model.eval_transforms(img)
        img_input = img.unsqueeze(0).to("cuda:0")
        mask = segmentation_model(img_input)["out"]
        mask = mask.detach().cpu().numpy()
        mask = mask[0, 1]
        mask = mask > 0.45
        mask = mask.astype(np.uint8)
        if do_dilation:
            mask = binary_dilation(mask, iterations=10)
        
        if fill_holes:
            mask = binary_fill_holes(mask)

        if return_largest_mask_only:
            mask = label(mask)
            mask = (mask == np.bincount(mask.ravel())[1:].argmax() + 1)
        del img_input
        torch.cuda.empty_cache()
        return mask.astype(bool)


def convert_png_to_zarr(path_to_png: Path, path_to_zarr: Path, rescale_factor: float = None):
    data = Image.open(path_to_png)
    # convert to numpy array with CHW channel order
    data = np.array(data, order="C").transpose(2, 0, 1)  # Convert to CHW format
    save_rgb_as_zarr(data, path_to_zarr, rescale_factor=rescale_factor)

def save_rgb_as_zarr(image: np.ndarray, path_to_zarr: Path, rescale_factor: float = None):
    image = np.asarray(image, order="C")
    
    if rescale_factor is not None:
            image = rescale(image, (1, rescale_factor, rescale_factor), anti_aliasing=True, preserve_range=True).astype(image.dtype)
    
    crop_size = 256
    shard_size = 1024
    if image.shape[0] != 3:
        print("Warning: Image does not have 3 channels, but has shape:", image.shape)
        if 3 in image.shape:
            # Move channel axis to front
            print("Moving channel axis to front...")
            channel_axis = np.where(np.array(image.shape) == 3)[0][0]
            image = np.moveaxis(image, channel_axis, 0)
    
    assert image.shape[0] == 3, f"Image must have 3 channels, but has shape: {image.shape}"
    channels = image.shape[0]

    arr = zarr.create_array(
        path_to_zarr,
        dtype=image.dtype,
        zarr_format=3,
        shape=image.shape,
        shards=(channels, shard_size, shard_size),
        chunks=(channels, crop_size, crop_size),
        compressors=ZARR_COMPRESSOR,
        overwrite=True,
    )
    arr[:] = image