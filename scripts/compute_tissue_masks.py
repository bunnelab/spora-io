""" Compute tissue masks for a given dataset. This script will create tissue masks for all images of all modalities and resolutions in the given dataset. 
For multimodal paired datasets, the H&E images are used for creation of the tissue mask.
This script works both on .zarr and .ome.zarr images.
"""

import os

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import (binary_dilation, binary_opening, disk,
                                isotropic_closing)
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from spora_io._config import get_datasets_dir

class HESTSegmenter(torch.nn.Module):
    def __init__(self, freeze=True, **build_kwargs):
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = 0.5

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

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

    def forward(self, batch):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(batch)
        return z

he_segmenter = HESTSegmenter(device='cuda')
he_segmenter.eval()
eval_transforms = he_segmenter.eval_transforms

def create_tissue_mask_sp(img: np.ndarray):
    # Reduce allocations: compute max once, then apply log1p in-place.
    img_max = np.max(img, axis=0).astype(np.float32, copy=False)
    np.log1p(img_max, out=img_max)
    np.log1p(img_max, out=img_max)
    np.log1p(img_max, out=img_max)
    if img.shape[1] * img.shape[2] > 1e8:
        thresh = threshold_otsu(img_max[::20, ::20])
    else:
        thresh = threshold_otsu(img_max)
    tissue_mask = img_max > thresh

    tissue_mask = binary_opening(tissue_mask, disk(2))
    tissue_mask = isotropic_closing(tissue_mask, radius=10)
    return tissue_mask.astype(bool)


def create_tissue_mask_he(
    img: np.ndarray,
    segmentation_model: HESTSegmenter,
    return_largest_mask_only=True,
    do_dilation=True,
    fill_holes=True,
    tile_size=1024,
    tile_overlap=128,
):
    from skimage.measure import label

    def _tile_starts(length: int, window: int, stride: int):
        if length <= window:
            return [0]
        starts = list(range(0, length - window + 1, stride))
        last_start = length - window
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    with torch.no_grad():
        if tile_overlap * 2 >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size / 2")

        if img.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {img.shape}")

        is_chw = img.shape[0] in (1, 3, 4) and img.shape[1] > 4 and img.shape[2] > 4
        is_hwc = img.shape[2] in (1, 3, 4) and img.shape[0] > 4 and img.shape[1] > 4

        if not (is_chw or is_hwc):
            raise ValueError(
                f"Unable to infer image layout for shape {img.shape}. Expected CHW or HWC with 1/3/4 channels."
            )

        if is_chw:
            height, width = img.shape[1], img.shape[2]
        else:
            height, width = img.shape[0], img.shape[1]

        stride = tile_size - 2 * tile_overlap

        y_starts = _tile_starts(height, tile_size, stride)
        x_starts = _tile_starts(width, tile_size, stride)

        mask = np.zeros((height, width), dtype=bool)
        device = getattr(segmentation_model, "device", "cuda:0")

        for y0 in y_starts:
            y1 = min(y0 + tile_size, height)
            for x0 in x_starts:
                x1 = min(x0 + tile_size, width)

                if is_chw:
                    tile = img[:, y0:y1, x0:x1]
                    tile = np.moveaxis(tile, 0, -1)
                else:
                    tile = img[y0:y1, x0:x1, :]

                tile = segmentation_model.eval_transforms(tile)
                img_input = tile.unsqueeze(0).to(device)

                tile_mask = segmentation_model(img_input)["out"]
                tile_mask = tile_mask.detach().cpu().numpy()[0, 1] > 0.45

                in_y0 = 0 if y0 == 0 else tile_overlap
                in_x0 = 0 if x0 == 0 else tile_overlap
                in_y1 = tile_mask.shape[0] if y1 == height else tile_mask.shape[0] - tile_overlap
                in_x1 = tile_mask.shape[1] if x1 == width else tile_mask.shape[1] - tile_overlap

                out_y0 = y0 + in_y0
                out_x0 = x0 + in_x0
                out_y1 = y0 + in_y1
                out_x1 = x0 + in_x1

                mask[out_y0:out_y1, out_x0:out_x1] |= tile_mask[in_y0:in_y1, in_x0:in_x1]

        if mask.sum() == 0:
            return mask.astype(bool)
        
        if do_dilation:
            mask = binary_dilation(mask, footprint=disk(3))

        if fill_holes:
            mask = binary_fill_holes(mask)

        if return_largest_mask_only:
            mask = label(mask)
            mask = (mask == np.bincount(mask.ravel())[1:].argmax() + 1)
        return mask.astype(bool)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to create tissue masks for")
    args = parser.parse_args()

    datasets_dir = get_datasets_dir()
    dataset_path = datasets_dir / args.dataset_name

    modalities = set([item.name for item in dataset_path.iterdir() if item.is_dir()]).intersection(set(["imc", "cycif", "codex", "he"]))
    modalities = sorted(list(modalities), key=lambda x: ["he", "imc", "cycif", "codex"].index(x))  # he are processed first
    tissue_ann = pd.read_parquet(dataset_path / "metadata" / "tissues.parquet")
    
    for modality in modalities:
        print("Processing modality:", modality)
        zarr_files = list((dataset_path / modality).glob("*/images/*.zarr"))
        if modality in ["codex", "cycif", "imc", "mibi"]:
            if os.path.exists(dataset_path / modality / "channels.parquet") and os.path.exists(dataset_path / modality / "channels_per_tissue.parquet"):
                channels = pd.read_parquet(dataset_path / modality / "channels.parquet")
                ch_per_image = pd.read_parquet(dataset_path / modality / "channels_per_tissue.parquet")
            else:
                channels, ch_per_image = None, None
        mpps = set([zarr_file.parent.parent.name for zarr_file in zarr_files])
        for mpp in mpps:
            os.makedirs(dataset_path / "segmentations" / mpp / "tissue_masks" , exist_ok=True, mode=0o775)
        
        for zarr_file in tqdm(zarr_files):
            print(f"Processing {zarr_file}")
            mpp = zarr_file.parent.parent.name
            save_path = dataset_path / "segmentations" / mpp / "tissue_masks" / zarr_file.stem.replace(".ome", "")
            if os.path.exists(str(save_path) + ".npz"):
                print(f"File {save_path}.npz already exists, skipping.")
                continue
            
            if "ome.zarr" in zarr_file.name:
                img = zarr.open(zarr_file/"0", mode='r')[:]
            else:
                img = zarr.open(zarr_file, mode='r')[:]

            if modality == "he":
                tissue_mask = create_tissue_mask_he(img, he_segmenter, tile_size=4096, return_largest_mask_only=False)
            else:
                if channels is not None:
                    tid = zarr_file.name.replace(".ome.zarr", "").replace(".zarr", "")
                    channels_for_image = ch_per_image.loc[tid].values.astype(bool)
                    valid_channels = channels["qc_pass"].values[channels_for_image]
                    if sum(valid_channels) == 0:
                        valid_channels[:] = True
                    img = img[valid_channels]
                tissue_mask = create_tissue_mask_sp(img)
            
            np.savez_compressed(save_path, mask=tissue_mask)