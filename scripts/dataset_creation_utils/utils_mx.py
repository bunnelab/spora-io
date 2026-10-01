import hashlib
import string

import numpy as np
import zarr
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk, isotropic_closing

from spora_io._config import get_datasets_dir

from . import ZARR_COMPRESSOR


def create_folderstructure(modality_to_mpp:dict, new_folder_name:str):
    """
    Creates the folder structure for a new dataset. Only the folders that are always required are created.
    args:
    modality_to_mpp: dict mapping modality to list of mpp values
    new_folder_name: name of the new dataset
    """
    modality_to_mpp = {m: [str(mpp).replace(".", "_") + "mpp" for mpp in mpp_list] for m, mpp_list in modality_to_mpp.items()}

    new_path = get_datasets_dir() / new_folder_name
    new_path.mkdir(exist_ok=True)
    (new_path/"metadata").mkdir(exist_ok=True)

    (new_path/"segmentations").mkdir(exist_ok=True)
    
    for modality in modality_to_mpp.keys():

        (new_path/modality).mkdir(exist_ok=True)
        for mpp in modality_to_mpp[modality]:
            (new_path/modality/mpp).mkdir(exist_ok=True)
            (new_path/"segmentations"/mpp/"tissue_masks").mkdir(exist_ok=True, parents=True)

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

def hash_patient(patient_id: str, length=8) -> str:
    """
    Hashes a patient id and returns a hash of length
    containing only lowercase letters.

    Args:
        input_string (str): The input string to hash.
        length (int): The length of the output hash.

    Returns:
        str: The hashed patient id
    """

    # Create a SHA-256 hash of the input string
    sha256_hash = hashlib.sha256(patient_id.encode()).hexdigest()

    # Define the set of lowercase letters
    lowercase_letters = string.ascii_lowercase
    num_letters = len(lowercase_letters)

    # Map the hash to a length-character string of lowercase letters
    patient_hash = ''.join(
        lowercase_letters[int(sha256_hash[i:i+2], 16) % num_letters]
        for i in range(0, length*2, 2)  # Use only the first 2 * length bytes of the hash
    )

    return patient_hash

def save_np_as_zarr(image: np.ndarray, path_to_zarr: str):
    image = np.array(image, order="C")
    C, H, W = image.shape

    crop_size = 256
    shard_size = 1024 # adapted based on WSI size
    
    arr = zarr.create_array(
        path_to_zarr,
        dtype=image.dtype,
        zarr_format=3,
        shape=image.shape, 
        shards=(C, shard_size, shard_size), 
        chunks=(C, crop_size, crop_size),
        compressors=ZARR_COMPRESSOR,
        overwrite=True
    )
    arr[:] = image
    del image, arr

def save_np_as_npz(image: np.ndarray, path_to_npz: str):
    np.savez_compressed(path_to_npz, mask=image)