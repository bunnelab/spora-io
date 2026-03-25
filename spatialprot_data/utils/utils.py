import os
from loguru import logger
import random
import numpy as np
import torch
from safetensors import safe_open

def is_rank0() -> bool:
    """ 
    Check if the current process runs in RANK 0
    """
    return os.environ.get('RANK', '0') == '0'

def print_verbose(msg, level="INFO") -> None:
    """
    Print a message if the current process is RANK 0
    Args:
        msg: The message to print
    """
    if is_rank0():
        if level == "INFO":
            logger.info(msg)
        elif level == "WARNING":
            logger.warning(msg)
        elif level == "ERROR":
            logger.error(msg)
        elif level == "DEBUG":
            logger.debug(msg)
        else:
            logger.info(msg)

def set_seed(seed: int | None, omit_random: bool=False, omit_numpy: bool=False, omit_torch: bool=False, set_cudnn_deterministic: bool=True,
             set_cudnn_benchmark: bool = False, use_determinstic_algorithms: bool = False) -> None:
    """
    Configure the seed settings for reproducibility.
    Args:
        seed (int | None): The seed value to set. If None, no seed is set.
        omit_random (bool): If True, do not set the seed for the random module.
        omit_numpy (bool): If True, do not set the seed for numpy.
        omit_torch (bool): If True, do not set the seed for torch.
        set_cudnn_deterministic (bool): If True, sets torch.backends.cudnn.deterministic to True.
        set_cudnn_benchmark (bool): If True, sets torch.backends.cudnn.benchmark to True.
        use_determinstic_algorithms (bool): If True, enables the use of deterministic algorithms in PyTorch.
    """
    if seed is None:
        if is_rank0():
            logger.warning("seed is None, no seed is set.")
        return
    
    if not omit_random:
        random.seed(seed)
    
    if not omit_numpy:
        np.random.seed(seed)

    if not omit_torch:
        torch.manual_seed(seed) # type: ignore
        torch.cuda.manual_seed_all(seed)

    if set_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    
    if set_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if use_determinstic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)



def load_checkpoint_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def get_modalities_of_dataset(dataset_name, base_path):
    dataset_path = os.path.join(base_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset {dataset_name} does not exist in {base_path}")
    
    possible_modalities = ['he', 'imc', 'cycif', 'mibi', 'mif', 'ihc', 'codex']
    modalities = []
    for modality in possible_modalities:
        modality_path = os.path.join(dataset_path, modality)
        if os.path.exists(modality_path):
            modalities.append(modality)
    return modalities