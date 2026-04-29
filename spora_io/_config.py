"""Package-level configuration constants."""

import os
from pathlib import Path


def get_datasets_dir() -> Path:
    """Return the root datasets directory.

    Uses the SPORA_DATASETS_DIR environment variable if set,
    otherwise falls back to /mnt/aimm/scratch/datasets_v2.
    """
    return Path(os.environ.get(
        "SPORA_DATASETS_DIR",
        "/mnt/aimm/scratch/datasets_v2",
    ))
