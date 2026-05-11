"""Package-level configuration constants."""

import os
from pathlib import Path


def get_datasets_dir() -> Path:
    """Return the root datasets directory.
    """
    if "SPORA_DATASETS_DIR" not in os.environ:
        raise RuntimeError(
            "Environment variable SPORA_DATASETS_DIR is not set. Please set it to the root datasets directory."
        )
    return Path(os.environ["SPORA_DATASETS_DIR"])
