"""Sphinx configuration for spatialprot-data."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "spatialprot-data"
copyright = "2025, Bunne Lab"
author = "Eeshaan Jain, Lukas Klein"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "zarr",
    "einops",
    "safetensors",
    "loguru",
    "PIL",
    "tqdm",
]
