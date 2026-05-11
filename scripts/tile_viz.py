"""
Tiling Visualizer — FastAPI + Canvas Edition
=============================================
A proper web app: FastAPI serves data as JSON/PNG, the browser renders
everything on an HTML5 Canvas with real click-to-select tile support.

For multiplex modalities (imc, codex, cycif, mibi) you can pick 1 to 5
markers directly in the sidebar.

Usage:
    python viz_tile3.py <dataset_name> <tile_size> [modality]
    python tiling_visualizer.py --help

Then open http://localhost:8000 in your browser.

Dependencies:
    pip install fastapi uvicorn pillow numpy
    (plus your aimm_internal package)
"""

import sys
import io
import argparse
from pathlib import Path
from typing import List
from loguru import logger

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from spora_io._config import get_datasets_dir

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

VALID_MODALITIES     = ["he", "imc", "codex", "cycif", "mibi"]
MULTIPLEX_MODALITIES = {"imc", "codex", "cycif", "mibi"}
DATASET_DIR          = get_datasets_dir()

def discover_modalities(dataset_name: str) -> List[str]:
    dataset_root = DATASET_DIR / dataset_name
    available: List[str] = []
    for modality in VALID_MODALITIES:
        if (dataset_root / modality).is_dir():
            available.append(modality)
    ihc_root = dataset_root / "ihc"
    if ihc_root.is_dir():
        available.extend(sorted(path.name for path in ihc_root.iterdir() if path.is_dir() and path.name.startswith("ihc_")))
    return available


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(obj):
    if hasattr(obj, "image"):  obj = obj.image
    if hasattr(obj, "data"):   obj = obj.data
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(obj)


def _normalise_channel(ch: np.ndarray) -> np.ndarray:
    """Percentile-stretch a 2-D float channel → uint8 [0, 255]."""
    ch = ch.astype(np.float32)
    lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
    if hi > lo:
        ch = (ch - lo) / (hi - lo)
    return (ch.clip(0, 1) * 255).astype(np.uint8)


def _raw_to_chw(raw) -> np.ndarray:
    """
    Convert any tissue object → float32 (C, H, W).
    Handles (H,W), (H,W,C) and (C,H,W) inputs.
    """
    arr = _to_numpy(raw).squeeze()
    if arr.ndim == 2:
        return arr[np.newaxis].astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= arr.shape[2]:
            return arr.astype(np.float32)
        return np.moveaxis(arr, -1, 0).astype(np.float32)
    return arr.astype(np.float32)


def _compose_rgb(chw: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Pick three channels from (C,H,W), stretch each independently → uint8 (H,W,3)."""
    C = chw.shape[0]
    def _ch(i):
        return _normalise_channel(chw[max(0, min(C - 1, i))])
    return np.stack([_ch(r), _ch(g), _ch(b)], axis=-1)


MARKER_COLORS = np.asarray(
    [
        (1.00, 0.20, 0.20),
        (0.20, 1.00, 0.35),
        (0.25, 0.55, 1.00),
        (1.00, 0.85, 0.15),
        (1.00, 0.25, 0.95),
    ],
    dtype=np.float32,
)


def _parse_marker_indices(channels: str | None, fallback: list[int], num_channels: int) -> list[int]:
    if channels:
        parsed: list[int] = []
        for part in channels.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError:
                continue
        indices = parsed
    else:
        indices = fallback

    if not indices:
        indices = [0]
    return [max(0, min(num_channels - 1, idx)) for idx in indices[:5]]


def _compose_marker_overlay(chw: np.ndarray, indices: list[int]) -> np.ndarray:
    """Render 1 marker as grayscale or 2-5 markers as additive colored overlays."""
    C = chw.shape[0]
    indices = [max(0, min(C - 1, idx)) for idx in indices[:5]]
    if len(indices) == 1:
        ch = _normalise_channel(chw[indices[0]])
        return np.stack([ch, ch, ch], axis=-1)

    rgb = np.zeros((*chw.shape[1:], 3), dtype=np.float32)
    for channel_idx, color in zip(indices, MARKER_COLORS, strict=False):
        ch = _normalise_channel(chw[channel_idx]).astype(np.float32) / 255.0
        rgb += ch[..., None] * color
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _he_to_rgb(chw: np.ndarray) -> np.ndarray:
    """H&E: use first 3 channels (or replicate greyscale)."""
    C = chw.shape[0]
    if C == 1:
        ch = _normalise_channel(chw[0])
        return np.stack([ch, ch, ch], axis=-1)
    return np.stack([_normalise_channel(chw[min(i, C-1)]) for i in range(3)], axis=-1)


def resolution_to_dir(resolution: float | str) -> str:
    return f"{str(resolution).replace('.', '_')}mpp"


def load_dataset(dataset_name, modality, crop_size, resolution, tile_strategy):
    from spora_io.datasets.he import HEImagingDataset
    from spora_io.datasets.ihc import SingleIHCImagingDataset
    from spora_io.datasets.multiplex import MultiplexImagingDataset
    if modality == "he":
        return HEImagingDataset(
            name=dataset_name, path=DATASET_DIR / dataset_name,
            verbose=False, resolution=resolution, tile_size=crop_size, tile_strategy=tile_strategy,
        )
    if modality.startswith("ihc_"):
        return SingleIHCImagingDataset(
            name=dataset_name, path=DATASET_DIR / dataset_name,
            marker_name=modality, verbose=False, resolution=resolution, tile_size=crop_size, tile_strategy=tile_strategy,
        )
    return MultiplexImagingDataset(
        name=dataset_name, modality=modality,
        path=DATASET_DIR / dataset_name,
        verbose=False, resolution=resolution, tile_size=crop_size, tile_strategy=tile_strategy,
        standardization="identity",
    )


def load_tile_coords(dataset_name, resolution, crop_size, tile_strategy):
    p = DATASET_DIR / dataset_name / "tiling" / resolution_to_dir(resolution) / tile_strategy / f"{crop_size}_tile_coordinates.parquet"
    if p.exists():
        import pandas as pd
        coords_df = pd.read_parquet(p)
        required_columns = {"tissue_id", "tile_id", "row", "col"}
        if not required_columns.issubset(coords_df.columns):
            raise ValueError(
                f"Tile coordinate parquet {p} is missing required columns {sorted(required_columns)}."
            )
        coords_df = coords_df.sort_values(["tissue_id", "tile_id"], kind="stable")
        return {
            str(tissue_id): [(int(row), int(col)) for row, col in zip(group["row"], group["col"], strict=False)]
            for tissue_id, group in coords_df.groupby("tissue_id", sort=False)
        }
    return {}


def filter_tile_coords_to_dataset(tile_coords, dataset):
    modality_tissue_ids = set(dataset.get_tissue_ids(kind="modality").tolist())
    return {
        tissue_id: coords
        for tissue_id, coords in tile_coords.items()
        if tissue_id in modality_tissue_ids
    }


def load_stats_df(dataset_name, resolution, crop_size, tile_strategy):
    p = DATASET_DIR / dataset_name / "tiling" / resolution_to_dir(resolution) / tile_strategy / f"{crop_size}_tile_stats.parquet"
    if p.exists():
        import pandas as pd
        df = pd.read_parquet(p)
        if "tissue_id" in df.columns and df.index.name != "tissue_id":
            df = df.set_index("tissue_id")
        return df
    return None


def ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _get_channel_names(dataset, tid: str) -> List[str]:
    """Best-effort channel name retrieval; falls back to 'Ch N'."""
    try:
        names = dataset.get_channel_names(tid, kind="uniprot_filtered")
        if names is not None and len(names) > 0:
            return [str(n) for n in names]
    except Exception:
        pass
    try:
        tissue = dataset.get_tissue(tid, preprocess=False)
        if hasattr(tissue, "channel_names") and tissue.channel_names is not None:
            return [str(n) for n in tissue.channel_names]
        if hasattr(tissue, "channels") and tissue.channels is not None:
            channels = tissue.channels
            if isinstance(channels, str) and channels.upper() == "RGB":
                return ["R", "G", "B"]
            if isinstance(channels, str):
                return [str(channels)]
    except Exception:
        pass
    try:
        chw = _raw_to_chw(dataset.get_tissue(tid, preprocess=False))
        return [f"Ch {i}" for i in range(chw.shape[0])]
    except Exception:
        return [f"Ch {i}" for i in range(10)]


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    dataset_name: str = ""
    modality: str = ""
    crop_size: int = 256
    resolution: float = 1.0
    tile_strategy: str = "default"
    is_multiplex: bool = False
    mask_only: bool = False
    dataset = None
    tile_coords: dict = {}
    stats_df = None
    image_cache: dict = {}    # tid → chw_float32
    mask_cache: dict = {}     # tid → mask_uint8
    valid_tile_cache: dict = {}  # tid → [(row, col), ...]
    channel_names: List[str] = []

STATE = AppState()


def _get_tissue_mask_array(tid: str) -> np.ndarray:
    if tid not in STATE.mask_cache:
        mask_obj = STATE.dataset.get_tissue_mask(tid)
        raw_mask = np.asarray(mask_obj.mask if hasattr(mask_obj, "mask") else _to_numpy(mask_obj))
        mask = (raw_mask > 0).astype(np.uint8) * 255
        STATE.mask_cache[tid] = mask
        if len(STATE.mask_cache) > 16:
            del STATE.mask_cache[next(iter(STATE.mask_cache))]
    return STATE.mask_cache[tid]


def _get_tissue_chw(tid: str) -> np.ndarray:
    if tid not in STATE.image_cache:
        tissue = STATE.dataset.get_tissue(tid, preprocess=False)
        STATE.image_cache[tid] = _raw_to_chw(tissue)
        if len(STATE.image_cache) > 8:
            del STATE.image_cache[next(iter(STATE.image_cache))]
    return STATE.image_cache[tid]


def _get_valid_tiles(tid: str):
    if tid in STATE.valid_tile_cache:
        return STATE.valid_tile_cache[tid]

    tiles = STATE.tile_coords.get(tid, [])
    if not tiles:
        STATE.valid_tile_cache[tid] = []
        return []

    if STATE.mask_only:
        mask = _get_tissue_mask_array(tid)
        h, w = mask.shape
    else:
        chw = _get_tissue_chw(tid)
        _, h, w = chw.shape
    cs = STATE.crop_size

    valid_tiles = [
        (int(ty), int(tx))
        for ty, tx in tiles
        if 0 <= int(ty) and 0 <= int(tx) and int(ty) + cs <= h and int(tx) + cs <= w
    ]
    STATE.valid_tile_cache[tid] = valid_tiles
    return valid_tiles


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Tiling Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/tissues")
def get_tissues():
    return {
        "tissues":      sorted(STATE.tile_coords.keys()),
        "crop_size":    STATE.crop_size,
        "modality":     STATE.modality,
        "is_multiplex": STATE.is_multiplex,
        "mask_only":    STATE.mask_only,
        "channels":     STATE.channel_names,
    }


@app.get("/api/tiles/{tid}")
def get_tiles(tid: str):
    if tid not in STATE.tile_coords:
        raise HTTPException(404, f"Tissue '{tid}' not found")
    tiles = _get_valid_tiles(tid)
    return {"tid": tid, "tiles": tiles, "crop_size": STATE.crop_size}


@app.get("/api/image/{tid}")
def get_image(
    tid: str,
    mask: bool = False,
    r: int = Query(0, ge=0),
    g: int = Query(1, ge=0),
    b: int = Query(2, ge=0),
    channels: str | None = Query(None),
):
    if tid not in STATE.tile_coords:
        raise HTTPException(404, f"Tissue '{tid}' not found")
    mask_arr = _get_tissue_mask_array(tid)

    if mask or STATE.mask_only:
        arr = np.stack([mask_arr] * 3, axis=-1)
        full_h, full_w = mask_arr.shape
    else:
        chw = _get_tissue_chw(tid)
        full_h, full_w = chw.shape[1], chw.shape[2]
        if STATE.is_multiplex:
            indices = _parse_marker_indices(channels, [r, g, b], chw.shape[0])
            arr = _compose_marker_overlay(chw, indices)
        else:
            arr = _he_to_rgb(chw)

    headers = {
        "X-Full-W": str(full_w),
        "X-Full-H": str(full_h),
        "Access-Control-Expose-Headers": "X-Full-W, X-Full-H",
    }
    return StreamingResponse(
        io.BytesIO(ndarray_to_png_bytes(arr)), media_type="image/png", headers=headers
    )


@app.get("/api/tile_crop/{tid}/{tile_idx}")
def get_tile_crop(
    tid: str,
    tile_idx: int,
    mask: bool = False,
    r: int = Query(0, ge=0),
    g: int = Query(1, ge=0),
    b: int = Query(2, ge=0),
    channels: str | None = Query(None),
):
    tiles = _get_valid_tiles(tid)
    if not tiles:
        raise HTTPException(404, f"Tissue '{tid}' not found")
    if tile_idx < 0 or tile_idx >= len(tiles):
        raise HTTPException(400, f"Tile index {tile_idx} out of range")
    ty, tx        = tiles[tile_idx]
    cs            = STATE.crop_size
    mask_arr      = _get_tissue_mask_array(tid)
    mask_crop     = mask_arr[ty:ty+cs, tx:tx+cs]

    if mask or STATE.mask_only:
        arr = np.stack([mask_crop] * 3, axis=-1)
    elif STATE.is_multiplex:
        chw = _get_tissue_chw(tid)
        chw_crop = chw[:, ty:ty+cs, tx:tx+cs]
        indices = _parse_marker_indices(channels, [r, g, b], chw_crop.shape[0])
        arr = _compose_marker_overlay(chw_crop, indices)
    else:
        chw = _get_tissue_chw(tid)
        chw_crop = chw[:, ty:ty+cs, tx:tx+cs]
        arr = _he_to_rgb(chw_crop)

    valid_ratio = float((mask_crop > 0).mean())
    headers = {
        "X-Tile-Y":      str(ty),
        "X-Tile-X":      str(tx),
        "X-Valid-Ratio": f"{valid_ratio:.4f}",
        "Access-Control-Expose-Headers": "X-Tile-Y, X-Tile-X, X-Valid-Ratio",
    }
    return StreamingResponse(io.BytesIO(ndarray_to_png_bytes(arr)), media_type="image/png", headers=headers)


@app.get("/api/stats/{tid}")
def get_stats(tid: str):
    mask_arr = _get_tissue_mask_array(tid)
    if STATE.mask_only:
        H, W = mask_arr.shape
    else:
        chw = _get_tissue_chw(tid)
        _, H, W = chw.shape
    mfrac   = float((mask_arr > 0).mean())
    mpix    = int((mask_arr > 0).sum())
    tiles   = _get_valid_tiles(tid)
    cs      = STATE.crop_size
    ratios  = [float(((mask_arr[ty:ty+cs, tx:tx+cs]) > 0).mean()) for ty, tx in tiles]

    row_stats = {}
    if STATE.stats_df is not None and tid in STATE.stats_df.index:
        row   = STATE.stats_df.loc[tid]
        row_stats = {
            "coverage":     float(row["coverage_ratio"]) if "coverage_ratio" in row.index else None,
            "stored_tiles": int(row["num_tiles"]) if "num_tiles" in row.index else None,
        }

    return {
        "tid": tid, "H": H, "W": W,
        "mask_fraction": mfrac, "masked_px": mpix,
        "num_tiles": len(tiles), "tile_valid_ratios": ratios,
        **row_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frontend HTML
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tiling Visualizer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg:      #0a0a12;
    --panel:   #12121f;
    --panel2:  #1a1a2e;
    --border:  #252545;
    --border2: #333360;
    --text:    #e8e8f0;
    --muted:   #5a5a80;
    --dim:     #3a3a60;
    --blue:    #4fc3f7;
    --yellow:  #ffd54f;
    --green:   #81c784;
    --orange:  #ffb74d;
    --red:     #ef5350;
    --ch-r:    #ff6b6b;
    --ch-g:    #69db7c;
    --ch-b:    #74c0fc;
    --ch-y:    #ffd43b;
    --ch-m:    #f783ff;
    --font-mono: 'JetBrains Mono', monospace;
    --font-ui:   'Syne', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100%; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: var(--font-mono); font-size: 13px; overflow: hidden;
  }

  #app {
    display: grid;
    grid-template-columns: 240px 1fr 280px;
    grid-template-rows: 48px 1fr 200px;
    grid-template-areas:
      "header  header  header"
      "sidebar canvas  detail"
      "sidebar bottom  bottom";
    width: 100vw; height: 100vh;
  }

  /* Header */
  #header {
    grid-area: header;
    display: flex; align-items: center; gap: 16px;
    padding: 0 20px;
    background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .logo { font-family: var(--font-ui); font-weight: 800; font-size: 16px; color: var(--blue); letter-spacing: -0.02em; }
  .logo span { color: var(--muted); font-weight: 400; font-size: 12px; margin-left: 8px; }
  .metrics-row { display: flex; gap: 24px; margin-left: auto; }
  .metric { display: flex; flex-direction: column; align-items: flex-end; }
  .metric .val { color: var(--blue); font-size: 13px; font-weight: 600; }
  .metric .lbl { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }

  /* Sidebar */
  #sidebar {
    grid-area: sidebar;
    background: var(--panel); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-section { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  .sidebar-section.grow { flex: 1; overflow-y: auto; border-bottom: none; }
  .section-label {
    font-family: var(--font-ui); font-size: 9px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
  }

  /* Tissue list */
  #tissue-list { display: flex; flex-direction: column; gap: 2px; }
  .tissue-item {
    padding: 6px 8px; border-radius: 4px; cursor: pointer;
    font-size: 11px; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: background 0.1s, color 0.1s;
  }
  .tissue-item:hover { background: var(--panel2); color: var(--text); }
  .tissue-item.active { background: var(--panel2); color: var(--yellow); border-left: 2px solid var(--yellow); padding-left: 6px; }

  /* Buttons */
  .nav-row { display: flex; gap: 6px; align-items: center; margin-bottom: 8px; }
  .btn {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    border-radius: 4px; padding: 5px 10px; cursor: pointer;
    font-family: var(--font-mono); font-size: 11px;
    transition: border-color 0.15s, background 0.15s;
  }
  .btn:hover { border-color: var(--blue); background: var(--border); }
  .btn.primary { border-color: var(--blue); color: var(--blue); }
  .btn.primary:hover { background: rgba(79,195,247,0.1); }
  .btn.small { padding: 3px 7px; font-size: 10px; }
  .btn.danger { border-color: var(--red); color: var(--red); }
  .btn.danger:hover { background: rgba(239,83,80,0.1); }

  /* Inputs */
  .input-row { display: flex; gap: 6px; align-items: center; }
  input[type=number] {
    background: var(--panel2); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; padding: 5px 8px;
    font-family: var(--font-mono); font-size: 11px; outline: none; width: 80px;
  }
  input[type=number]:focus { border-color: var(--blue); }

  /* Toggle */
  .toggle-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .toggle { position: relative; width: 34px; height: 18px; cursor: pointer; flex-shrink: 0; }
  .toggle input { display: none; }
  .toggle-track { position: absolute; inset: 0; background: var(--border); border-radius: 9px; transition: background 0.2s; }
  .toggle input:checked ~ .toggle-track { background: var(--blue); }
  .toggle-thumb { position: absolute; top: 2px; left: 2px; width: 14px; height: 14px; background: var(--text); border-radius: 50%; transition: transform 0.2s; }
  .toggle input:checked ~ .toggle-thumb { transform: translateX(16px); }
  .toggle-label { font-size: 11px; color: var(--muted); }

  /* ── Channel selector ── */
  #channel-section { display: none; }
  #channel-section.visible { display: block; }

  .ch-count-row { display: flex; align-items: center; gap: 7px; margin-bottom: 8px; }
  .ch-count-row .ch-count-label { color: var(--muted); font-size: 10px; width: 68px; flex-shrink: 0; }
  .ch-row { display: flex; align-items: center; gap: 7px; margin-bottom: 7px; }
  .ch-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .ch-dot.r { background: var(--ch-r); box-shadow: 0 0 6px var(--ch-r); }
  .ch-dot.g { background: var(--ch-g); box-shadow: 0 0 6px var(--ch-g); }
  .ch-dot.b { background: var(--ch-b); box-shadow: 0 0 6px var(--ch-b); }
  .ch-lbl { font-size: 10px; color: var(--muted); width: 10px; flex-shrink: 0; font-weight: 600; }

  select {
    flex: 1; min-width: 0;
    background: var(--panel2); border: 1px solid var(--border);
    color: var(--text); border-radius: 4px; padding: 4px 22px 4px 7px;
    font-family: var(--font-mono); font-size: 10px; outline: none; cursor: pointer;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%235a5a80'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 7px center;
  }
  select:focus { border-color: var(--blue); }
  select option { background: var(--panel2); color: var(--text); }
  .ch-apply-row { display: flex; justify-content: flex-end; margin-top: 2px; }

  /* ── Resolution selector ── */
  /* Canvas */
  #canvas-area {
    grid-area: canvas; position: relative;
    overflow: hidden; background: var(--bg); cursor: crosshair;
  }
  #main-canvas { position: absolute; top: 0; left: 0; image-rendering: pixelated; }
  #canvas-hint {
    position: absolute; bottom: 10px; right: 12px;
    font-size: 10px; color: var(--muted); pointer-events: none;
    background: rgba(10,10,18,0.75); padding: 4px 8px;
    border-radius: 3px; border: 1px solid var(--border);
  }
  #canvas-coords {
    position: absolute; top: 8px; left: 8px;
    font-size: 10px; color: var(--muted); pointer-events: none;
    background: rgba(10,10,18,0.75); padding: 3px 7px; border-radius: 3px;
  }
  #loading-overlay {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(10,10,18,0.85); font-size: 12px; color: var(--muted); z-index: 10;
  }
  #loading-overlay.hidden { display: none; }
  .spinner {
    width: 20px; height: 20px; border: 2px solid var(--border);
    border-top-color: var(--blue); border-radius: 50%;
    animation: spin 0.6s linear infinite; margin-right: 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Detail */
  #detail {
    grid-area: detail; background: var(--panel);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
  }
  #detail-header {
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-family: var(--font-ui); font-size: 9px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
    display: flex; align-items: center; justify-content: space-between;
  }
  #tile-canvas-wrap {
    padding: 10px; display: flex; align-items: center; justify-content: center;
    border-bottom: 1px solid var(--border);
  }
  #tile-canvas { border: 2px solid var(--yellow); border-radius: 3px; image-rendering: pixelated; }
  #tile-info { padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }
  .info-row { display: flex; justify-content: space-between; align-items: center; }
  .info-row .ik { color: var(--muted); font-size: 10px; }
  .info-row .iv { color: var(--text); font-size: 11px; font-weight: 600; }
  .info-row .iv.good { color: var(--green); }
  .info-row .iv.warn { color: var(--orange); }
  #tile-placeholder {
    flex: 1; display: flex; align-items: center; justify-content: center;
    flex-direction: column; gap: 8px;
    color: var(--muted); font-size: 11px; text-align: center; padding: 20px;
  }
  #tile-placeholder .icon { font-size: 32px; margin-bottom: 4px; }

  /* Channel badges in detail panel */
  .ch-badge-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
  .ch-badge {
    font-size: 9px; padding: 2px 7px; border-radius: 3px;
    border: 1px solid; font-weight: 600; letter-spacing: 0.03em;
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }

  /* Bottom */
  #bottom {
    grid-area: bottom; display: grid; grid-template-columns: 1fr 1fr;
    border-top: 1px solid var(--border); overflow: hidden;
  }
  #histogram-wrap { padding: 10px 14px; border-right: 1px solid var(--border); display: flex; flex-direction: column; }
  #hist-canvas { flex: 1; min-height: 0; }
  #tile-table-wrap { display: flex; flex-direction: column; overflow: hidden; }
  #tile-table-header {
    padding: 8px 14px 4px; font-family: var(--font-ui); font-size: 9px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px;
  }
  #tile-table { flex: 1; overflow-y: auto; font-size: 10px; }
  #tile-table table { width: 100%; border-collapse: collapse; }
  #tile-table th {
    position: sticky; top: 0; background: var(--panel); color: var(--muted);
    text-align: left; padding: 4px 10px; font-weight: 400; border-bottom: 1px solid var(--border);
  }
  #tile-table td { padding: 3px 10px; color: var(--muted); border-bottom: 1px solid rgba(37,37,69,0.4); cursor: pointer; transition: background 0.1s; }
  #tile-table tr:hover td { background: var(--panel2); color: var(--text); }
  #tile-table tr.selected td { background: rgba(46,46,16,0.7); color: var(--yellow); }

  .chip { display: inline-block; background: var(--panel2); border: 1px solid var(--border); border-radius: 3px; padding: 1px 6px; font-size: 10px; color: var(--muted); }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
</head>
<body>
<div id="app">

  <header id="header">
    <div class="logo">🔬 Tiling Visualizer <span id="header-sub">loading…</span></div>
    <div class="metrics-row">
      <div class="metric"><span class="val" id="m-tiles">—</span><span class="lbl">tiles</span></div>
      <div class="metric"><span class="val" id="m-size">—</span><span class="lbl">image</span></div>
      <div class="metric"><span class="val" id="m-mask">—</span><span class="lbl">mask %</span></div>
      <div class="metric"><span class="val" id="m-cov">—</span><span class="lbl">coverage</span></div>
    </div>
  </header>

  <aside id="sidebar">

    <div class="sidebar-section">
      <div class="section-label">Navigation</div>
      <div class="nav-row">
        <button class="btn small" id="btn-prev">◀</button>
        <button class="btn small" id="btn-next">▶</button>
        <span id="tissue-counter" style="color:var(--muted);font-size:10px;margin-left:4px;">—</span>
      </div>
      <div class="section-label" style="margin-top:8px;">Display</div>
      <div class="toggle-row">
        <label class="toggle">
          <input type="checkbox" id="mask-toggle">
          <div class="toggle-track"></div><div class="toggle-thumb"></div>
        </label>
        <span class="toggle-label">Show tissue mask</span>
      </div>
    </div>

    <!-- Channel picker — only shown for multiplex modalities -->
    <div class="sidebar-section" id="channel-section">
      <div class="section-label">Marker Overlay</div>
      <div class="ch-count-row">
        <span class="ch-count-label">Markers</span>
        <select id="sel-ch-count"></select>
      </div>
      <div id="channel-picker-rows"></div>
      <div class="ch-apply-row">
        <button class="btn primary small" id="btn-apply-ch">Apply ↵</button>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-label">Jump to tile</div>
      <div class="input-row">
        <input type="number" id="jump-input" min="0" value="0" placeholder="#">
        <button class="btn primary small" id="btn-jump">Go</button>
      </div>
      <div id="sel-info" style="margin-top:8px;display:none;">
        <div style="color:var(--yellow);font-size:11px;font-weight:600;" id="sel-label"></div>
        <button class="btn danger small" id="btn-desel" style="margin-top:6px;">✕ Deselect</button>
      </div>
    </div>

    <div class="sidebar-section grow">
      <div class="section-label">Tissues</div>
      <div id="tissue-list"></div>
    </div>

  </aside>

  <div id="canvas-area">
    <canvas id="main-canvas"></canvas>
    <div id="canvas-coords">x=— y=—</div>
    <div id="canvas-hint">scroll=zoom · drag=pan · click=select tile</div>
    <div id="loading-overlay"><div class="spinner"></div> Loading…</div>
  </div>

  <div id="detail">
    <div id="detail-header">
      <span>Tile Detail</span>
      <span id="detail-tile-id" style="color:var(--yellow);"></span>
    </div>
    <div id="tile-placeholder">
      <div class="icon">🖱</div>
      <div>Click a tile on the canvas<br>or use Jump to tile</div>
    </div>
    <div id="tile-detail-content" style="display:none;flex-direction:column;flex:1;overflow:hidden;">
      <div id="tile-canvas-wrap">
        <canvas id="tile-canvas" width="200" height="200"></canvas>
      </div>
      <div id="tile-info">
        <div class="info-row"><span class="ik">Position</span><span class="iv" id="ti-pos">—</span></div>
        <div class="info-row"><span class="ik">Valid pixels</span><span class="iv" id="ti-valid">—</span></div>
        <div class="info-row"><span class="ik">Size</span><span class="iv" id="ti-size">—</span></div>
        <div id="ti-ch-badges" class="ch-badge-row" style="display:none;"></div>
      </div>
    </div>
  </div>

  <div id="bottom">
    <div id="histogram-wrap">
      <div class="section-label">Valid-pixel fraction distribution</div>
      <canvas id="hist-canvas"></canvas>
    </div>
    <div id="tile-table-wrap">
      <div id="tile-table-header">
        All tiles <span class="chip" id="table-count">0</span>
      </div>
      <div id="tile-table">
        <table>
          <thead><tr><th>#</th><th>y</th><th>x</th><th>valid%</th></tr></thead>
          <tbody id="tile-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────
const S = {
  tissues: [], tidIdx: 0, tiles: [], cropSize: 256,
  selTile: null, showMask: false,
  maskOnly: false,
  isMultiplex: false, channels: [],
  selectedChannels: [0, 1, 2],
  imgFullW: 0,       // true full-resolution image dimensions (from X-Full-W/H headers)
  imgFullH: 0,
  offsetX: 0, offsetY: 0, scale: 1,
  dragging: false, dragStartX: 0, dragStartY: 0, dragOffX: 0, dragOffY: 0,
  imgBitmap: null, stats: null,
};

// ── DOM ────────────────────────────────────────────────────────────────────
const canvas     = document.getElementById('main-canvas');
const ctx        = canvas.getContext('2d');
const histCanvas = document.getElementById('hist-canvas');
const histCtx    = histCanvas.getContext('2d');
const tileCanvas = document.getElementById('tile-canvas');
const tileCtx    = tileCanvas.getContext('2d');
const overlay    = document.getElementById('loading-overlay');

// ── URL builders ───────────────────────────────────────────────────────────
function imageUrl(tid) {
  const maskQ = `mask=${S.showMask}`;
  const chQ   = S.isMultiplex ? `&channels=${encodeURIComponent(S.selectedChannels.join(','))}` : '';
  return `/api/image/${tid}?${maskQ}${chQ}`;
}
function cropUrl(tid, idx) {
  if (S.showMask) return `/api/tile_crop/${tid}/${idx}?mask=true`;
  const q = S.isMultiplex ? `&channels=${encodeURIComponent(S.selectedChannels.join(','))}` : '';
  return `/api/tile_crop/${tid}/${idx}?mask=false${q}`;
}

// ── Fetch helpers ──────────────────────────────────────────────────────────
async function fetchBitmap(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  const bmp    = await createImageBitmap(await res.blob());
  const fullW  = parseInt(res.headers.get('X-Full-W') ?? '0') || bmp.width;
  const fullH  = parseInt(res.headers.get('X-Full-H') ?? '0') || bmp.height;
  return { bmp, res, fullW, fullH };
}

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  const data    = await fetch('/api/tissues').then(r => r.json());
  S.tissues     = data.tissues;
  S.cropSize    = data.crop_size;
  S.isMultiplex = data.is_multiplex;
  S.maskOnly    = data.mask_only;
  S.channels    = data.channels || [];

  document.getElementById('header-sub').textContent =
    `${data.modality.toUpperCase()} · ${S.cropSize}px tiles${S.maskOnly ? ' · mask-only' : ''}`;

  if (S.maskOnly) {
    const maskToggle = document.getElementById('mask-toggle');
    S.showMask = true;
    maskToggle.checked = true;
    maskToggle.disabled = true;
  }

  if (S.isMultiplex && !S.maskOnly && S.channels.length > 0) {
    document.getElementById('channel-section').classList.add('visible');
    S.selectedChannels = defaultSelectedChannels();
    buildChannelSelects();
    syncChannelSelects();
  }

  buildTissueList();
  await selectTissue(0);
}

// ── Channel selects ────────────────────────────────────────────────────────
const CHANNEL_COLORS = ['#ff6b6b', '#69db7c', '#74c0fc', '#ffd43b', '#f783ff'];
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function defaultSelectedChannels() {
  const n = Math.min(3, Math.max(1, S.channels.length));
  return Array.from({length: n}, (_, i) => i);
}
function buildChannelSelects() {
  const countSel = document.getElementById('sel-ch-count');
  const maxCount = Math.min(5, Math.max(1, S.channels.length));
  countSel.innerHTML = Array.from({length: maxCount}, (_, i) => {
    const count = i + 1;
    return `<option value="${count}">${count}</option>`;
  }).join('');
  countSel.value = String(Math.min(S.selectedChannels.length, maxCount));
  renderChannelRows();
  countSel.addEventListener('change', () => {
    const count = +countSel.value;
    const next = S.selectedChannels.slice(0, count);
    while (next.length < count) next.push(Math.min(next.length, S.channels.length - 1));
    S.selectedChannels = next;
    renderChannelRows();
    syncChannelSelects();
  });
}
function renderChannelRows() {
  const rows = document.getElementById('channel-picker-rows');
  const count = +document.getElementById('sel-ch-count').value || S.selectedChannels.length || 1;
  const options = S.channels.map((name, i) =>
    `<option value="${i}">${i} — ${escapeHtml(name)}</option>`
  ).join('');
  rows.innerHTML = Array.from({length: count}, (_, i) => {
    const color = CHANNEL_COLORS[i % CHANNEL_COLORS.length];
    return `
      <div class="ch-row">
        <div class="ch-dot" style="background:${color};box-shadow:0 0 6px ${color};"></div>
        <span class="ch-lbl">${i + 1}</span>
        <select id="sel-ch-${i}">${options}</select>
      </div>`;
  }).join('');
  for (let i = 0; i < count; i++) {
    document.getElementById(`sel-ch-${i}`).addEventListener('keydown', e => {
      if (e.key === 'Enter') applyChannels();
    });
  }
}
function syncChannelSelects() {
  const countSel = document.getElementById('sel-ch-count');
  countSel.value = String(S.selectedChannels.length);
  for (let i = 0; i < S.selectedChannels.length; i++) {
    const sel = document.getElementById(`sel-ch-${i}`);
    if (sel) sel.value = String(S.selectedChannels[i]);
  }
}
async function applyChannels() {
  const count = +document.getElementById('sel-ch-count').value || 1;
  S.selectedChannels = Array.from({length: count}, (_, i) => +document.getElementById(`sel-ch-${i}`).value);
  await reloadImage();
  if (S.selTile !== null) await loadTileDetail(S.selTile);
}
document.getElementById('btn-apply-ch').addEventListener('click', applyChannels);

// ── Resolution selector ────────────────────────────────────────────────────
// ── Tissue list ────────────────────────────────────────────────────────────
function buildTissueList() {
  const list = document.getElementById('tissue-list');
  list.innerHTML = '';
  S.tissues.forEach((tid, i) => {
    const el = document.createElement('div');
    el.className = 'tissue-item'; el.title = tid; el.textContent = tid;
    el.addEventListener('click', () => selectTissue(i));
    list.appendChild(el);
  });
}
function updateTissueList() {
  document.querySelectorAll('.tissue-item').forEach((el, i) =>
    el.classList.toggle('active', i === S.tidIdx));
  document.querySelector('.tissue-item.active')?.scrollIntoView({ block: 'nearest' });
  document.getElementById('tissue-counter').textContent = `${S.tidIdx + 1} / ${S.tissues.length}`;
}

// ── Select tissue ──────────────────────────────────────────────────────────
async function selectTissue(idx) {
  S.tidIdx  = ((idx % S.tissues.length) + S.tissues.length) % S.tissues.length;
  S.selTile = null;
  showOverlay(true); updateTissueList(); clearDetail();
  const tid = S.tissues[S.tidIdx];
  const [tilesData, stats, imgData] = await Promise.all([
    fetch(`/api/tiles/${tid}`).then(r => r.json()),
    fetch(`/api/stats/${tid}`).then(r => r.json()),
    fetchBitmap(imageUrl(tid)),
  ]);
  S.tiles    = tilesData.tiles;
  S.stats    = stats;
  S.imgBitmap = imgData.bmp;
  S.imgFullW  = imgData.fullW;
  S.imgFullH  = imgData.fullH;
  fitCanvas(); updateMetrics(); buildTileTable(); drawHistogram(); renderCanvas();
  showOverlay(false);
}

// ── Reload image (mask / channel / resolution change) ─────────────────────
async function reloadImage() {
  showOverlay(true);
  const imgData = await fetchBitmap(imageUrl(S.tissues[S.tidIdx]));
  S.imgBitmap = imgData.bmp;
  S.imgFullW  = imgData.fullW;
  S.imgFullH  = imgData.fullH;
  renderCanvas();
  showOverlay(false);
}

// ── Canvas fit & render ────────────────────────────────────────────────────
function fitCanvas() {
  const area = document.getElementById('canvas-area');
  canvas.width = area.clientWidth; canvas.height = area.clientHeight;
  const m = 20;
  S.scale   = Math.min((canvas.width-m*2)/S.imgFullW, (canvas.height-m*2)/S.imgFullH);
  S.offsetX = (canvas.width  - S.imgFullW * S.scale) / 2;
  S.offsetY = (canvas.height - S.imgFullH * S.scale) / 2;
}
function renderCanvas() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#0a0a12'; ctx.fillRect(0,0,W,H);
  if (!S.imgBitmap) return;
  ctx.imageSmoothingEnabled = S.scale > 1;
  ctx.drawImage(S.imgBitmap, S.offsetX, S.offsetY, S.imgFullW * S.scale, S.imgFullH * S.scale);

  const cs = S.cropSize * S.scale;
  for (let i = 0; i < S.tiles.length; i++) {
    const [ty, tx] = S.tiles[i], sel = i===S.selTile;
    const px = S.offsetX + tx*S.scale, py = S.offsetY + ty*S.scale;
    ctx.fillStyle   = sel ? 'rgba(255,213,79,0.28)' : 'rgba(79,195,247,0.10)';
    ctx.strokeStyle = sel ? '#ffd54f' : '#4fc3f7';
    ctx.lineWidth   = sel ? Math.max(1.5,S.scale) : Math.max(0.5,S.scale*0.5);
    ctx.fillRect(px,py,cs,cs); ctx.strokeRect(px,py,cs,cs);
  }
  if (S.selTile!==null && S.tiles[S.selTile]) {
    const [ty,tx] = S.tiles[S.selTile];
    ctx.shadowColor='#ffd54f'; ctx.shadowBlur=10;
    ctx.strokeStyle='#ffd54f'; ctx.lineWidth=2.5;
    ctx.strokeRect(S.offsetX+tx*S.scale, S.offsetY+ty*S.scale, cs, cs);
    ctx.shadowBlur=0;
  }
}

// ── Select tile ────────────────────────────────────────────────────────────
async function selectTile(idx) {
  if (idx < 0 || idx >= S.tiles.length) return;
  S.selTile = idx; renderCanvas(); updateTileTableSelection(); updateSelInfo();
  await loadTileDetail(idx);
}
async function loadTileDetail(idx) {
  const tid = S.tissues[S.tidIdx];
  document.getElementById('tile-placeholder').style.display    = 'none';
  document.getElementById('tile-detail-content').style.display = 'flex';
  document.getElementById('detail-tile-id').textContent = `#${idx}`;

  const { bmp, res } = await fetchBitmap(cropUrl(tid, idx));
  const [ty, tx]     = S.tiles[idx];
  const validRatio   = parseFloat(res.headers.get('X-Valid-Ratio') ?? '0');

  tileCanvas.width = tileCanvas.height = 200;
  tileCtx.imageSmoothingEnabled = false;
  tileCtx.drawImage(bmp, 0, 0, 200, 200);

  document.getElementById('ti-pos').textContent  = `y=${ty}  x=${tx}`;
  document.getElementById('ti-size').textContent = `${S.cropSize} × ${S.cropSize} px`;
  const el = document.getElementById('ti-valid');
  el.textContent = (validRatio*100).toFixed(1)+'%';
  el.className   = 'iv ' + (validRatio>=0.5?'good':validRatio>=0.2?'warn':'');

  const badgeWrap = document.getElementById('ti-ch-badges');
  if (S.isMultiplex) {
    badgeWrap.innerHTML = S.selectedChannels.map((channelIdx, i) => {
      const name = S.channels[channelIdx] || `Ch${channelIdx}`;
      const color = CHANNEL_COLORS[i % CHANNEL_COLORS.length];
      return `<span class="ch-badge" style="color:${color};border-color:${color};background:${color}14;" title="${escapeHtml(name)}">${i + 1}: ${escapeHtml(name)}</span>`;
    }).join('');
    badgeWrap.style.display = 'flex';
  } else {
    badgeWrap.style.display = 'none';
  }
}
function clearDetail() {
  document.getElementById('tile-placeholder').style.display    = 'flex';
  document.getElementById('tile-detail-content').style.display = 'none';
  document.getElementById('detail-tile-id').textContent = '';
  document.getElementById('sel-info').style.display = 'none';
  S.selTile = null;
}
function updateSelInfo() {
  if (S.selTile===null) { document.getElementById('sel-info').style.display='none'; return; }
  const [ty,tx] = S.tiles[S.selTile];
  document.getElementById('sel-label').textContent = `Tile #${S.selTile}  y=${ty}  x=${tx}`;
  document.getElementById('sel-info').style.display = 'block';
}

// ── Tile table ─────────────────────────────────────────────────────────────
function buildTileTable() {
  const tbody  = document.getElementById('tile-tbody');
  const ratios = S.stats?.tile_valid_ratios ?? [];
  tbody.innerHTML = '';
  S.tiles.forEach(([ty,tx], i) => {
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.innerHTML = `<td>${i}</td><td>${ty}</td><td>${tx}</td><td>${ratios[i]!=null?(ratios[i]*100).toFixed(1)+'%':'—'}</td>`;
    tr.addEventListener('click', () => selectTile(i));
    tbody.appendChild(tr);
  });
  document.getElementById('table-count').textContent = S.tiles.length;
}
function updateTileTableSelection() {
  document.querySelectorAll('#tile-tbody tr').forEach(tr =>
    tr.classList.toggle('selected', +tr.dataset.idx===S.selTile));
  document.querySelector('#tile-tbody tr.selected')?.scrollIntoView({ block: 'nearest' });
}

// ── Histogram ──────────────────────────────────────────────────────────────
function drawHistogram() {
  const wrap = document.getElementById('histogram-wrap');
  histCanvas.width  = Math.max(wrap.clientWidth-28, 100);
  histCanvas.height = Math.max(wrap.clientHeight-28, 60);
  const cW=histCanvas.width, cH=histCanvas.height;
  histCtx.clearRect(0,0,cW,cH);
  histCtx.fillStyle='#0a0a12'; histCtx.fillRect(0,0,cW,cH);
  const ratios = S.stats?.tile_valid_ratios??[];
  if (!ratios.length) return;
  const bins=20, counts=new Array(bins).fill(0);
  ratios.forEach(r=>{ counts[Math.min(bins-1,Math.floor(r*bins))]++; });
  const maxC=Math.max(...counts);
  const pL=28,pR=8,pT=8,pB=24, bW=(cW-pL-pR)/bins, bH=cH-pT-pB;
  counts.forEach((c,i)=>{
    histCtx.fillStyle='#4fc3f7';
    histCtx.fillRect(pL+i*bW+1, pT+bH-(maxC>0?c/maxC*bH:0), bW-2, maxC>0?c/maxC*bH:0);
  });
  const mean=ratios.reduce((a,b)=>a+b,0)/ratios.length;
  const mx=pL+mean*(cW-pL-pR);
  histCtx.strokeStyle='#ffb74d'; histCtx.lineWidth=1.5; histCtx.setLineDash([3,3]);
  histCtx.beginPath(); histCtx.moveTo(mx,pT); histCtx.lineTo(mx,pT+bH); histCtx.stroke();
  histCtx.setLineDash([]);
  histCtx.fillStyle='#5a5a80'; histCtx.font='9px JetBrains Mono,monospace'; histCtx.textAlign='center';
  [0,.25,.5,.75,1].forEach(v=>histCtx.fillText(v.toFixed(2),pL+v*(cW-pL-pR),cH-6));
  histCtx.textAlign='right'; histCtx.fillText(maxC,pL-2,pT+8);
}

// ── Metrics ────────────────────────────────────────────────────────────────
function updateMetrics() {
  const s=S.stats; if (!s) return;
  document.getElementById('m-tiles').textContent = s.num_tiles.toLocaleString();
  document.getElementById('m-size').textContent  = `${s.H}×${s.W}`;
  document.getElementById('m-mask').textContent  = (s.mask_fraction*100).toFixed(2)+'%';
  document.getElementById('m-cov').textContent   = s.coverage!=null?s.coverage.toFixed(4):'—';
}

// ── Overlay ────────────────────────────────────────────────────────────────
function showOverlay(v) { overlay.classList.toggle('hidden', !v); }

// ── Canvas events ──────────────────────────────────────────────────────────
function canvasToImage(cx,cy) {
  return { ix:(cx-S.offsetX)/S.scale, iy:(cy-S.offsetY)/S.scale };
}
canvas.addEventListener('mousedown', e => {
  S.dragging=true; S.dragStartX=e.clientX; S.dragStartY=e.clientY;
  S.dragOffX=S.offsetX; S.dragOffY=S.offsetY; canvas.style.cursor='grabbing';
});
canvas.addEventListener('mousemove', e => {
  const rect=canvas.getBoundingClientRect();
  const {ix,iy}=canvasToImage(e.clientX-rect.left, e.clientY-rect.top);
  document.getElementById('canvas-coords').textContent=`x=${Math.round(ix)}  y=${Math.round(iy)}`;
  if (S.dragging) {
    S.offsetX=S.dragOffX+(e.clientX-S.dragStartX);
    S.offsetY=S.dragOffY+(e.clientY-S.dragStartY);
    renderCanvas();
  }
});
canvas.addEventListener('mouseup', e => {
  if (!S.dragging) return;
  const dx=Math.abs(e.clientX-S.dragStartX), dy=Math.abs(e.clientY-S.dragStartY);
  S.dragging=false; canvas.style.cursor='crosshair';
  if (dx<4&&dy<4) {
    const rect=canvas.getBoundingClientRect();
    handleCanvasClick(e.clientX-rect.left, e.clientY-rect.top);
  }
});
canvas.addEventListener('mouseleave', ()=>{ S.dragging=false; canvas.style.cursor='crosshair'; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect=canvas.getBoundingClientRect();
  const cx=e.clientX-rect.left, cy=e.clientY-rect.top;
  const f=e.deltaY<0?1.12:1/1.12, ns=Math.max(0.05,Math.min(50,S.scale*f));
  S.offsetX=cx-(cx-S.offsetX)*(ns/S.scale);
  S.offsetY=cy-(cy-S.offsetY)*(ns/S.scale);
  S.scale=ns; renderCanvas();
}, { passive:false });
function handleCanvasClick(cx,cy) {
  // Coordinates are in full-resolution image space — tile coords live there too.
  const {ix,iy}=canvasToImage(cx,cy), cs=S.cropSize;
  for (let i=0;i<S.tiles.length;i++) {
    const [ty,tx]=S.tiles[i];
    if (ix>=tx&&ix<tx+cs&&iy>=ty&&iy<ty+cs) { selectTile(i); return; }
  }
  S.selTile=null; clearDetail(); renderCanvas();
}

// ── Sidebar controls ───────────────────────────────────────────────────────
document.getElementById('btn-prev').addEventListener('click', ()=>selectTissue(S.tidIdx-1));
document.getElementById('btn-next').addEventListener('click', ()=>selectTissue(S.tidIdx+1));
document.getElementById('btn-jump').addEventListener('click', ()=>{
  const v=parseInt(document.getElementById('jump-input').value);
  if (!isNaN(v)) selectTile(v);
});
document.getElementById('jump-input').addEventListener('keydown', e=>{
  if (e.key==='Enter') document.getElementById('btn-jump').click();
});
document.getElementById('btn-desel').addEventListener('click', ()=>{
  S.selTile=null; clearDetail(); renderCanvas(); updateTileTableSelection();
});
document.getElementById('mask-toggle').addEventListener('change', async e=>{
  S.showMask=e.target.checked;
  await reloadImage();
  if (S.selTile!==null) await loadTileDetail(S.selTile);
});

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if (e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if (e.key==='ArrowLeft' ||e.key==='h') selectTissue(S.tidIdx-1);
  if (e.key==='ArrowRight'||e.key==='l') selectTissue(S.tidIdx+1);
  if (e.key==='Escape') { S.selTile=null; clearDetail(); renderCanvas(); }
  if (e.key==='m') document.getElementById('mask-toggle').click();
  if ((e.key==='ArrowUp'  ||e.key==='k')&&S.selTile!==null) selectTile(S.selTile-1);
  if ((e.key==='ArrowDown'||e.key==='j')&&S.selTile!==null) selectTile(S.selTile+1);
});

// ── Resize ─────────────────────────────────────────────────────────────────
window.addEventListener('resize', ()=>{
  if (!S.imgBitmap) return;
  fitCanvas(); renderCanvas(); drawHistogram();
});

// ── Boot ───────────────────────────────────────────────────────────────────
init().catch(err=>{
  document.getElementById('loading-overlay').innerHTML=
    `<span style="color:#ef5350">Error: ${err.message}</span>`;
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(FRONTEND_HTML)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    parser = argparse.ArgumentParser(description="Tiling Visualizer")
    parser.add_argument("dataset_name")
    parser.add_argument("tile_size", type=int)
    parser.add_argument("modality", nargs="?", default=None)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--tile-strategy", default="default")
    parser.add_argument("--mask-only", action="store_true", help="Only render tissue masks; never load full images.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    available = discover_modalities(args.dataset_name)
    if args.modality:
        modality = args.modality
        if modality not in available:
            print(f"ERROR: Modality '{args.modality}' not found for '{args.dataset_name}'.")
            print(f"Available modalities: {available}")
            sys.exit(1)
    else:
        if not available:
            print(f"ERROR: No modalities found for '{args.dataset_name}' in {DATASET_DIR}")
            sys.exit(1)
        modality = available[0]
        if len(available) > 1:
            print(f"Multiple modalities found: {available}. Using '{modality}'.")

    STATE.dataset_name  = args.dataset_name
    STATE.modality      = modality
    STATE.crop_size     = args.tile_size
    STATE.resolution    = args.resolution
    STATE.tile_strategy = args.tile_strategy
    STATE.mask_only     = args.mask_only
    STATE.is_multiplex  = modality in MULTIPLEX_MODALITIES

    print(f"Loading dataset '{args.dataset_name}' [{modality}]…")
    STATE.dataset     = load_dataset(
        args.dataset_name,
        modality,
        args.tile_size,
        args.resolution,
        args.tile_strategy,
    )
    STATE.tile_coords = filter_tile_coords_to_dataset(
        load_tile_coords(
            args.dataset_name,
            args.resolution,
            args.tile_size,
            args.tile_strategy,
        ),
        STATE.dataset,
    )
    STATE.stats_df    = load_stats_df(
        args.dataset_name,
        args.resolution,
        args.tile_size,
        args.tile_strategy,
    )

    if not STATE.tile_coords:
        print(
            "ERROR: No tile coordinates found at "
            f"{DATASET_DIR / args.dataset_name / 'tiling' / resolution_to_dir(args.resolution) / args.tile_strategy / f'{args.tile_size}_tile_coordinates.parquet'}."
        )
        sys.exit(1)

    first_tid           = next(iter(STATE.tile_coords))
    STATE.channel_names = [] if STATE.mask_only else _get_channel_names(STATE.dataset, first_tid)
    n_ch                = len(STATE.channel_names)
    print(f"Found {len(STATE.tile_coords)} tissues, {n_ch} channels.")
    if STATE.mask_only:
        print("Running in mask-only mode.")
    elif STATE.is_multiplex:
        print(f"Channels: {STATE.channel_names[:8]}{'…' if n_ch > 8 else ''}")

    print(f"Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
