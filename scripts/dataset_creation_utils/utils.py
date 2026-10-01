
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from spora_io._config import get_datasets_dir

Image.MAX_IMAGE_PIXELS = 2_147_483_647

SPORA_DATASETS_PATH = get_datasets_dir()
ZARR_COMPRESSOR = zarr.codecs.BloscCodec(cname='lz4', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
MAX_BLOSC_BUFFER_BYTES = 2_147_483_647

NUCLEAR_RULES: list[tuple[str, str]] = [
    (r"\bDAPI.*\b", "DAPI stain"),
    (r"HOECHST", "Hoechst stain"),
    (r"\bDNA[ _-]?\d*\b", "DNA stain"),
    (r"DSDNA", "dsDNA marker"),
    (r"191\s*IR|193\s*IR|IR191|IR193|IRIDIUM", "Iridium DNA intercalator"),
    (r"HISTONE\s*-?\s*H3|HISTONEH3|\bH3\b", "Histone H3 marker"),
    (r"NUCLEAR", "Explicit nuclear label"),
    (r"DRAQ5", "DRAQ5 DNA stain"),
]

def convert_np_to_zarr(path_to_np: Path, path_to_zarr: Path):
    data = np.load(path_to_np)
    data = np.array(data, order="C")
    C, H, W = data.shape

    crop_size = 256
    shard_size = 1024 # adapted based on WSI size
    
    print(C,H,W)
    arr = zarr.create_array(
        path_to_zarr,
        dtype=data.dtype,
        zarr_format=3,
        shape=data.shape, 
        shards=(C,shard_size, shard_size), 
        chunks=(C, crop_size, crop_size),
        compressors=ZARR_COMPRESSOR
    )
    arr[:] = data
    del data, arr

def convert_zarr_to_ome_zarr(
    store_path,
    channel_names,
    image_array_path="",
    rename_to_ome_zarr=True,
):
    import shutil

    def make_channel_metadata(names, max_val=255):
        colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "FFFFFF", "FFA500"]
        return [
            {
                "active": True,
                "coefficient": 1.0,
                "color": colors[i % len(colors)],
                "family": "linear",
                "inverted": False,
                "label": str(name),
                "window": {"start": 0.0, "end": float(max_val), "min": 0.0, "max": float(max_val)},
            }
            for i, name in enumerate(names)
        ]

    def normalize_root_to_group(base_store):
        root_zarr_json = base_store / "zarr.json"
        if not root_zarr_json.exists():
            raise FileNotFoundError(f"Missing {root_zarr_json}")

        root_meta = json.loads(root_zarr_json.read_text())
        if root_meta.get("node_type") != "array":
            return

        # Move the root array to dataset path '0' so the store is a proper NGFF image group.
        array_dir = base_store / "0"
        array_dir.mkdir(exist_ok=True)

        for child in sorted(base_store.iterdir()):
            if child.name in {"zarr.json", "0", "labels"}:
                continue
            target = array_dir / child.name
            if target.exists():
                continue
            child.rename(target)

        arr_meta = root_meta
        arr_attrs = dict(arr_meta.get("attributes", {}))
        ome = arr_attrs.pop("ome", None)
        arr_attrs.pop("multiscales", None)
        arr_attrs.pop("omero", None)
        arr_meta["attributes"] = arr_attrs
        arr_meta["dimension_names"] = arr_meta.get("dimension_names", ["c", "y", "x"])
        (array_dir / "zarr.json").write_text(json.dumps(arr_meta, indent=2))

        if ome is None:
            ome = {"version": "0.5", "multiscales": [{"datasets": [{"path": "0"}], "axes": []}]}
        else:
            ome = dict(ome)
            ms = list(ome.get("multiscales", []))
            if ms:
                ds = list(ms[0].get("datasets", []))
                if ds:
                    ds[0] = dict(ds[0])
                    ds[0]["path"] = "0"
                    ms[0] = dict(ms[0])
                    ms[0]["datasets"] = ds
                ome["multiscales"] = ms

        group_meta = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": ome},
        }
        root_zarr_json.write_text(json.dumps(group_meta, indent=2))

    def transpose_rgb_yxc_to_cyx(base_store, dataset_path):
        src_path = base_store / dataset_path
        src_arr = zarr.open_array(str(src_path), mode="r")
        if len(src_arr.shape) != 3 or src_arr.shape[-1] != 3:
            return src_arr

        y, x, c = src_arr.shape
        src_chunks = src_arr.chunks or (256, 256, 3)
        dst_chunks = (c, src_chunks[0], src_chunks[1])

        tmp_dataset_path = f"{dataset_path}__cyx_tmp"
        tmp_path = base_store / tmp_dataset_path
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        dst_arr = zarr.create_array(
            str(tmp_path),
            dtype=src_arr.dtype,
            zarr_format=3,
            shape=(c, y, x),
            chunks=dst_chunks,
            shards=(c, 1024, 1024),
            compressors=ZARR_COMPRESSOR,
        )

        for row_start in range(0, y, 6144):
            row_end = min(row_start + 6144, y)
            for col_start in range(0, x, 6144):
                col_end = min(col_start + 6144, x)
                for ch in range(c):
                    dst_arr[ch, row_start:row_end, col_start:col_end] = src_arr[row_start:row_end, col_start:col_end, ch]
            
        if src_path.exists():
            shutil.rmtree(src_path)
        tmp_path.rename(src_path)
        del dst_arr, src_arr
        return zarr.open_array(str(src_path), mode="r")

    store_path = Path(store_path)
    normalize_root_to_group(store_path)
    print(store_path)
    root = zarr.open_group(str(store_path), mode="a")
    if image_array_path:
        arr = root[image_array_path]
        dataset_path = image_array_path
    else:
        dataset_path = "0" if "0" in root else list(root)[0]
        arr = root[dataset_path]

    shape = arr.shape
    dtype = arr.dtype
    if len(shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {shape}")

    dtype = arr.dtype
    if dtype.fields is not None:
        raise ValueError(f"Structured dtype '{dtype}' not supported. Expected numeric dtype like u1/f4")

    # Convert RGB Y,X,C into C,Y,X so channel axis is always first in OME-Zarr arrays.
    is_rgb_channels_last = (shape[-1] == 3) and (shape[0] != 3)
    if is_rgb_channels_last:
        arr = transpose_rgb_yxc_to_cyx(store_path, dataset_path)
        shape = arr.shape
        n_channels = shape[0]
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        dimension_names = ["c", "y", "x"]
        if len(channel_names) != 3:
            channel_names = ["R", "G", "B"]
    else:
        n_channels = shape[0]
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        dimension_names = ["c", "y", "x"]

    if len(channel_names) != n_channels:
        raise ValueError(f"Need {n_channels} channel names, got {len(channel_names)}")

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "name": store_path.stem,
                "axes": axes,
                "datasets": [
                    {
                        "path": dataset_path,
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                        ],
                    }
                ],
            }
        ],
        "omero": {
            "id": 1,
            "name": store_path.stem,
            "version": "0.5",
            "channels": make_channel_metadata(channel_names, max_val=255 if dtype == np.uint16 else 255),
            "rdefs": {"model": "color", "defaultT": 0, "defaultZ": 0},
        },
    }

    array_zarr_json = store_path / dataset_path / "zarr.json"
    created_subdir_metadata = []
    del arr
    if array_zarr_json.exists():
        array_meta = json.loads(array_zarr_json.read_text())
        array_meta["dimension_names"] = dimension_names
        array_zarr_json.write_text(json.dumps(array_meta, indent=2))

    if rename_to_ome_zarr and store_path.suffix == ".zarr":
        new_path = store_path.with_suffix(".ome.zarr")
        if new_path.exists():
            raise FileExistsError(f"{new_path} exists")
        store_path.rename(new_path)
        return new_path, created_subdir_metadata

    return store_path, created_subdir_metadata