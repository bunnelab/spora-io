import argparse
import json
import shutil
import zarr
from scripts.dataset_creation_utils.utils import convert_zarr_to_ome_zarr, ZARR_COMPRESSOR
from spora_io._config import get_datasets_dir
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

spora_dir = get_datasets_dir()

def convert_all_images_spmodality(modality, V2_NAME):
    zarr_files = glob(str(spora_dir / V2_NAME / modality / "*mpp" / "images" / "*.zarr"))
    zarr_files = [f for f in zarr_files if not f.endswith(".ome.zarr")]

    if len(zarr_files) > 0:
        print(f"Converting {len(zarr_files)} Zarr files for modality {modality} in dataset {V2_NAME}...")
        channels_per_tissue = pd.read_parquet(spora_dir / V2_NAME / modality / "channels_per_tissue.parquet")
        if "tissue_id" not in channels_per_tissue.columns:
            channels_per_tissue["tissue_id"] = channels_per_tissue.index.astype(str)
        # assert len(set(channels_per_tissue.index)) == len(set(channels_per_tissue["tissue_id"])), "Duplicate tissue_ids found in channels_per_tissue.parquet"
        channels_per_tissue = channels_per_tissue.set_index("tissue_id")
        channel_names = channels_per_tissue.columns
    
    for zarr_file in tqdm(zarr_files, desc=f"Converting Zarr files for {V2_NAME} - {modality}"):
        tissue_id = Path(zarr_file).name.split(".")[0]
        try:
            channel_names_tid = channel_names[channels_per_tissue.loc[tissue_id].values]
            convert_zarr_to_ome_zarr(
                store_path=zarr_file,
                channel_names=channel_names_tid,
                rename_to_ome_zarr=True,
            )
        except Exception as e:
            print(f"Error converting {zarr_file}: {e}")

def convert_all_images_he(V2_NAME):
    zarr_files = glob(str(spora_dir / V2_NAME / "he" / "*mpp" / "images" / "*.zarr"))
    zarr_files = [f for f in zarr_files if not f.endswith(".ome.zarr")]

    if len(zarr_files) > 0:
        print(f"Converting {len(zarr_files)} Zarr files for HE modality in dataset {V2_NAME}...")
    for zarr_file in tqdm(zarr_files, desc=f"Converting Zarr files for {V2_NAME} - HE"):
        if os.path.exists(zarr_file.replace(".zarr", ".ome.zarr")):
            print(f"OME-Zarr file already exists for {zarr_file}, skipping conversion.")
            continue
        convert_zarr_to_ome_zarr(
            store_path=zarr_file,
            channel_names=["R", "G", "B"],
            rename_to_ome_zarr=True,
        )

def convert_all_images_ihc(V2_NAME):
    zarr_files = glob(str(spora_dir / V2_NAME / "ihc" / "ihc_*" / "*mpp" / "images" / "*.zarr"))
    zarr_files = [f for f in zarr_files if not f.endswith(".ome.zarr")]

    if len(zarr_files) > 0:
        print(f"Converting {len(zarr_files)} Zarr files for IHC modality in dataset {V2_NAME}...")
    for zarr_file in tqdm(zarr_files, desc=f"Converting Zarr files for {V2_NAME} - IHC"):
        if os.path.exists(zarr_file.replace(".zarr", ".ome.zarr")):
            print(f"OME-Zarr file already exists for {zarr_file}, skipping conversion.")
            continue
        convert_zarr_to_ome_zarr(
            store_path=zarr_file,
            channel_names=["R", "G", "B"],
            rename_to_ome_zarr=True,
        )

def _has_shards(ome_zarr_path: str) -> bool:
    zjson = Path(ome_zarr_path) / "0" / "zarr.json"
    if not zjson.exists():
        return False
    try:
        meta = json.loads(zjson.read_text())
    except Exception:
        return False
    return "shards" in meta

def _reshard_ome_zarr(ome_zarr_path: str):
    base = Path(ome_zarr_path)
    dataset = "0"
    src_path = base / dataset
    if not src_path.exists():
        return
    src_arr = zarr.open_array(str(src_path), mode="r")
    try:
        if len(src_arr.shape) != 3:
            return
        c, y, x = src_arr.shape
        src_chunks = src_arr.chunks or (c, 256, 256)

        tmp_dataset = f"{dataset}__reshard_tmp"
        tmp_path = base / tmp_dataset
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        dst_arr = zarr.create_array(
            str(tmp_path),
            dtype=src_arr.dtype,
            zarr_format=3,
            shape=(c, y, x),
            shards=(c, 1024, 1024),
            chunks=src_chunks,
            compressors=ZARR_COMPRESSOR
        )

        # Preserve dimension names so the reshared array is still readable as dataset 0.
        try:
            dimension_names = ["c", "y", "x"]
            tmp_zarr_json = tmp_path / "zarr.json"
            if tmp_zarr_json.exists():
                tmp_meta = json.loads(tmp_zarr_json.read_text())
                tmp_meta["dimension_names"] = dimension_names
                tmp_zarr_json.write_text(json.dumps(tmp_meta, indent=2))
        except Exception as e:
            print(f"Warning: could not preserve dimension_names for {ome_zarr_path}: {e}")



        # copy in large tiles to avoid excessive small IO
        tile = 6144
        for row_start in range(0, y, tile):
            row_end = min(row_start + tile, y)
            for col_start in range(0, x, tile):
                col_end = min(col_start + tile, x)
                dst_arr[:, row_start:row_end, col_start:col_end] = src_arr[:, row_start:row_end, col_start:col_end]

        # replace original dataset with reshaped one
        if src_path.exists():
            shutil.rmtree(src_path)
        tmp_path.rename(src_path)
        del dst_arr, src_arr

    except Exception as e:
        try:
            del src_arr
        except Exception:
            pass
        print(f"Error resharing {ome_zarr_path}: {e}")

def check_and_reshard_dataset(V2_NAME: str, modality: str):
    patterns = [
        spora_dir / V2_NAME / modality / "**" / "images" / "*.ome.zarr",
    ]
    zarr_paths = []
    for p in patterns:
        zarr_paths += glob(str(p), recursive=True)

    for zp in tqdm(sorted(zarr_paths), desc=f"Checking shards for {V2_NAME}"):
        try:
            if not _has_shards(zp):
                print(f"No shards found in {zp}/0/zarr.json — re-sharding...")
                _reshard_ome_zarr(zp)
            else:
                # already has shards
                continue
        except Exception as e:
            print(f"Error while checking/resharding {zp}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to create tissue masks for")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    print("Processing dataset:", dataset_name)
    convert_all_images_spmodality("mibi", dataset_name)
    convert_all_images_spmodality("imc", dataset_name)
    convert_all_images_spmodality("codex", dataset_name)
    convert_all_images_spmodality("cycif", dataset_name)
    convert_all_images_spmodality("if", dataset_name)
    convert_all_images_he(dataset_name)
    convert_all_images_ihc(dataset_name)
    # for V2_NAME in V2_NAMES:
    #     check_and_reshard_dataset(V2_NAME, "he")
    #     check_and_reshard_dataset(V2_NAME, "ihc")