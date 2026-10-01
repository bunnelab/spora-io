import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize
from tqdm import tqdm

from scripts.dataset_creation_utils.utils_mx import (create_folderstructure,
                                                      hash_patient,
                                                      save_np_as_npz,
                                                      save_np_as_zarr)
from scripts.dataset_creation_utils.utils_he import (
    get_mask, save_rgb_as_zarr, segmentation_model_factory)
from spora_io._config import get_datasets_dir

"""
This scripts creates the images for the hoefflin2026spatial datasets.
For this, first, download the raw images from: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2840 and rename the 'Files' folder to 'hoefflin2026spatial_raw'.
Secondly, download the samples metadata from https://zenodo.org/records/18378034, export the xlsx file to csv and rename it to 'sample_metadata.csv' and place it in the 'hoefflin2026spatial_raw' folder.
After executing this script, proceed with the next steps of the dataset creation pipeline.md file to create the final dataset.
"""

RAW_DATA_PATH = "examples/hoefflin2026spatial_raw"
SPORA_DATASETS_PATH = get_datasets_dir()
DS_NAME = "hoefflin2026spatial"
CODEX_MPP = 0.5
HE_MPP = 0.5


def get_channel_names(mx_path):
    file = tifffile.TiffFile(mx_path)
    tree = ET.fromstring(file.ome_metadata)
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    channels = tree.findall(".//ome:Channel", ns)
    channel_names = [channel.attrib["Name"] for channel in channels]
    return channel_names

def resize_stack(img, out_hw, interp=cv2.INTER_AREA, workers=None):
    """img: (Z, H, W) int16.  out_hw: (new_H, new_W).  Returns (Z, new_H, new_W)."""
    Z = img.shape[0]
    out = np.empty((Z, *out_hw), dtype=img.dtype)      # preallocate, stays int16
    def _one(i):
        out[i] = cv2.resize(img[i], (out_hw[1], out_hw[0]), interpolation=interp)
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(_one, range(Z)), total=Z, desc="Resizing stack"))
    return out



# all tissues have the same channels with the same order:
channels = ['DAPI', 'CD69', 'CD45', 'CD3', 'Ki67', 'CD90', 'CD11c', 'MHCII', 'CD8', 'CD4', 'CD31', 'FN1', 'SOX10', 'PDPN', 'VIM', 'IDHm', 'CD279', 'GFAP', 'EGFR', 'DCX', 'SOX4', 'CD14', 'NeuN', 'AQP4', 'OLIG2', 'GLUT1', 'PDGFRA', 'CA9', 'CHI3L1', 'BCAN', 'CD44', 'NDRG1', 'GAP43', 'CD163', 'APOE', 'APOD', 'SOX2', 'SNAP25', 'MT1MMP', 'S100B', 'MAP2', 'aSMA', 'CD19', 'CD206', 'IL1B', 'TNC', 'CD83', 'CSPG5', 'CD24', 'SPARC', 'NG2', 'STMN1', 'CXCR4', 'NCAN', 'TUBB3', 'VCAN', 'POSTN', 'P2Y12', 'CD15', 'COL1', 'HPLN1', 'HSPG2', 'TNR', 'DCXv2', 'SOX9', 'FLT1', 'PTPRZ1', 'ATRX', 'ACAN', 'PLP1']

uniprot_mapping = {
 'DAPI': None,
 'CD69': 'Q07108',
 'CD45': 'P08575',
 'CD3': 'P07766',
 'Ki67': 'P46013',
 'CD90': 'P04216',
 'CD11c': 'P20702',
 'MHCII': 'P01903',
 'CD8': 'P01732',
 'CD4': 'P01730',
 'CD31': 'P16284',
 'FN1': 'P02751',
 'SOX10': 'P56693',
 'PDPN': 'Q86YL7',
 'VIM': 'P08670',
 'IDHm': None,
 'CD279': 'Q15116',
 'GFAP': 'P14136',
 'EGFR': 'P00533',
 'DCX': 'O43602',
 'SOX4': 'Q06945',
 'CD14': 'P08571',
 'NeuN': 'A6NFN3',
 'AQP4': 'P55087',
 'OLIG2': 'Q13516',
 'GLUT1': 'P11166',
 'PDGFRA': 'P16234',
 'CA9': 'Q16790',
 'CHI3L1': 'P36222',
 'BCAN': 'Q96GW7',
 'CD44': 'P16070',
 'NDRG1': '	Q92597',
 'GAP43': 'P17677',
 'CD163': 'Q86VB7',
 'APOE': 'P02649',
 'APOD': 'P05090',
 'SOX2': 'P48431',
 'SNAP25': 'P60880',
 'MT1MMP': 'P50281',
 'S100B': 'P04271',
 'MAP2': 'P11137',
 'aSMA': 'P62736',
 'CD19': 'P15391',
 'CD206': 'P22897',
 'IL1B': None,
 'TNC': 'P24821',
 'CD83': 'Q01151',
 'CSPG5': 'O95196',
 'CD24': 'P25063',
 'SPARC': 'P09486',
 'NG2': 'Q99942',
 'STMN1': 'P16949',
 'CXCR4': 'P61073',
 'NCAN': None,
 'TUBB3': 'Q13509',
 'VCAN': 'P13611',
 'POSTN': 'Q15063',
 'P2Y12': 'Q9H244',
 'CD15': 'P22083',
 'COL1': 'P02452',
 'HPLN1': 'P10915',
 'HSPG2': 'P98160',
 'TNR': 'Q92752',
 'DCXv2': None,
 'SOX9': 'P48436',
 'FLT1': None,
 'PTPRZ1': 'P23471',
 'ATRX': 'P46100',
 'ACAN': 'P16112',
 'PLP1': 'P60201'
}

if __name__ == "__main__":
    modality_mpp = {
        "codex": [1.0, CODEX_MPP],
        "he": [1.0, HE_MPP]
    }
    sample_metadata = pd.read_csv(f"{RAW_DATA_PATH}/sample_metadata.csv")

    # Create folder structure
    create_folderstructure(modality_mpp, DS_NAME)
    (SPORA_DATASETS_PATH / DS_NAME / "segmentations" / "1_0mpp" / "tissue_masks").mkdir(exist_ok=True, mode=0o775, parents=True)
    (SPORA_DATASETS_PATH / DS_NAME / "segmentations" / "0_5mpp" / "tissue_masks").mkdir(exist_ok=True, mode=0o775, parents=True)

    all_mx_files = glob(f"{RAW_DATA_PATH}/**/*[0-9].ome.tif", recursive=True)
    all_mx_files = [mx_file for mx_file in all_mx_files if "HE" not in mx_file]  # Exclude HE files
    all_mx_files = sorted(all_mx_files)

    # Sanity check to ensure all channels have the right names and order across all Codex files
    for mx_file in tqdm(all_mx_files, desc="Processing MX files"):
        try:
            channel_names = get_channel_names(mx_file)
            assert channel_names == channels, f"Channel names do not match for {mx_file}"
        except:
            print(f"Failed to process {mx_file}")
    channels_df = pd.DataFrame({"channel_name": channels, "uniprot_id": [uniprot_mapping[ch] for ch in channels], "qc_pass" : True, "description" : None, "is_nuclear_marker": False})
    channels_df.loc[channels_df["channel_name"] == "DAPI", "is_nuclear_marker"] = True
    channels_df.to_parquet(SPORA_DATASETS_PATH / DS_NAME / "codex" / "channels.parquet", index=False)

    tissue_ann = pd.DataFrame(columns=["dataset", "patient_id", "specimen_id", "tissue_id", "original_patient_identifier", "modality", "alignment"])
    tissues_per_patient = {}

    segmentation_model = segmentation_model_factory("hest", device=f"cuda:0")
    segmentation_model.eval()

    for mx_file in tqdm(all_mx_files, desc="Processing MX files"):

        sample_id = Path(mx_file).name.replace(".ome.tif", "")
        patient_name = sample_metadata.query(f"SampleID == '{sample_id}'")["PatientID"].values[0]
        patient_id = hash_patient(patient_name)
        specimen_id = tissues_per_patient.get(patient_id, 0)
        tissue_id = f"{DS_NAME}_{patient_id}_{specimen_id:04d}"
        tissues_per_patient[patient_id] = specimen_id + 1
        tissue_ann = pd.concat([tissue_ann, pd.DataFrame([{
            "dataset": DS_NAME,
            "patient_id": patient_id,
            "specimen_id": specimen_id,
            "tissue_id": tissue_id,
            "original_patient_identifier": patient_name,
            "modality": "codex",
            "alignment": "exact",
        }])], ignore_index=True)
        tissue_ann = pd.concat([tissue_ann, pd.DataFrame([{
            "dataset": DS_NAME,
            "patient_id": patient_id,
            "specimen_id": specimen_id,
            "tissue_id": tissue_id,
            "original_patient_identifier": patient_name,
            "modality": "he",
            "alignment": "exact",
        }])], ignore_index=True)

        if os.path.exists(SPORA_DATASETS_PATH / DS_NAME / "he" / "1_0mpp" / "images" / f"{tissue_id}.zarr"):
            print(f"Skipping {tissue_id} as it already exists.")
            continue
        if not "hoefflin2026spatial_qdcllysp_0000" == tissue_id:
            print(f"Skipping {tissue_id} as it is not the target tissue for processing.")
            continue
        # MX
        codex_05mpp = tifffile.imread(mx_file)
        print("Starting write of codex stack to zarr to file:", SPORA_DATASETS_PATH / DS_NAME / "codex" / "0_5mpp" / "images" / f"{tissue_id}.zarr")
        save_np_as_zarr(codex_05mpp, SPORA_DATASETS_PATH / DS_NAME / "codex" / "0_5mpp" / "images" / f"{tissue_id}.zarr")
        print("Starting resize of codex stack to 1mpp...")
        codex_1mpp = resize_stack(codex_05mpp, (codex_05mpp.shape[1] // 2, codex_05mpp.shape[2] // 2), interp=cv2.INTER_AREA)
        del codex_05mpp
        # codex_1mpp = resize(codex_05mpp, (codex_05mpp.shape[0] // 2, codex_05mpp.shape[1] // 2), anti_aliasing=True)
        print("Starting write of codex stack to zarr to file:", SPORA_DATASETS_PATH / DS_NAME / "codex" / "1_0mpp" / "images" / f"{tissue_id}.zarr")
        save_np_as_zarr(codex_1mpp, SPORA_DATASETS_PATH / DS_NAME / "codex" / "1_0mpp" / "images" / f"{tissue_id}.zarr")
        del codex_1mpp

        # HE
        print("Starting read of HE image...")
        he_05mpp = tifffile.imread(mx_file.replace(".ome.tif", "_HE_warped_to_DAPI.ome.tif"))
        print("Saving HE 0_5mpp image...")
        save_rgb_as_zarr(he_05mpp, SPORA_DATASETS_PATH / DS_NAME / "he" / "0_5mpp" / "images" / f"{tissue_id}.zarr")
        he_1mpp = cv2.resize(he_05mpp, (he_05mpp.shape[1] // 2, he_05mpp.shape[0] // 2), interpolation=cv2.INTER_AREA)
        print("Saving HE 1_0mpp image...")
        save_rgb_as_zarr(he_1mpp, SPORA_DATASETS_PATH / DS_NAME / "he" / "1_0mpp" / "images" / f"{tissue_id}.zarr")
        del he_1mpp, he_05mpp

        # Tissue Mask
        print("Starting tissue segmentation...")
        mask_05mpp = get_mask(he_05mpp[::5, ::5], segmentation_model, return_largest_mask_only=False)
        mask_05mpp = resize(mask_05mpp, (he_05mpp.shape[0], he_05mpp.shape[1]), order=0).astype(bool)
        print("Saving tissue mask to npz...")
        save_np_as_npz(mask_05mpp, SPORA_DATASETS_PATH / DS_NAME / "segmentations" / "0_5mpp" / "tissue_masks" / f"{tissue_id}.npz")
        mask_1mpp = resize(mask_05mpp, (mask_05mpp.shape[0] // 2, mask_05mpp.shape[1] // 2), order=0).astype(bool)
        save_np_as_npz(mask_1mpp, SPORA_DATASETS_PATH / DS_NAME / "segmentations" / "1_0mpp" / "tissue_masks" / f"{tissue_id}.npz")

    
    tissue_ann.to_parquet(SPORA_DATASETS_PATH / DS_NAME / "metadata" / "tissues.parquet", index=False)
    channels_per_tissue = pd.DataFrame(columns=["tissue_id"] + channels)
    channels_per_tissue["tissue_id"] = tissue_ann["tissue_id"].unique()
    channels_per_tissue.set_index("tissue_id", inplace=True)
    channels_per_tissue.loc[:, :] = True
    channels_per_tissue.to_parquet(SPORA_DATASETS_PATH / DS_NAME / "codex" / "channels_per_tissue.parquet", index=True)