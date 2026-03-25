import torch
import os
import sys
import pandas as pd

def load_esm_marker_embedding_dict(embedding_dir):
    embedding_dict = {}
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Could not find embedding_dir {embedding_dir}")
    files = os.listdir(embedding_dir)
    files = list(sorted(files))
    index = 0
    for file in files:
        if file.endswith(".pt"):
            protein_id = file.removesuffix(".pt")
            embedding_dict[protein_id] = index
            index += 1
    return embedding_dict

def load_esm_marker_embeddings(embedding_dir):
    if not os.path.exists(embedding_dir):
        raise ValueError(f"Could not find embedding_dir {embedding_dir}")
    files = os.listdir(embedding_dir)
    files = sorted(files)
    embeddings = []
    for file in files:
        if file.endswith(".pt"):
            embeddings.append(torch.load(os.path.join(embedding_dir, file)))
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings