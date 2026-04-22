import faiss
import pickle
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any

from .config import *
from .preprocessing import *
from .embeddings import *


def build_indexes(df: pd.DataFrame) -> None:

    df = prepare_historical_data(df)

    desc_embeddings = generate_embeddings(df["text_desc"].tolist())
    gl_embeddings = generate_embeddings(df["text_gl"].tolist())

    dim = desc_embeddings.shape[1]

    desc_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    gl_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    desc_index.add(desc_embeddings)
    gl_index.add(gl_embeddings)

    supplier_map = build_supplier_map(df)

    faiss.write_index(desc_index, DESC_INDEX_FILE)
    faiss.write_index(gl_index, GL_INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(df, f)

    with open(SUPPLIER_MAP_FILE, "wb") as f:
        pickle.dump(supplier_map, f)

    print("Historical data processed and both FAISS indexes created successfully.")


def load_indexes() -> Tuple:

    desc_index = faiss.read_index(DESC_INDEX_FILE)
    gl_index = faiss.read_index(GL_INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    with open(SUPPLIER_MAP_FILE, "rb") as f:
        supplier_map = pickle.load(f)

    return desc_index, gl_index, metadata, supplier_map


def attach_embeddings_to_metadata(metadata, desc_index, gl_index):

    metadata = metadata.copy()

    n_rows = len(metadata)

    desc_vectors = desc_index.reconstruct_batch(list(range(n_rows)))
    gl_vectors = gl_index.reconstruct_batch(list(range(n_rows)))

    metadata["emb_desc"] = list(desc_vectors)
    metadata["emb_gl"] = list(gl_vectors)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return metadata


def build_indexes_with_embeddings(df: pd.DataFrame) -> None:

    df = prepare_historical_data(df)

    desc_embeddings = generate_embeddings(df["text_desc"].tolist())
    gl_embeddings = generate_embeddings(df["text_gl"].tolist())

    df["emb_desc"] = list(desc_embeddings)
    df["emb_gl"] = list(gl_embeddings)

    dim = desc_embeddings.shape[1]

    desc_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    gl_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    desc_index.add(desc_embeddings)
    gl_index.add(gl_embeddings)

    supplier_map = build_supplier_map(df)

    faiss.write_index(desc_index, DESC_INDEX_FILE)
    faiss.write_index(gl_index, GL_INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(df, f)

    with open(SUPPLIER_MAP_FILE, "wb") as f:
        pickle.dump(supplier_map, f)

    print("Both FAISS indexes saved successfully.")


def update_indexes(record: Dict[str, Any],
                   desc_index,
                   gl_index,
                   metadata,
                   supplier_map):

    supplier = record.get("supplier", "")
    description = record.get("description", "")
    gl = record.get("gl", "")
    category = record.get("category", None)

    supplier_norm = normalize_supplier(supplier)
    desc_clean = clean_description(description)
    gl_clean = clean_description(gl)

    text_desc = f"{supplier_norm} {desc_clean}".strip()
    text_gl = f"{supplier_norm} {gl_clean}".strip()

    emb_desc = generate_single_embedding(text_desc)
    emb_gl = generate_single_embedding(text_gl)

    desc_index.add(np.array([emb_desc]).astype("float32"))
    gl_index.add(np.array([emb_gl]).astype("float32"))

    new_row = {
        "supplier": supplier,
        "description": description,
        "gl": gl,
        "category": category,
        "supplier_norm": supplier_norm,
        "desc_clean": desc_clean,
        "gl_clean": gl_clean,
        "text_desc": text_desc,
        "text_gl": text_gl,
        "emb_desc": emb_desc,
        "emb_gl": emb_gl,
    }

    new_id = len(metadata)
    metadata.loc[new_id] = new_row

    supplier_map.setdefault(supplier_norm, []).append(new_id)

    faiss.write_index(desc_index, DESC_INDEX_FILE)
    faiss.write_index(gl_index, GL_INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    with open(SUPPLIER_MAP_FILE, "wb") as f:
        pickle.dump(supplier_map, f)

    return desc_index, gl_index, metadata, supplier_map