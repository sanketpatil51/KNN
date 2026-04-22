import numpy as np
import faiss
import pandas as pd
from typing import Tuple, List

from .preprocessing import normalize_supplier, clean_description
from .config import TOP_K_SEARCH


def create_query_text(supplier, description, gl) -> Tuple[str, str]:

    supplier_norm = normalize_supplier(supplier)
    description_clean = clean_description(description)
    gl_clean = clean_description(gl)

    if description_clean != "":
        query_text = f"{supplier_norm} {description_clean}".strip()
        query_type = "desc"
    else:
        query_text = f"{supplier_norm} {gl_clean}".strip()
        query_type = "gl"

    return query_text, query_type


def search_within_supplier(
    query_emb: np.ndarray,
    supplier: str,
    index_type: str,
    metadata: pd.DataFrame,
    supplier_map,
    desc_index,
    gl_index,
    top_k: int = TOP_K_SEARCH
):

    supplier_norm = normalize_supplier(supplier)

    if supplier_norm not in supplier_map:
        raise ValueError("Supplier not found in supplier_map")

    ids = supplier_map[supplier_norm]

    if index_type == "desc":
        vectors = np.vstack(metadata.loc[ids, "emb_desc"].values).astype("float32")

    elif index_type == "gl":
        vectors = np.vstack(metadata.loc[ids, "emb_gl"].values).astype("float32")

    else:
        raise ValueError("index_type must be 'desc' or 'gl'")

    temp_index = faiss.IndexFlatIP(vectors.shape[1])
    temp_index.add(vectors)

    D, I = temp_index.search(
        np.array([query_emb]).astype("float32"),
        min(top_k, len(ids))
    )

    global_ids = [ids[i] for i in I[0] if i != -1]

    return D[0], global_ids, "supplier_found"


def global_search(
    query_emb: np.ndarray,
    index_type: str,
    desc_index,
    gl_index,
    top_k: int = TOP_K_SEARCH
):

    if index_type == "desc":
        D, I = desc_index.search(np.array([query_emb]).astype("float32"), top_k)

    elif index_type == "gl":
        D, I = gl_index.search(np.array([query_emb]).astype("float32"), top_k)

    else:
        raise ValueError("index_type must be 'desc' or 'gl'")

    return D[0], I[0], "supplier_new"


def supplier_matching_and_search(
    query_emb: np.ndarray,
    supplier: str,
    query_type: str,
    metadata: pd.DataFrame,
    supplier_map,
    desc_index,
    gl_index
):

    supplier_norm = normalize_supplier(supplier)

    if supplier_norm in supplier_map:

        D, ids, branch = search_within_supplier(
            query_emb=query_emb,
            supplier=supplier_norm,
            index_type=query_type,
            metadata=metadata,
            supplier_map=supplier_map,
            desc_index=desc_index,
            gl_index=gl_index
        )

        return D, ids, branch

    else:

        D, I, branch = global_search(
            query_emb=query_emb,
            index_type=query_type,
            desc_index=desc_index,
            gl_index=gl_index
        )

        ids = [int(i) for i in I if i != -1]

        return D, ids, branch