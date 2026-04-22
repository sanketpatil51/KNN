import re
import pandas as pd
from typing import Dict, List, Any

from .embeddings import generate_embeddings


def normalize_supplier(name: Any) -> str:

    if pd.isna(name) or name is None:
        return ""

    name = str(name).lower()

    name = re.sub(r"\b(ltd|limited|inc|incorporated|gmbh|corp|corporation|company|co|llc|plc)\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def clean_description(text: Any) -> str:

    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"\binv[- ]?\d+\b", " ", text)
    text = re.sub(r"\bpo[- ]?\d+\b", " ", text)
    text = re.sub(r"\bgrn[- ]?\d+\b", " ", text)
    text = re.sub(r"\bref[- ]?\d+\b", " ", text)

    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_historical_data(df: pd.DataFrame) -> pd.DataFrame:

    required_cols = {"supplier", "description", "gl", "category"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["supplier_norm"] = df["supplier"].apply(normalize_supplier)
    df["desc_clean"] = df["description"].apply(clean_description)
    df["gl_clean"] = df["gl"].apply(clean_description)

    df["text_desc"] = (df["supplier_norm"] + " " + df["desc_clean"]).str.strip()
    df["text_gl"] = (df["supplier_norm"] + " " + df["gl_clean"]).str.strip()

    return df


def build_supplier_map(df: pd.DataFrame) -> Dict[str, List[int]]:

    supplier_map: Dict[str, List[int]] = {}

    for i, supplier in enumerate(df["supplier_norm"].tolist()):
        supplier_map.setdefault(supplier, []).append(i)

    return supplier_map


def batch_generate_test_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans all test records and generates embeddings upfront in batches.
    Call this ONCE before your classification loop.
    Returns the same DataFrame with added columns ready for classification.

    Expected input columns: supplier, description, gl
    New columns added: supplier_norm, desc_clean, gl_clean,
                       text_desc, text_gl, query_type, emb_desc, emb_gl
    """

    required_cols = {"supplier", "description", "gl"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # --------------------------------------------------------
    # STEP 1 — CLEAN ALL TEXT
    # --------------------------------------------------------

    print("Step 1/4 — Cleaning supplier names and descriptions...")

    df["supplier_norm"] = df["supplier"].apply(normalize_supplier)
    df["desc_clean"]    = df["description"].apply(clean_description)
    df["gl_clean"]      = df["gl"].apply(clean_description)

    df["text_desc"] = (df["supplier_norm"] + " " + df["desc_clean"]).str.strip()
    df["text_gl"]   = (df["supplier_norm"] + " " + df["gl_clean"]).str.strip()

    # --------------------------------------------------------
    # STEP 2 — DETERMINE QUERY TYPE PER RECORD
    # --------------------------------------------------------

    print("Step 2/4 — Determining query type per record...")

    # If description is not empty use desc, otherwise fall back to gl
    df["query_type"] = df["desc_clean"].apply(
        lambda x: "desc" if str(x).strip() != "" else "gl"
    )

    desc_count = (df["query_type"] == "desc").sum()
    gl_count   = (df["query_type"] == "gl").sum()

    print(f"           {desc_count} records will use description  |  {gl_count} records will use GL")

    # --------------------------------------------------------
    # STEP 3 — BATCH GENERATE DESC EMBEDDINGS
    # --------------------------------------------------------

    print(f"Step 3/4 — Generating DESC embeddings for {len(df)} records (batch size 100)...")

    desc_embeddings = generate_embeddings(df["text_desc"].tolist(), batch_size=100)
    df["emb_desc"]  = list(desc_embeddings)

    print(f"           DESC embeddings done.")

    # --------------------------------------------------------
    # STEP 4 — BATCH GENERATE GL EMBEDDINGS
    # --------------------------------------------------------

    print(f"Step 4/4 — Generating GL embeddings for {len(df)} records (batch size 100)...")

    gl_embeddings = generate_embeddings(df["text_gl"].tolist(), batch_size=100)
    df["emb_gl"]  = list(gl_embeddings)

    print(f"           GL embeddings done.")

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print(f"All embeddings ready. {len(df)} records prepared for classification.")

    return df