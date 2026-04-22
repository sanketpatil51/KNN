import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from Src.index_builder import load_indexes
from Src.classifier import classify_record_fast
from Src.preprocessing import batch_generate_test_embeddings


# -------------------------------------------------
# Worker function
# -------------------------------------------------
def process_row(args):

    row, desc_index, gl_index, metadata, supplier_map = args

    emb        = row["emb_desc"] if row["query_type"] == "desc" else row["emb_gl"]
    query_type = row["query_type"]

    record = {
        "supplier"  : str(row.get("supplier", "")),
        "description": str(row.get("description", "")),
        "gl"        : str(row.get("gl", "")),
        "text_desc" : str(row.get("text_desc", "")),
        "text_gl"   : str(row.get("text_gl", "")),
    }

    result = classify_record_fast(
        record          = record,
        precomputed_emb = emb,
        query_type      = query_type,
        desc_index      = desc_index,
        gl_index        = gl_index,
        metadata        = metadata,
        supplier_map    = supplier_map
    )

    return result


# -------------------------------------------------
# Main Prediction Pipeline
# -------------------------------------------------
def predict(input_file="Test_data.xlsx", output_file="classification_results.xlsx"):

    # -------------------------------
    # STEP 1 — Load indexes once
    # -------------------------------
    print("Loading FAISS indexes...")

    desc_index, gl_index, metadata, supplier_map = load_indexes()

    print("Indexes loaded successfully")

    # -------------------------------
    # STEP 2 — Load input file
    # -------------------------------
    print("Reading input file...")

    df = pd.read_excel(input_file)
    df = df.fillna("")

    # Preserve original input columns
    input_columns = df.columns.tolist()

    print(f"Total rows loaded: {len(df)}")

    # -------------------------------
    # STEP 3 — Batch generate ALL
    #          embeddings upfront
    #          (~50 API calls instead
    #          of 5000)
    # -------------------------------
    print("Generating embeddings in batches...")

    df = batch_generate_test_embeddings(df)

    print("Embeddings ready.")

    # -------------------------------
    # STEP 4 — Prepare args for
    #          parallel workers
    # -------------------------------
    rows = [row for _, row in df.iterrows()]

    args_list = [
        (row, desc_index, gl_index, metadata, supplier_map)
        for row in rows
    ]

    # -------------------------------
    # STEP 5 — Parallel classification
    #          using threads
    #          (ThreadPoolExecutor
    #          works with FAISS,
    #          ProcessPoolExecutor
    #          does not)
    # -------------------------------
    num_workers = 5

    print(f"Classifying {len(rows)} records using {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_row, args_list))

    print("Classification finished.")

    # -------------------------------
    # STEP 6 — Convert results
    # -------------------------------
    result_df = pd.DataFrame(results)

    # Prevent column name clash with input columns
    result_df = result_df.add_prefix("pred_")

    # -------------------------------
    # STEP 7 — Merge input + results
    # -------------------------------
    final_df = pd.concat(
        [df.reset_index(drop=True), result_df.reset_index(drop=True)],
        axis=1
    )

    # -------------------------------
    # STEP 8 — Restore original
    #          column order
    # -------------------------------

    # Drop embedding columns — not needed in output
    cols_to_drop = ["emb_desc", "emb_gl", "supplier_norm",
                    "desc_clean", "gl_clean", "text_desc",
                    "text_gl", "query_type"]

    final_df = final_df.drop(
        columns=[c for c in cols_to_drop if c in final_df.columns]
    )

    final_df = final_df[
        input_columns + [col for col in final_df.columns if col not in input_columns]
    ]

    # -------------------------------
    # STEP 9 — Validation
    # -------------------------------
    missing_cols = set(input_columns) - set(final_df.columns)

    if missing_cols:
        raise ValueError(f"Columns lost in pipeline: {missing_cols}")

    # -------------------------------
    # STEP 10 — Save output
    # -------------------------------
    final_df.to_excel(output_file, index=False)

    print(f"Results saved to {output_file}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    predict()