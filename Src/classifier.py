import numpy as np
from typing import Dict, Any

from .search import create_query_text, supplier_matching_and_search
from .embeddings import generate_single_embedding
from .knn_logic import (
    radius_filter,
    dynamic_k,
    voting,
    neighbour_category_check,
    rerank,
    confidence_score
)
from .decision import decision_layer
from .config import SIMILARITY_THRESHOLD


def classify_record(
    record: Dict[str, Any],
    desc_index,
    gl_index,
    metadata,
    supplier_map,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:

    supplier = record.get("supplier", "")
    description = record.get("description", "")
    gl = record.get("gl", "")

    # --------------------------------------------------------
    # CREATE QUERY
    # --------------------------------------------------------

    query_text, query_type = create_query_text(
        supplier,
        description,
        gl
    )

    # --------------------------------------------------------
    # GENERATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_emb = generate_single_embedding(query_text)

    # --------------------------------------------------------
    # SUPPLIER MATCHING + SEARCH
    # --------------------------------------------------------

    distances, ids, supplier_branch = supplier_matching_and_search(
        query_emb=query_emb,
        supplier=supplier,
        query_type=query_type,
        metadata=metadata,
        supplier_map=supplier_map,
        desc_index=desc_index,
        gl_index=gl_index
    )

    # --------------------------------------------------------
    # RADIUS FILTERING
    # --------------------------------------------------------

    neighbours = radius_filter(
        distances,
        ids,
        threshold=similarity_threshold
    )

    # --------------------------------------------------------
    # FALLBACK IF NOTHING FOUND
    # --------------------------------------------------------

    if len(neighbours) == 0:

        fallback_pairs = []

        for sim, idx in zip(distances, ids):

            if idx == -1:
                continue

            fallback_pairs.append((int(idx), float(sim)))

        neighbours = fallback_pairs[:5]

    # --------------------------------------------------------
    # DYNAMIC K
    # --------------------------------------------------------

    neighbours = dynamic_k(neighbours)

    if len(neighbours) == 0:

        return {
            "predicted_category": None,
            "confidence_score": 0.0,
            "decision": "MANUAL",
            "query_type": query_type,
            "supplier_branch": supplier_branch,
            "neighbours": [],
            "category_counter": {},
            "query_text": query_text,
        }

    # --------------------------------------------------------
    # PERFECT MATCH CHECK
    # --------------------------------------------------------

    for idx, sim in neighbours:

        if sim >= 0.9999:

            category = metadata.iloc[idx]["category"]

            return {
                "predicted_category": category,
                "confidence_score": 1.0,
                "decision": "AUTO",
                "query_type": query_type,
                "supplier_branch": supplier_branch,
                "neighbours": neighbours,
                "category_counter": {category: 1},
                "query_text": query_text,
            }

    # --------------------------------------------------------
    # KNN VOTING
    # --------------------------------------------------------

    counter = voting(neighbours, metadata)

    # --------------------------------------------------------
    # CATEGORY CHECK
    # --------------------------------------------------------

    check = neighbour_category_check(counter)

    # --------------------------------------------------------
    # CONFLICT RESOLUTION
    # --------------------------------------------------------

    if check == "conflict":

        neighbours = rerank(
            query_emb,
            neighbours,
            metadata,
            query_type
        )

        neighbours = dynamic_k(neighbours)

        counter = voting(neighbours, metadata)

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    predicted_category = counter.most_common(1)[0][0]

    # --------------------------------------------------------
    # CONFIDENCE SCORE
    # --------------------------------------------------------

    score = confidence_score(counter, neighbours)

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = decision_layer(score)

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "predicted_category": predicted_category,
        "confidence_score": score,
        "decision": decision,
        "query_type": query_type,
        "supplier_branch": supplier_branch,
        "neighbours": neighbours,
        "category_counter": dict(counter),
        "query_text": query_text,
    }


def classify_record_fast(
    record: Dict[str, Any],
    precomputed_emb: np.ndarray,
    query_type: str,
    desc_index,
    gl_index,
    metadata,
    supplier_map,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:

    supplier = record.get("supplier", "")

    # --------------------------------------------------------
    # QUERY TEXT — for output only, no embedding needed
    # --------------------------------------------------------

    query_text = record.get("text_desc", "") if query_type == "desc" else record.get("text_gl", "")

    # --------------------------------------------------------
    # USE PRE-COMPUTED EMBEDDING — skip OpenAI API call
    # --------------------------------------------------------

    query_emb = precomputed_emb

    # --------------------------------------------------------
    # SUPPLIER MATCHING + SEARCH
    # --------------------------------------------------------

    distances, ids, supplier_branch = supplier_matching_and_search(
        query_emb=query_emb,
        supplier=supplier,
        query_type=query_type,
        metadata=metadata,
        supplier_map=supplier_map,
        desc_index=desc_index,
        gl_index=gl_index
    )

    # --------------------------------------------------------
    # RADIUS FILTERING
    # --------------------------------------------------------

    neighbours = radius_filter(
        distances,
        ids,
        threshold=similarity_threshold
    )

    # --------------------------------------------------------
    # FALLBACK IF NOTHING FOUND
    # --------------------------------------------------------

    if len(neighbours) == 0:

        fallback_pairs = []

        for sim, idx in zip(distances, ids):

            if idx == -1:
                continue

            fallback_pairs.append((int(idx), float(sim)))

        neighbours = fallback_pairs[:5]

    # --------------------------------------------------------
    # DYNAMIC K
    # --------------------------------------------------------

    neighbours = dynamic_k(neighbours)

    if len(neighbours) == 0:

        return {
            "predicted_category": None,
            "confidence_score": 0.0,
            "decision": "MANUAL",
            "query_type": query_type,
            "supplier_branch": supplier_branch,
            "neighbours": [],
            "category_counter": {},
            "query_text": query_text,
        }

    # --------------------------------------------------------
    # PERFECT MATCH CHECK
    # --------------------------------------------------------

    for idx, sim in neighbours:

        if sim >= 0.9999:

            category = metadata.iloc[idx]["category"]

            return {
                "predicted_category": category,
                "confidence_score": 1.0,
                "decision": "AUTO",
                "query_type": query_type,
                "supplier_branch": supplier_branch,
                "neighbours": neighbours,
                "category_counter": {category: 1},
                "query_text": query_text,
            }

    # --------------------------------------------------------
    # KNN VOTING
    # --------------------------------------------------------

    counter = voting(neighbours, metadata)

    # --------------------------------------------------------
    # CATEGORY CHECK
    # --------------------------------------------------------

    check = neighbour_category_check(counter)

    # --------------------------------------------------------
    # CONFLICT RESOLUTION
    # --------------------------------------------------------

    if check == "conflict":

        neighbours = rerank(
            query_emb,
            neighbours,
            metadata,
            query_type
        )

        neighbours = dynamic_k(neighbours)

        counter = voting(neighbours, metadata)

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    predicted_category = counter.most_common(1)[0][0]

    # --------------------------------------------------------
    # CONFIDENCE SCORE
    # --------------------------------------------------------

    score = confidence_score(counter, neighbours)

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = decision_layer(score)

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "predicted_category": predicted_category,
        "confidence_score": score,
        "decision": decision,
        "query_type": query_type,
        "supplier_branch": supplier_branch,
        "neighbours": neighbours,
        "category_counter": dict(counter),
        "query_text": query_text,
    }