import numpy as np
from collections import Counter
from typing import List, Tuple

from .config import MIN_K, MAX_K


def radius_filter(distances, ids, threshold):

    neighbours = []

    for sim, idx in zip(distances, ids):

        if idx == -1:
            continue

        if float(sim) >= threshold:
            neighbours.append((int(idx), float(sim)))

    return neighbours


def dynamic_k(
    neighbours,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
    cumulative_threshold: float = 0.35
):

    if not neighbours:
        return []

    neighbours = sorted(neighbours, key=lambda x: x[1], reverse=True)

    selected = [neighbours[0]]
    top_similarity = neighbours[0][1]

    for i in range(1, len(neighbours)):

        curr_sim = neighbours[i][1]

        cumulative_diff = top_similarity - curr_sim

        if cumulative_diff > cumulative_threshold and len(selected) >= min_k:
            break

        selected.append(neighbours[i])

        if len(selected) >= max_k:
            break

    k = len(selected)

    if k % 2 == 0:

        if k > min_k:
            selected = selected[:-1]

        elif k < len(neighbours):
            selected.append(neighbours[k])

    return selected


def voting(neighbours, metadata):

    cats = []

    for idx, _sim in neighbours:
        cats.append(metadata.iloc[idx]["category"])

    return Counter(cats)


def neighbour_category_check(counter):

    if len(counter) <= 1:
        return "agree"

    return "conflict"


def rerank(query_emb, neighbours, metadata, query_type):

    rescored = []

    emb_col = "emb_desc" if query_type == "desc" else "emb_gl"

    for idx, _sim in neighbours:

        hist_emb = metadata.iloc[idx][emb_col]

        cosine = float(np.dot(query_emb, hist_emb))

        rescored.append((idx, cosine))

    rescored.sort(key=lambda x: x[1], reverse=True)

    return rescored


def confidence_score(counter, neighbours):

    if not neighbours or len(counter) == 0:
        return 0.0

    vote_ratio = counter.most_common(1)[0][1] / sum(counter.values())

    avg_sim = float(np.mean([sim for _, sim in neighbours]))

    score = 0.6 * vote_ratio + 0.4 * avg_sim

    return round(score, 4)