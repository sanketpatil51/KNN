from .config import AUTO_THRESHOLD, REVIEW_THRESHOLD


def decision_layer(score: float):

    if score >= AUTO_THRESHOLD:
        return "AUTO"

    elif score >= REVIEW_THRESHOLD:
        return "REVIEW"

    else:
        return "MANUAL"