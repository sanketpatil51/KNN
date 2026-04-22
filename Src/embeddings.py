import numpy as np
import faiss
from typing import List
from .config import client


def l2_normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


def generate_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i+batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]

        all_embeddings.extend(batch_embeddings)

    emb = np.array(all_embeddings).astype("float32")

    faiss.normalize_L2(emb)

    return emb


def generate_single_embedding(text: str) -> np.ndarray:

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )

    emb = np.array(response.data[0].embedding).astype("float32")
    emb = emb.reshape(1, -1)

    faiss.normalize_L2(emb)

    return emb[0]