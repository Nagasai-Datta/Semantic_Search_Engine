"""
embeddings.py
-------------
Pipeline Step 2 of 4.

Converts cleaned text documents into dense vector embeddings and stores them
in a persistent ChromaDB vector database for similarity search.

Model: sentence-transformers/all-MiniLM-L6-v2
    Produces one 384-dimensional vector per document.
    Unlike raw BERT (which produces per-token vectors requiring manual pooling),
    this model is fine-tuned specifically for sentence-level semantic similarity.
    It is 5x faster than larger alternatives with approximately 95% of the quality.

Vector store: ChromaDB
    Runs as a Python library with no external infrastructure.
    Persists to disk and survives process restarts.
    Supports cosine similarity search and metadata filtering natively.

Outputs:
    data/embeddings_matrix.npy    raw numpy matrix, shape (n_docs, 384)
    data/chroma_db/               persistent ChromaDB storage

Usage:
    python src/embeddings.py

Requires:
    data/cleaned_newsgroups.csv
"""

import os
import math
import logging
import time

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_FILE      = os.path.join("data", "cleaned_newsgroups.csv")
CHROMA_DIR      = os.path.join("data", "chroma_db")
MATRIX_FILE     = os.path.join("data", "embeddings_matrix.npy")
COLLECTION_NAME = "newsgroups"
MODEL_NAME      = "all-MiniLM-L6-v2"
EMBED_BATCH     = 64     # documents per embedding batch — safe for low-memory machines
CHROMA_BATCH    = 500    # documents per ChromaDB insert batch


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_corpus(texts: list, model: SentenceTransformer) -> np.ndarray:
    """
    Encodes all documents into normalised embedding vectors.

    Documents are processed in batches of EMBED_BATCH to avoid out-of-memory
    errors on machines without a GPU.

    normalize_embeddings=True produces unit-length vectors so that
    cosine similarity reduces to a dot product — faster to compute at query time.

    Returns:
        numpy array of shape (n_docs, 384)
    """
    n         = len(texts)
    n_batches = math.ceil(n / EMBED_BATCH)
    all_embs  = []
    start     = time.time()

    log.info(f"Encoding {n} documents across {n_batches} batches...")

    for i in range(n_batches):
        batch = texts[i * EMBED_BATCH : (i + 1) * EMBED_BATCH]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            log.info(f"  Batch {i+1}/{n_batches} | elapsed: {time.time()-start:.1f}s")

    matrix = np.vstack(all_embs)
    log.info(f"Encoding complete. Shape: {matrix.shape} | Total time: {time.time()-start:.1f}s")
    return matrix


# ── ChromaDB storage ───────────────────────────────────────────────────────────

def store_in_chroma(df: pd.DataFrame, matrix: np.ndarray):
    """
    Inserts all document vectors into a ChromaDB collection.

    Each record contains:
    - id        : unique doc_id string (e.g. 'doc_0')
    - embedding : 384-dimensional float vector
    - document  : original cleaned text (returned in search results)
    - metadata  : {'label': int, 'category': str} for filtered queries

    The collection is configured with cosine distance, which is the standard
    metric for high-dimensional text embedding spaces. Cosine measures the
    angle between vectors rather than their magnitude, making it robust to
    documents of varying length.

    An existing collection with the same name is dropped before re-creation
    to allow clean reruns without duplicate entries.
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Existing collection deleted for fresh insertion.")
    except Exception:
        pass

    collection = client.create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )

    n         = len(df)
    n_batches = math.ceil(n / CHROMA_BATCH)
    texts     = df["text"].tolist()
    doc_ids   = df["doc_id"].tolist()
    labels    = df["label"].tolist()
    cats      = df["category"].tolist()

    for i in range(n_batches):
        s = i * CHROMA_BATCH
        e = min(s + CHROMA_BATCH, n)

        collection.add(
            ids        = doc_ids[s:e],
            embeddings = matrix[s:e].tolist(),
            documents  = texts[s:e],
            metadatas  = [
                {"label": int(l), "category": c}
                for l, c in zip(labels[s:e], cats[s:e])
            ]
        )
        log.info(f"  Inserted {e}/{n} documents")

    log.info(f"ChromaDB population complete. Total documents: {collection.count()}")
    return collection


# ── Utility functions (imported by other modules) ──────────────────────────────

def get_model() -> SentenceTransformer:
    """
    Loads the embedding model.
    Imported by clustering.py, cache.py, and api.py to avoid redundant loading.
    """
    return SentenceTransformer(MODEL_NAME)


def get_collection():
    """
    Returns the existing ChromaDB collection.
    Imported by api.py at startup.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Encodes a single query string into a normalised 384-dim vector.
    Called on every incoming API request.

    Returns:
        1D numpy array of shape (384,)
    """
    return model.encode([query], normalize_embeddings=True)[0]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info(f"Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    log.info(f"Documents loaded: {len(df)}")

    log.info(f"Loading embedding model: {MODEL_NAME}")
    model = get_model()
    log.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    matrix = embed_corpus(df["text"].tolist(), model)

    os.makedirs("data", exist_ok=True)
    np.save(MATRIX_FILE, matrix)
    log.info(f"Embedding matrix saved → {MATRIX_FILE}  ({matrix.nbytes / 1e6:.1f} MB)")

    store_in_chroma(df, matrix)

    print(f"\n  Embeddings complete.")
    print(f"  Matrix shape : {matrix.shape}")
    print(f"  ChromaDB at  : {CHROMA_DIR}")
    print(f"  Next step    : python analysis/justification.py")


if __name__ == "__main__":
    main()
