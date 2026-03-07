"""
src/api.py
----------
Pipeline Step 4 of 4.

FastAPI service exposing semantic search with cluster-aware caching.

Endpoints:
    POST   /query         semantic search with cache lookup
    GET    /cache/stats   cache hit/miss statistics
    DELETE /cache         flush cache and reset statistics
    GET    /health        readiness check

Startup loads:
    - sentence-transformers model (for embedding queries)
    - ChromaDB collection (for semantic search)
    - cluster_memberships.npy (for cache initialisation)
    - cluster_centers.npy (GMM centers in 50-dim PCA space — for cluster routing)
    - pca_model.joblib (fitted PCA — projects 384-dim queries to 50-dim for routing)

The PCA model and cluster centers must match — both are produced by the same
run of src/clustering.py and must not be mixed across runs.

Usage:
    uvicorn src.api:app --reload

Requires:
    data/chroma_db/
    data/cluster_memberships.npy
    data/cluster_centers.npy
    data/pca_model.joblib
"""

import os
import sys
import logging
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_model, get_collection, embed_query
from src.cache import SemanticCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MEMBERSHIP_FILE = os.path.join("data", "cluster_memberships.npy")
CENTERS_FILE    = os.path.join("data", "cluster_centers.npy")
PCA_MODEL_FILE  = os.path.join("data", "pca_model.joblib")
N_RESULTS       = 5


# ── App state ──────────────────────────────────────────────────────────────────

class _AppState:
    model      = None
    collection = None
    cache      = None

state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all components once at server startup.
    Order matters:
        1. Embedding model — needed for every request
        2. ChromaDB — needed for cache miss search
        3. Cluster memberships — initialises cache bucket structure
        4. Cluster centers — enables PCA-space cluster routing
        5. PCA model — projects 384-dim queries to 50-dim for routing

    The PCA model and cluster centers must be from the same clustering.py run.
    If clustering.py is re-run, restart the API to reload both.
    """
    log.info("Loading embedding model...")
    state.model = get_model()

    log.info("Connecting to ChromaDB...")
    state.collection = get_collection()
    log.info(f"Collection: {state.collection.count()} documents")

    log.info("Loading cluster memberships...")
    memberships = np.load(MEMBERSHIP_FILE)
    state.cache = SemanticCache(memberships=memberships)

    if os.path.exists(CENTERS_FILE):
        state.cache.set_cluster_centers(np.load(CENTERS_FILE))
        log.info("Cluster centers loaded.")
    else:
        log.warning(f"Cluster centers not found at {CENTERS_FILE} — routing will use fallback.")

    if os.path.exists(PCA_MODEL_FILE):
        pca = joblib.load(PCA_MODEL_FILE)
        state.cache.set_pca_model(pca)
        log.info("PCA model loaded.")
    else:
        log.warning(f"PCA model not found at {PCA_MODEL_FILE} — routing will use fallback.")

    log.info("API ready.")
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Semantic Search — 20 Newsgroups",
    description = (
        "Semantic search over the 20 Newsgroups corpus using dense embeddings, "
        "GMM soft clustering, and a cluster-aware semantic cache."
    ),
    version     = "1.0.0",
    lifespan    = lifespan
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {"example": {"query": "space shuttle NASA orbit mission"}}
    }


class QueryResponse(BaseModel):
    query           : str
    cache_hit       : bool
    matched_query   : str | None
    similarity_score: float | None
    result          : list
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries       : int
    hit_count           : int
    miss_count          : int
    hit_rate            : float
    n_clusters          : int
    threshold           : float
    entries_per_cluster : dict


# ── Search ─────────────────────────────────────────────────────────────────────

def semantic_search(query_embedding: np.ndarray) -> list:
    """
    Queries ChromaDB for the top N_RESULTS most similar documents.
    Uses the full 384-dim normalised query embedding — ChromaDB operates
    in the original embedding space, not the PCA-reduced space.
    """
    results = state.collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results        = N_RESULTS,
        include          = ["documents", "metadatas", "distances"]
    )
    return [
        {
            "text"      : doc[:300],
            "category"  : meta["category"],
            "similarity": round(1 - dist, 4)
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main search endpoint.

    Flow:
    1. Embed query → 384-dim normalised vector.
    2. Check semantic cache:
       - Cluster routing: PCA-project to 50-dim → find nearest GMM center
       - Similarity check: cosine similarity against stored 384-dim embeddings
    3a. HIT  → return cached result.
    3b. MISS → ChromaDB search → store in cache → return result.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"Query: '{request.query}'")
    query_emb    = embed_query(request.query, state.model)
    cache_result = state.cache.lookup(query_emb)

    if cache_result:
        return QueryResponse(
            query            = request.query,
            cache_hit        = True,
            matched_query    = cache_result["matched_query"],
            similarity_score = round(cache_result["similarity_score"], 4),
            result           = cache_result["result"],
            dominant_cluster = cache_result["dominant_cluster"]
        )

    search_results   = semantic_search(query_emb)
    dominant_cluster = state.cache.store(request.query, query_emb, search_results)

    return QueryResponse(
        query            = request.query,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        result           = search_results,
        dominant_cluster = dominant_cluster
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Returns current cache statistics."""
    return CacheStats(**state.cache.get_stats())


@app.delete("/cache")
async def flush_cache():
    """Clears all cache entries and resets hit/miss counters."""
    state.cache.flush()
    return {"message": "Cache flushed.", "status": "ok"}


@app.get("/health")
async def health():
    """Readiness check — confirms all components are loaded."""
    return {
        "status"       : "ok",
        "chroma_docs"  : state.collection.count(),
        "cache_entries": state.cache.total_entries,
        "hit_rate"     : state.cache.hit_rate
    }
