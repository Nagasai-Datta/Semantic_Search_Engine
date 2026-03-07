"""
src/cache.py
------------
Cluster-aware semantic cache.

A standard key-value cache fails when two users phrase the same question
differently. This cache operates on query embeddings — two queries are
considered equivalent if their embedding vectors exceed a cosine similarity
threshold.

Two embedding spaces are used:
    384-dim (full)  : used for cosine similarity comparison inside buckets.
                      Captures fine-grained semantic similarity.
    50-dim (PCA)    : used for cluster routing (assigning a query to a bucket).
                      Must match the space the GMM cluster centers live in.

The PCA model (fitted in clustering.py) is passed in at API startup via
set_pca_model(). This ensures incoming query embeddings are projected into
the same 50-dim space that the GMM was trained on before cluster assignment.

Cluster routing for scalability:
    Without clustering, cache lookup is O(n) — compare against every entry.
    With k clusters, lookup is O(n/k) — compare only within one bucket.
    At k=15, this is approximately 15x faster as the cache grows.

Similarity threshold:
    The key tunable parameter. Controls what counts as "close enough" to hit.
    Too low  → false hits (wrong results returned).
    Too high → paraphrases missed (cache never hits).
    Justified by threshold experiment in analysis/justification.py.

No external caching libraries are used.
"""

import time
import logging
import numpy as np
from typing import Optional, Any

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.75


class SemanticCache:
    """
    Cluster-partitioned semantic cache backed by cosine similarity.

    Internal structure:
        _cache: dict[int, list[dict]]
            Key   = cluster index (0 .. n_clusters-1)
            Value = list of entries, each containing:
                      query            : original query string
                      embedding_full   : 384-dim normalised vector (for similarity)
                      result           : stored search result
                      timestamp        : float
                      dominant_cluster : int
    """

    def __init__(self, memberships: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
        """
        Args:
            memberships : (n_docs, k) from clustering.py — used only for n_clusters.
            threshold   : cosine similarity threshold for a cache hit.
        """
        self.n_clusters      = memberships.shape[1]
        self.threshold       = threshold
        self._cache          = {k: [] for k in range(self.n_clusters)}
        self._hit_count      = 0
        self._miss_count     = 0
        self._cluster_centers = None   # (k, 50) — GMM means in PCA space
        self._pca            = None    # fitted PCA object from clustering.py

        log.info(f"SemanticCache ready | clusters={self.n_clusters} | threshold={self.threshold}")

    # ── Setup ──────────────────────────────────────────────────────────────────

    def set_cluster_centers(self, centers: np.ndarray) -> None:
        """
        Stores GMM cluster centers (shape: k × 50, in PCA space).
        Called by api.py after loading cluster_centers.npy.
        """
        self._cluster_centers = centers
        log.info(f"Cluster centers set. Shape: {centers.shape}")

    def set_pca_model(self, pca) -> None:
        """
        Stores the fitted PCA model from clustering.py.
        Used to project 384-dim query embeddings into 50-dim PCA space
        before computing cluster assignment via GMM centers.

        Called by api.py after loading pca_model.joblib.
        Without this, cluster routing falls back to bucket-mean comparison
        in 384-dim space, which is less accurate.
        """
        self._pca = pca
        log.info("PCA model set for cluster routing.")

    # ── Cluster assignment ─────────────────────────────────────────────────────

    def _assign_cluster(self, embedding_full: np.ndarray) -> int:
        """
        Assigns an incoming 384-dim query embedding to a cluster bucket.

        If PCA model and cluster centers are available:
            Project embedding to 50-dim PCA space, then find nearest
            GMM center by Euclidean distance. This matches the space
            the GMM was trained in.

        Fallback (PCA/centers not loaded):
            Compute mean of 384-dim embeddings already in each bucket
            and compare by cosine similarity. Less accurate but functional.

        Returns:
            int — cluster index
        """
        if self._pca is not None and self._cluster_centers is not None:
            # Project to PCA space — same space as GMM centers
            embedding_pca = self._pca.transform(embedding_full.reshape(1, -1))[0]
            # Nearest GMM center by Euclidean distance
            dists = np.linalg.norm(self._cluster_centers - embedding_pca, axis=1)
            return int(np.argmin(dists))

        # Fallback: cosine similarity against mean of each bucket (384-dim)
        sims = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            bucket = self._cache[k]
            if bucket:
                embs   = np.array([e["embedding_full"] for e in bucket])
                center = embs.mean(axis=0)
                center = center / (np.linalg.norm(center) + 1e-10)
                sims[k] = float(np.dot(embedding_full, center))
            else:
                sims[k] = np.random.uniform(0, 0.05)
        return int(np.argmax(sims))

    # ── Core cache operations ──────────────────────────────────────────────────

    def lookup(self, query_embedding: np.ndarray) -> Optional[dict]:
        """
        Checks the cache for a semantically similar prior query.

        Cluster routing uses PCA space (50-dim) for accurate GMM-based
        assignment. Cosine similarity comparison inside the bucket uses
        the full 384-dim normalised vectors for fine-grained matching.

        Returns:
            dict with matched_query, result, similarity_score, dominant_cluster
            None on a miss.
        """
        cluster_id = self._assign_cluster(query_embedding)
        bucket     = self._cache[cluster_id]

        if not bucket:
            self._miss_count += 1
            log.debug(f"MISS — cluster {cluster_id} is empty")
            return None

        # Compare against full 384-dim embeddings in the bucket
        best_sim, best_entry = -1.0, None
        for entry in bucket:
            sim = float(np.dot(query_embedding, entry["embedding_full"]))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_sim >= self.threshold:
            self._hit_count += 1
            log.info(f"HIT  | cluster={cluster_id} | similarity={best_sim:.4f}")
            return {
                "matched_query"   : best_entry["query"],
                "result"          : best_entry["result"],
                "similarity_score": best_sim,
                "dominant_cluster": cluster_id
            }

        self._miss_count += 1
        log.info(f"MISS | cluster={cluster_id} | best_sim={best_sim:.4f} < {self.threshold}")
        return None

    def store(self, query: str, query_embedding: np.ndarray, result: Any) -> int:
        """
        Stores a query-result pair in the appropriate cluster bucket.

        query_embedding is the full 384-dim normalised vector.
        Cluster assignment uses PCA projection internally.

        Returns:
            int — cluster index the entry was stored in.
        """
        cluster_id = self._assign_cluster(query_embedding)
        self._cache[cluster_id].append({
            "query"           : query,
            "embedding_full"  : query_embedding,   # 384-dim, for similarity comparison
            "result"          : result,
            "timestamp"       : time.time(),
            "dominant_cluster": cluster_id
        })
        log.info(f"Stored | cluster={cluster_id} | total={self.total_entries}")
        return cluster_id

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def total_entries(self) -> int:
        return sum(len(b) for b in self._cache.values())

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return round(self._hit_count / total, 4) if total > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "total_entries"      : self.total_entries,
            "hit_count"          : self._hit_count,
            "miss_count"         : self._miss_count,
            "hit_rate"           : self.hit_rate,
            "n_clusters"         : self.n_clusters,
            "threshold"          : self.threshold,
            "entries_per_cluster": {
                f"cluster_{k}": len(v) for k, v in self._cache.items()
            }
        }

    def flush(self) -> None:
        """Clears all entries and resets statistics. Called by DELETE /cache."""
        self._cache      = {k: [] for k in range(self.n_clusters)}
        self._hit_count  = 0
        self._miss_count = 0
        log.info("Cache flushed.")