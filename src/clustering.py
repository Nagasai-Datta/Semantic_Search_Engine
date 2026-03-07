"""
src/clustering.py
-----------------
Pipeline Step 3 of 4.

Applies PCA dimensionality reduction followed by Gaussian Mixture Model (GMM)
clustering to produce soft membership distributions across all documents.

Why GMM over Fuzzy C-Means (skfuzzy):
    Both are soft clustering algorithms producing per-document probability
    distributions across clusters. skfuzzy's cmeans degenerates to uniform
    memberships (FPC = 1/k) on high-dimensional text embeddings because all
    pairwise distances converge (curse of dimensionality). sklearn's
    GaussianMixture uses Expectation-Maximisation and converges reliably.

Why PCA before clustering:
    In 384 dimensions, distance metrics become unreliable. PCA to 50 components
    retains dominant variance while making cluster boundaries distinguishable.
    The PCA model is saved to disk so api.py can project incoming query
    embeddings into the same 50-dim space for cluster routing.

    Full 384-dim vectors are still used by ChromaDB for semantic search
    and by the cache for cosine similarity comparison. PCA is used only
    for cluster assignment routing.

BEST_K:
    Run analysis/justification.py first. Update BEST_K to the recommended value.

Outputs:
    data/cluster_memberships.npy   shape (n_docs, k)
    data/cluster_centers.npy       shape (k, 50)  — in PCA space
    data/pca_model.joblib          fitted PCA — used by api.py at query time
    data/samples/cluster_sample.json

Usage:
    python src/clustering.py

Requires:
    data/embeddings_matrix.npy
    data/cleaned_newsgroups.csv
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MATRIX_FILE     = os.path.join("data", "embeddings_matrix.npy")
CSV_FILE        = os.path.join("data", "cleaned_newsgroups.csv")
MEMBERSHIP_FILE = os.path.join("data", "cluster_memberships.npy")
CENTERS_FILE    = os.path.join("data", "cluster_centers.npy")
PCA_MODEL_FILE  = os.path.join("data", "pca_model.joblib")
SAMPLE_FILE     = os.path.join("data", "samples", "cluster_sample.json")

# ── Update after running analysis/justification.py ────────────────────────────
BEST_K         = 20
PCA_COMPONENTS = 50


# ── PCA reduction ──────────────────────────────────────────────────────────────

def reduce_dimensions(matrix: np.ndarray) -> tuple:
    """
    Reduces 384-dim embeddings to PCA_COMPONENTS dimensions.

    The fitted PCA object is saved to PCA_MODEL_FILE so that incoming
    query embeddings at API request time can be projected into the same
    reduced space for cluster assignment.

    Returns:
        reduced : np.ndarray of shape (n_docs, PCA_COMPONENTS)
        pca     : fitted PCA object
    """
    log.info(f"Applying PCA: {matrix.shape[1]} → {PCA_COMPONENTS} dimensions...")
    pca     = PCA(n_components=PCA_COMPONENTS, random_state=42)
    reduced = pca.fit_transform(matrix)
    log.info(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return reduced, pca


# ── GMM clustering ─────────────────────────────────────────────────────────────

def run_gmm(reduced: np.ndarray, n_clusters: int) -> tuple:
    """
    Fits a Gaussian Mixture Model on PCA-reduced embeddings.

    Returns:
        memberships : (n_docs, k) — soft assignment probabilities per document
        centers     : (k, PCA_COMPONENTS) — component means in PCA space
        gmm         : fitted GaussianMixture object
    """
    log.info(f"Fitting GMM: k={n_clusters}...")
    gmm = GaussianMixture(
        n_components    = n_clusters,
        covariance_type = "diag",
        random_state    = 42,
        max_iter        = 200,
        n_init          = 3
    )
    gmm.fit(reduced)
    memberships = gmm.predict_proba(reduced)
    centers     = gmm.means_
    log.info(f"GMM converged. BIC={gmm.bic(reduced):.1f}")
    return memberships, centers, gmm


# ── Sample builders ────────────────────────────────────────────────────────────

def build_cluster_sample(df: pd.DataFrame, memberships: np.ndarray, n: int = 3) -> dict:
    """Top-n most representative documents per cluster (highest membership score)."""
    k      = memberships.shape[1]
    sample = {}
    for c in range(k):
        top = np.argsort(memberships[:, c])[::-1][:n]
        sample[f"cluster_{c}"] = [
            {
                "doc_id"      : df.iloc[i]["doc_id"],
                "category"    : df.iloc[i]["category"],
                "membership"  : round(float(memberships[i, c]), 4),
                "text_snippet": df.iloc[i]["text"][:200]
            }
            for i in top
        ]
    return sample


def build_boundary_sample(df: pd.DataFrame, memberships: np.ndarray, n: int = 10) -> list:
    """
    Top-n highest-entropy documents.
    High entropy = membership spread across clusters = document spans multiple topics.
    These are the strongest evidence for soft over hard clustering.
    """
    eps     = 1e-10
    entropy = -np.sum(memberships * np.log(memberships + eps), axis=1)
    top     = np.argsort(entropy)[::-1][:n]
    results = []
    for idx in top:
        top2 = np.argsort(memberships[idx])[::-1][:2]
        results.append({
            "doc_id"      : df.iloc[idx]["doc_id"],
            "category"    : df.iloc[idx]["category"],
            "entropy"     : round(float(entropy[idx]), 4),
            "top_clusters": {
                f"cluster_{c}": round(float(memberships[idx, c]), 4)
                for c in top2
            },
            "text_snippet": df.iloc[idx]["text"][:300]
        })
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info("Loading data...")
    matrix = np.load(MATRIX_FILE)
    df     = pd.read_csv(CSV_FILE)
    log.info(f"Matrix: {matrix.shape} | Documents: {len(df)}")

    # Step 1 — PCA
    reduced, pca = reduce_dimensions(matrix)

    # Step 2 — GMM
    memberships, centers, gmm = run_gmm(reduced, n_clusters=BEST_K)

    # Step 3 — Save outputs
    os.makedirs("data", exist_ok=True)
    np.save(MEMBERSHIP_FILE, memberships)
    log.info(f"Memberships saved → {MEMBERSHIP_FILE}  shape: {memberships.shape}")

    np.save(CENTERS_FILE, centers)
    log.info(f"Centers saved     → {CENTERS_FILE}  shape: {centers.shape}")

    # Save PCA model — api.py needs this to project query embeddings at runtime
    joblib.dump(pca, PCA_MODEL_FILE)
    log.info(f"PCA model saved   → {PCA_MODEL_FILE}")

    # Step 4 — Save sample for git
    os.makedirs(os.path.dirname(SAMPLE_FILE), exist_ok=True)
    with open(SAMPLE_FILE, "w") as f:
        json.dump({
            "n_clusters"     : BEST_K,
            "bic"            : round(float(gmm.bic(reduced)), 2),
            "cluster_samples": build_cluster_sample(df, memberships),
            "boundary_cases" : build_boundary_sample(df, memberships)
        }, f, indent=2)
    log.info(f"Cluster sample saved → {SAMPLE_FILE}")

    dom = np.argmax(memberships, axis=1)
    print(f"\n  Clustering complete.")
    print(f"  Algorithm  : Gaussian Mixture Model (GMM)")
    print(f"  Clusters   : {BEST_K}")
    print(f"  BIC        : {gmm.bic(reduced):.1f}")
    print(f"  PCA model  : {PCA_MODEL_FILE}")
    print(f"\n  Dominant cluster distribution:")
    for c in range(BEST_K):
        print(f"    Cluster {c:2d}: {int(np.sum(dom == c))} documents")
    print(f"\n  Next step: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()