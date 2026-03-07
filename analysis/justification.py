"""
analysis/justification.py
--------------------------
Cluster count selection and threshold analysis.

Run this script BEFORE src/clustering.py to determine the optimal number
of clusters (k) and the similarity threshold for the semantic cache.

Clustering approach: Gaussian Mixture Models (GMM)
    GMM is a soft/probabilistic clustering algorithm — each document receives
    a probability distribution across all clusters, identical in concept to
    Fuzzy C-Means. GMM is chosen over skfuzzy's cmeans implementation because
    sklearn's GMM is numerically stable, well-tested, and converges reliably
    on reduced-dimension text embeddings.

The script produces:
  1. BIC/AIC + Silhouette elbow curves — justifies cluster count choice
  2. Category-cluster heatmap — validates semantic coherence
  3. PCA projection coloured by cluster — visual separation check
  4. Cluster sample text files — qualitative semantic validation
  5. Threshold experiment plot — justifies cache similarity threshold
  6. A printed recommendation for BEST_K

After running:
  - Note the printed "Recommended k" value.
  - Update BEST_K in src/clustering.py to match.
  - Commit analysis/plots/ to git as evidence.

Usage:
    python analysis/justification.py

Requires:
    data/embeddings_matrix.npy
    data/cleaned_newsgroups.csv
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MATRIX_FILE       = os.path.join("data", "embeddings_matrix.npy")
CSV_FILE          = os.path.join("data", "cleaned_newsgroups.csv")
PLOTS_DIR         = os.path.join("analysis", "plots")
K_VALUES          = [5, 10, 15, 20, 25, 30]
PCA_COMPONENTS    = 50
SILHOUETTE_SAMPLE = 3000
THRESHOLD_VALUES  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
THRESHOLD_SAMPLE  = 500


# ── PCA reduction ──────────────────────────────────────────────────────────────

def reduce_dimensions(matrix: np.ndarray):
    """
    Reduces 384-dim embeddings to PCA_COMPONENTS dimensions before clustering.

    Clustering algorithms rely on distance metrics. In 384 dimensions,
    pairwise distances between all points converge to nearly the same value
    (curse of dimensionality), making cluster boundaries indistinguishable.
    PCA to 50 components retains dominant variance while making distances
    meaningful. Full 384-dim vectors continue to be used by ChromaDB and
    the cache at query time.
    """
    log.info(f"Applying PCA: {matrix.shape[1]} → {PCA_COMPONENTS} dimensions...")
    pca     = PCA(n_components=PCA_COMPONENTS, random_state=42)
    reduced = pca.fit_transform(matrix)
    log.info(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return reduced


# ── Experiment ─────────────────────────────────────────────────────────────────

def run_cluster_experiment(reduced: np.ndarray) -> dict:
    """
    Fits a Gaussian Mixture Model for each k in K_VALUES.

    Why GMM for soft clustering:
        GMM assigns each document a posterior probability of belonging to each
        component — mathematically equivalent to fuzzy membership.
        A document about gun legislation gets e.g.:
            {politics: 0.55, firearms: 0.38, other: 0.07}
        sklearn's GaussianMixture is numerically stable and converges reliably,
        unlike skfuzzy which degenerates to uniform memberships (FPC = 1/k)
        in high-dimensional spaces.

    BIC (Bayesian Information Criterion): lower is better.
        Penalises complexity — the elbow identifies the most efficient k.

    AIC (Akaike Information Criterion): lower is better.
        Less penalty than BIC — used as a cross-reference.

    Silhouette Score: higher is better.
        Measures cluster separation on a subsample.
    """
    results    = {"k": [], "bic": [], "aic": [], "silhouette": [], "memberships": []}
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(reduced), min(SILHOUETTE_SAMPLE, len(reduced)), replace=False)
    sample_mat = reduced[sample_idx]

    for k in K_VALUES:
        log.info(f"Fitting GMM k={k}...")
        gmm = GaussianMixture(
            n_components    = k,
            covariance_type = "diag",
            random_state    = 42,
            max_iter        = 200,
            n_init          = 3
        )
        gmm.fit(reduced)
        memberships   = gmm.predict_proba(reduced)
        hard_labels   = np.argmax(memberships, axis=1)
        sample_labels = hard_labels[sample_idx]
        unique        = np.unique(sample_labels)
        sil = (
            silhouette_score(sample_mat, sample_labels, metric="euclidean")
            if len(unique) >= 2 else 0.0
        )
        results["k"].append(k)
        results["bic"].append(gmm.bic(reduced))
        results["aic"].append(gmm.aic(reduced))
        results["silhouette"].append(sil)
        results["memberships"].append(memberships)
        log.info(
            f"  k={k:2d} | BIC={gmm.bic(reduced):.1f} | "
            f"Silhouette={sil:.4f}"
        )

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_elbow_curves(results: dict) -> None:
    """
    BIC/AIC and Silhouette plots vs k.
    BIC elbow = recommended k. Silhouette peak cross-validates it.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    best_bic_k = results["k"][int(np.argmin(results["bic"]))]
    best_sil_k = results["k"][int(np.argmax(results["silhouette"]))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results["k"], results["bic"], "o-", color="steelblue", lw=2, ms=8, label="BIC")
    axes[0].plot(results["k"], results["aic"], "s--", color="darkorange", lw=2, ms=8, label="AIC")
    axes[0].axvline(best_bic_k, color="red", ls="--", label=f"BIC elbow k={best_bic_k}")
    axes[0].set_title("BIC / AIC vs k\n(lower is better — look for elbow)", fontsize=12)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["k"], results["silhouette"], "o-", color="teal", lw=2, ms=8)
    axes[1].axvline(best_sil_k, color="red", ls="--", label=f"Best k={best_sil_k}")
    axes[1].set_title("Silhouette Score vs k\n(higher = better separation)", fontsize=12)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Cluster Count Justification — GMM on PCA-50 Embeddings", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elbow_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'elbow_curves.png')}")


def plot_cluster_heatmap(memberships: np.ndarray, df: pd.DataFrame, k: int) -> None:
    """Average GMM membership per original category per cluster."""
    cats        = df["category"].tolist()
    unique_cats = sorted(set(cats))
    cat_idx     = {c: i for i, c in enumerate(unique_cats)}
    avg         = np.zeros((len(unique_cats), k))
    for i, cat in enumerate(cats):
        avg[cat_idx[cat]] += memberships[i]
    counts = np.array([cats.count(c) for c in unique_cats]).reshape(-1, 1)
    avg   /= counts

    fig, ax = plt.subplots(figsize=(max(k, 12), 8))
    sns.heatmap(avg, xticklabels=[f"C{i}" for i in range(k)],
                yticklabels=unique_cats, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Average Membership Probability"})
    ax.set_title(
        f"Average GMM Membership: Category vs Cluster  (k={k})\n"
        "Bright = documents from that category strongly belong to that cluster",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cluster_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'cluster_heatmap.png')}")


def plot_pca(reduced: np.ndarray, memberships: np.ndarray, k: int) -> None:
    """2D PCA projection coloured by dominant cluster."""
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(reduced), min(3000, len(reduced)), replace=False)
    pca2   = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(reduced[idx])
    labels = np.argmax(memberships[idx], axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", alpha=0.5, s=8)
    plt.colorbar(sc, ax=ax, label="Dominant Cluster")
    ax.set_title(f"PCA Projection — Dominant GMM Cluster  (k={k})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_clusters.png"), dpi=150)
    plt.close()
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'pca_clusters.png')}")


def write_cluster_samples(df: pd.DataFrame, memberships: np.ndarray, k: int) -> None:
    """Top-3 most representative documents per cluster."""
    lines = [f"CLUSTER SAMPLES  k={k}\n{'='*60}\n"]
    for c in range(k):
        top = np.argsort(memberships[:, c])[::-1][:3]
        lines.append(f"\nCLUSTER {c}")
        lines.append("-" * 40)
        for rank, idx in enumerate(top):
            row = df.iloc[idx]
            lines.append(f"  [{rank+1}] category={row['category']}  membership={memberships[idx, c]:.3f}")
            lines.append(f"      {row['text'][:180]}...")
            lines.append("")
    with open(os.path.join(PLOTS_DIR, "cluster_samples.txt"), "w") as f:
        f.write("\n".join(lines))
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'cluster_samples.txt')}")


def write_boundary_cases(df: pd.DataFrame, memberships: np.ndarray, k: int) -> None:
    """10 highest-entropy documents — most spread across clusters."""
    eps     = 1e-10
    entropy = -np.sum(memberships * np.log(memberships + eps), axis=1)
    top     = np.argsort(entropy)[::-1][:10]
    lines   = [
        f"BOUNDARY / UNCERTAIN DOCUMENTS  k={k}\n{'='*60}",
        "High entropy = membership spread across clusters = document spans multiple topics.\n"
    ]
    for rank, idx in enumerate(top):
        row  = df.iloc[idx]
        top2 = np.argsort(memberships[idx])[::-1][:2]
        lines.append(f"[{rank+1}] category={row['category']}  entropy={entropy[idx]:.3f}")
        lines.append(
            f"    cluster {top2[0]}: {memberships[idx, top2[0]]:.3f}  |  "
            f"cluster {top2[1]}: {memberships[idx, top2[1]]:.3f}"
        )
        lines.append(f"    {row['text'][:250]}...")
        lines.append("")
    with open(os.path.join(PLOTS_DIR, "boundary_cases.txt"), "w") as f:
        f.write("\n".join(lines))
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'boundary_cases.txt')}")


def run_threshold_experiment(matrix: np.ndarray) -> None:
    """
    Simulates cache hit rate at multiple cosine similarity thresholds.
    Uses original 384-dim normalised embeddings — the space the cache operates in.
    """
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(matrix), THRESHOLD_SAMPLE, replace=False)
    sub   = matrix[idx].astype(np.float32)
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10
    sub   = sub / norms
    sims  = (sub @ sub.T).flatten()
    sims  = sims[sims < 0.9999]   # remove self-similarities

    hit_rates = [float(np.mean(sims >= t)) for t in THRESHOLD_VALUES]

    plt.figure(figsize=(9, 5))
    plt.plot(THRESHOLD_VALUES, hit_rates, "o-", color="teal", lw=2, ms=8)
    plt.axvline(0.75, color="red", ls="--", label="Selected threshold = 0.75")
    plt.title(
        "Cache Hit Rate vs Similarity Threshold\n"
        "Justification for DEFAULT_THRESHOLD = 0.75 in cache.py", fontsize=12
    )
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Fraction of Pairs Exceeding Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "threshold_experiment.png"), dpi=150)
    plt.close()
    log.info(f"Saved → {os.path.join(PLOTS_DIR, 'threshold_experiment.png')}")

    print("\n  Threshold experiment results:")
    for t, hr in zip(THRESHOLD_VALUES, hit_rates):
        marker = " ← selected" if t == 0.75 else ""
        print(f"    threshold={t:.2f} | hit_rate={hr:.4f}{marker}")


def print_recommendation(results: dict) -> int:
    best_bic_k = results["k"][int(np.argmin(results["bic"]))]
    best_sil_k = results["k"][int(np.argmax(results["silhouette"]))]

    print("\n" + "=" * 60)
    print("  CLUSTER COUNT JUSTIFICATION SUMMARY")
    print("=" * 60)
    print("\n  k  | BIC          | AIC          | Silhouette")
    print("  " + "-" * 52)
    for k, bic, aic, sil in zip(results["k"], results["bic"], results["aic"], results["silhouette"]):
        mark = " ← recommended" if k == best_bic_k else ""
        print(f"  {k:2d} | {bic:12.1f} | {aic:12.1f} | {sil:.4f}{mark}")

    print(f"\n  BIC best k        : {best_bic_k}")
    print(f"  Silhouette best k : {best_sil_k}")
    print(f"\n  Recommended k = {best_bic_k}")
    print(f"  Update BEST_K = {best_bic_k} in src/clustering.py")
    print("=" * 60)
    return best_bic_k


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    log.info("Loading data...")
    matrix = np.load(MATRIX_FILE)
    df     = pd.read_csv(CSV_FILE)
    log.info(f"Matrix: {matrix.shape} | Documents: {len(df)}")

    reduced = reduce_dimensions(matrix)

    log.info("Running GMM experiments (this may take several minutes)...")
    results = run_cluster_experiment(reduced)

    plot_elbow_curves(results)
    recommended_k = print_recommendation(results)

    best_idx    = results["k"].index(recommended_k)
    memberships = results["memberships"][best_idx]

    plot_cluster_heatmap(memberships, df, recommended_k)
    plot_pca(reduced, memberships, recommended_k)
    write_cluster_samples(df, memberships, recommended_k)
    write_boundary_cases(df, memberships, recommended_k)
    run_threshold_experiment(matrix)

    print(f"\n  All plots saved to {PLOTS_DIR}/")
    print(f"  Commit the plots/ folder to git as evidence.")
    print(f"\n  Next: update BEST_K = {recommended_k} in src/clustering.py")
    print(f"  Then: python src/clustering.py")


if __name__ == "__main__":
    main()