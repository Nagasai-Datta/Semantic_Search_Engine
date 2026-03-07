# Semantic Search System — 20 Newsgroups

Lightweight semantic search over the 20 Newsgroups corpus (~18,000 documents)
using dense vector embeddings, GMM soft clustering, and a cluster-aware semantic cache.

---

## Architecture

```
data_cleaning.py
      |
      |  cleaned_newsgroups.csv
      v
embeddings.py
      |
      |  embeddings_matrix.npy    chroma_db/
      v
analysis/justification.py        <── run this first to choose k
      |
      |  elbow_curves.png   cluster_heatmap.png   threshold_experiment.png ...
      v
clustering.py
      |
      |  cluster_memberships.npy   cluster_centers.npy   pca_model.joblib
      v
api.py  (uvicorn)
      |
      query embedding (384-dim)
      ├── ChromaDB search (384-dim cosine)
      └── Cache routing (PCA 50-dim → GMM center → bucket)
             └── Similarity check (384-dim cosine within bucket)
```

---

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clean corpus
python src/data_cleaning.py

# 4. Generate embeddings
python src/embeddings.py

# 5. Determine best cluster count
python analysis/justification.py
#    Read the printed "Recommended k" value.
#    Open src/clustering.py and set BEST_K to that value.

# 6. Run clustering
python src/clustering.py

# 7. Start API
uvicorn src.api:app --reload
#    Open http://localhost:8000/docs
```

Or with Make:

```bash
make setup
make justify    # step 5 — do before updating BEST_K
make pipeline   # steps 3, 4, 6
make api
```

---

## API Endpoints

| Method | Path           | Description                |
| ------ | -------------- | -------------------------- |
| POST   | `/query`       | Semantic search with cache |
| GET    | `/cache/stats` | Cache statistics           |
| DELETE | `/cache`       | Flush cache                |
| GET    | `/health`      | Readiness check            |

### Example: first query (cache miss)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "space shuttle NASA orbit"}'
```

```json
{
  "query": "space shuttle NASA orbit",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 4
}
```

### Example: paraphrase (cache hit)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "NASA rocket launch mission"}'
```

```json
{
  "query": "NASA rocket launch mission",
  "cache_hit": true,
  "matched_query": "space shuttle NASA orbit",
  "similarity_score": 0.87,
  "result": [...],
  "dominant_cluster": 4
}
```

Different words, same meaning — the semantic cache returns the stored result
without recomputation.

---

## Design Decisions

Full reasoning and evidence in [`analysis/justification.md`](analysis/justification.md).

### Embedding model

`all-MiniLM-L6-v2` (sentence-transformers) was chosen over raw BERT and TF-IDF.
Raw BERT produces per-token vectors requiring manual pooling.
TF-IDF cannot recognise paraphrases.
`all-MiniLM-L6-v2` produces one 384-dim semantic vector per document directly.

![Embedding similarity demo](analysis/plots/embedding_similarity_demo.png)

---

### Soft clustering — why GMM

Hard clustering forces one label per document. A post about gun legislation
belongs semantically to both politics and firearms.

GMM produces a probability distribution per document:

```
"Senate votes on NASA funding" → {politics: 0.55, space: 0.38, other: 0.07}
```

**Why GMM over Fuzzy C-Means:**
skfuzzy's cmeans was tested first but produced degenerate results
(FPC = exactly 1/k regardless of k) due to numerical instability on
high-dimensional text embeddings. sklearn's GaussianMixture uses
Expectation-Maximisation and converges reliably.

**PCA before clustering:**
In 384 dimensions, pairwise distances become nearly uniform (curse of
dimensionality). PCA reduces to 50 components — retaining dominant variance
while restoring meaningful distance geometry. The fitted PCA model is saved
to `data/pca_model.joblib` for use at query time.

---

### Cluster count

![Elbow curves](analysis/plots/elbow_curves.png)

GMM fitted for k ∈ {5, 10, 15, 20, 25, 30}. BIC elbow and Silhouette peak
identify the recommended cluster count.

---

### Semantic coherence

![Cluster heatmap](analysis/plots/cluster_heatmap.png)

Average GMM membership per original category per cluster. Clusters lighting
up for semantically related categories confirm meaningful topic grouping.

---

### Cluster separation

![PCA projection](analysis/plots/pca_clusters.png)

2D PCA projection coloured by dominant cluster. Distinct regions confirm
clusters occupy separate areas of the embedding space.

---

### Boundary cases

`analysis/plots/boundary_cases.txt` shows the highest-entropy documents.
These span multiple topics — the soft model captures this ambiguity correctly.

```
category: talk.politics.guns   entropy: 2.41
  cluster_N (politics): 0.44
  cluster_M (firearms): 0.39
```

---

### Cache threshold

![Threshold experiment](analysis/plots/threshold_experiment.png)

At threshold = 0.75, genuine paraphrases are recognised while distinct
queries are correctly treated as misses.

---

## Files generated at runtime (gitignored)

| File                           | Generated by       | Used by                             |
| ------------------------------ | ------------------ | ----------------------------------- |
| `data/cleaned_newsgroups.csv`  | `data_cleaning.py` | `embeddings.py`                     |
| `data/embeddings_matrix.npy`   | `embeddings.py`    | `clustering.py`, `justification.py` |
| `data/chroma_db/`              | `embeddings.py`    | `api.py`                            |
| `data/cluster_memberships.npy` | `clustering.py`    | `api.py`                            |
| `data/cluster_centers.npy`     | `clustering.py`    | `api.py` → `cache.py`               |
| `data/pca_model.joblib`        | `clustering.py`    | `api.py` → `cache.py`               |

---

## Project Structure

```
semantic-search/
├── src/
│   ├── data_cleaning.py   Step 1 — clean corpus
│   ├── embeddings.py      Step 2 — embed + ChromaDB
│   ├── clustering.py      Step 3 — PCA + GMM clustering
│   ├── cache.py           Semantic cache (used by api.py)
│   └── api.py             Step 4 — FastAPI service
├── analysis/
│   ├── justification.py   Run before clustering — determines k
│   ├── justification.md   Design decisions and evidence
│   └── plots/             Generated plots — committed to git
├── data/
│   └── samples/           Small samples — committed to git
├── requirements.txt
├── Makefile
└── .gitignore
```

---

## Dependencies

```
scikit-learn         PCA, GMM, silhouette scoring
sentence-transformers  embedding model
chromadb             vector database
fastapi + uvicorn    API
joblib               PCA model persistence
pandas, numpy        data handling
matplotlib, seaborn  justification plots
```
