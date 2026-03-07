"""
data_cleaning.py
----------------
Pipeline Step 1 of 4.

Loads the 20 Newsgroups dataset from scikit-learn, applies text cleaning,
drops documents too short to embed meaningfully, and persists the result.

Outputs:
    data/cleaned_newsgroups.csv          full cleaned corpus
    data/samples/cleaned_sample.csv      500-row sample committed to git

Usage:
    python src/data_cleaning.py
"""

import os
import re
import logging

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_WORD_COUNT = 20        # minimum words required to keep a document
DATA_DIR       = "data"
OUTPUT_FILE    = os.path.join(DATA_DIR, "cleaned_newsgroups.csv")
SAMPLE_FILE    = os.path.join(DATA_DIR, "samples", "cleaned_sample.csv")
SAMPLE_SIZE    = 500
RANDOM_STATE   = 42        # fixed seed — reproducible output across runs


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans a single raw news post body.

    Cleaning steps applied in order:
    1. Quoted reply lines (starting with '>') are removed.
       These contain text authored by other users, not the document itself.
    2. Email addresses are removed — no semantic content.
    3. URLs are removed — no semantic content.
    4. Non-ASCII characters are removed — encoding artifacts from 1990s posts.
    5. Excess whitespace and newlines are collapsed to a single space.
    """
    text = re.sub(r"^>.*$",        "",  text, flags=re.MULTILINE)  # quoted lines
    text = re.sub(r"\S+@\S+",      "",  text)                       # email addresses
    text = re.sub(r"http\S+|www\.\S+", "", text)                    # URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)                      # non-ASCII
    text = re.sub(r"\s+",          " ",  text)                       # whitespace
    return text.strip()


def is_valid(text: str) -> bool:
    """
    Returns True if the document contains at least MIN_WORD_COUNT words.

    Documents under this threshold are typically empty posts, email signatures,
    or header-only artifacts left after cleaning. Such documents produce
    low-quality embeddings and are excluded from the corpus.
    """
    return len(text.split()) >= MIN_WORD_COUNT


# ── Pipeline ───────────────────────────────────────────────────────────────────

def load_and_clean() -> pd.DataFrame:
    """
    Loads the raw dataset and returns a cleaned DataFrame.

    sklearn's fetch_20newsgroups accepts a remove parameter that strips the
    most obvious metadata fields (From, Subject, Organization, Lines headers)
    as a first pass. A second custom cleaning step handles remaining artifacts.

    subset='all' loads the full corpus (~20,000 documents) rather than the
    default train split, giving more coverage for unsupervised clustering.

    random_state=42 ensures consistent document ordering across runs.
    """
    log.info("Loading 20 Newsgroups dataset from scikit-learn...")
    raw = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE
    )
    log.info(f"Raw documents loaded: {len(raw.data)}")

    rows, dropped = [], 0

    for text, label in zip(raw.data, raw.target):
        cleaned = clean_text(text)
        if is_valid(cleaned):
            rows.append({
                "doc_id"    : f"doc_{len(rows)}",
                "text"      : cleaned,
                "label"     : int(label),
                "category"  : raw.target_names[label],
                "word_count": len(cleaned.split())
            })
        else:
            dropped += 1

    log.info(f"Retained: {len(rows)} | Dropped: {dropped} (fewer than {MIN_WORD_COUNT} words)")
    return pd.DataFrame(rows)


def save(df: pd.DataFrame) -> None:
    """Persists the cleaned DataFrame to CSV. Also saves a small sample for git."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Full dataset saved → {OUTPUT_FILE}")

    os.makedirs(os.path.dirname(SAMPLE_FILE), exist_ok=True)
    df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE) \
      .to_csv(SAMPLE_FILE, index=False)
    log.info(f"Sample ({SAMPLE_SIZE} rows) saved → {SAMPLE_FILE}")


def main():
    df = load_and_clean()
    save(df)
    print(f"\n  Cleaning complete.")
    print(f"  Documents retained : {len(df)}")
    print(f"  Categories         : {df['category'].nunique()}")
    print(f"  Average word count : {df['word_count'].mean():.0f}")
    print(f"  Next step          : python src/embeddings.py")


if __name__ == "__main__":
    main()
