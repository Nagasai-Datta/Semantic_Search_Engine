FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and analysis scripts
COPY src/ ./src/
COPY analysis/ ./analysis/

# ── Run the full pipeline at build time ───────────────────────────────────────

# Step 1: clean corpus
RUN python3 src/data_cleaning.py

# Step 2: generate embeddings + populate ChromaDB (~90s)
RUN python3 src/embeddings.py

# Step 3: run clustering with a fixed k=20
# (justification.py already determined k=20 as optimal)
RUN python3 src/clustering.py

# ── Start the API ──────────────────────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]