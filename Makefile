.PHONY: all setup pipeline justify api clean

all: setup justify pipeline api

setup:
	pip install -r requirements.txt

# Run after embeddings — determines best k before clustering
justify:
	python analysis/justification.py

# Full data pipeline (clean → embed → cluster)
pipeline:
	python src/data_cleaning.py
	python src/embeddings.py
	python src/clustering.py

# Start the API
api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Remove generated data files (keeps samples and plots)
clean:
	rm -f data/cleaned_newsgroups.csv
	rm -f data/embeddings_matrix.npy
	rm -f data/cluster_memberships.npy
	rm -f data/cluster_centers.npy
	rm -rf data/chroma_db/
	@echo "Done. Run 'make pipeline' to regenerate."
