import faiss
import numpy as np
import os
import json
import pandas as pd

# Paths
FAISS_INDEX_PATH = "faiss_index/index.faiss"
DATA_PATH = "data/dataset.jsonl"

# Load FAISS index
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}!")
    return faiss.read_index(FAISS_INDEX_PATH)

# Load dataset
def load_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset file missing: {DATA_PATH}")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [json.loads(line).get("text", "").strip() for line in f if line.strip()]

    return texts

# Query engine
def query_engine(query, top_k=3):
    index = load_faiss_index()
    documents = load_documents()

    # Convert query to NumPy vector (dummy example, replace with actual embedding)
    query_vector = np.random.rand(index.d)  # Replace with real embedding model
    query_vector = np.expand_dims(query_vector, axis=0)

    # Perform search
    distances, indices = index.search(query_vector, top_k)

    # Retrieve top documents
    results = [documents[idx] for idx in indices[0] if idx < len(documents)]
    return results if results else ["No relevant information found."]
