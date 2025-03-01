import faiss
import numpy as np
import os
import json

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

# Predefined topics and answers
TOPIC_ANSWERS = {
    "social media": "Social media spreads information quickly, but not all of it is reliable.",
    "ai": "AI and machine learning are changing how we analyze data.",
    "fake news": "Fake news spreads misinformation and influences opinions.",
    "misinformation": "Misinformation can be harmful, especially when spread online."
}

# Query engine with topic matching
def query_engine(query, top_k=3):
    query = query.lower()

    # Check if query matches predefined topics
    for topic, response in TOPIC_ANSWERS.items():
        if topic in query:
            return [response]

    # If no predefined response, search FAISS
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

# Example usage
if __name__ == "__main__":
    while True:
        query = input("Ask me a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = query_engine(query)
        print("\n".join(response))
