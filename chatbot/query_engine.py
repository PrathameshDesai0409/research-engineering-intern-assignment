import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Load FAISS index
FAISS_INDEX_PATH = "faiss_index/index.faiss"

def load_faiss_index():
    print("📌 Loading FAISS vector store...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("✅ FAISS index loaded successfully!")
    return index

def embed_query(query, model):
    """Embed the user query using the same model as before."""
    return np.array([model.embed_query(query)], dtype=np.float32)  # Reshape for FAISS

def query_engine(query, top_k=5):
    """Retrieves top_k similar results from FAISS based on user query."""
    if not query.strip():
        print("⚠️ Please enter a valid query.")
        return []

    # Load FAISS index
    index = load_faiss_index()

    # Load embedding model
    print("🔍 Initializing local embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Embed the query
    query_embedding = embed_query(query, embeddings_model)

    # Search in FAISS
    print(f"🔎 Searching for top {top_k} relevant results...")
    distances, indices = index.search(query_embedding, top_k)  # FAISS search

    return indices[0]  # Return the indices of the top results

if __name__ == "__main__":
    user_query = input("💡 Enter your search query: ")
    results = query_engine(user_query, top_k=5)
    print(f"🎯 Top matching results: {results}")
