import os
import time
import faiss
import json
import numpy as np
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Constants
DATA_PATH = "data/dataset.jsonl"
VECTOR_STORE_PATH = "faiss_index"
BATCH_SIZE = 64  # Optimized batch size for performance

def batch_embed(texts, model):
    """Embeds text in batches for efficiency."""
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🔄 Processing Batches"):
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_embeddings = model.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype=np.float32)  # Convert to NumPy array for FAISS

def load_documents():
    """Loads documents from a JSONL file."""
    print("🚀 Loading dataset...")
    
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Dataset file not found at {DATA_PATH}!")
        return []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [json.loads(line).get("text", "").strip() for line in f if line.strip()]

    if not texts:
        print("⚠️ No valid text data found in dataset.jsonl!")
        return []

    print(f"📄 Loaded {len(texts)} documents.")
    return [Document(page_content=text) for text in texts]

def create_vector_store():
    """Creates a FAISS vector store with embeddings."""
    docs = load_documents()
    if not docs:
        return

    print("🔍 Initializing local embeddings model...")
    start_time = time.time()
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("⚡ Generating embeddings in batches...")
    embedded_texts = batch_embed([doc.page_content for doc in docs], embeddings_model)
    print(f"✅ Embeddings generated in {time.time() - start_time:.2f} seconds.")

    print("📌 Creating FAISS vector store...")
    start_time = time.time()

    # Create FAISS index
    dimension = embedded_texts.shape[1]  # Get embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 similarity search
    index.add(embedded_texts)  # Add embeddings

    # Save FAISS index
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_STORE_PATH, "index.faiss"))

    print(f"✅ FAISS vector store created in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    create_vector_store()
