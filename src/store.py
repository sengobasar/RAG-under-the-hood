import faiss
import numpy as np
import pickle
import os


def create_index(embeddings):

    # Dimension of vectors
    dim = embeddings.shape[1]

    # Inner Product index (for cosine similarity)
    index = faiss.IndexFlatIP(dim)

    # Add vectors
    index.add(embeddings.astype("float32"))

    return index


def save_all(index, chunks, path):

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, path)

    # Save chunks
    with open(path + ".pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Index and chunks saved.")


def load_all(path):

    if not os.path.exists(path) or not os.path.exists(path + ".pkl"):
        raise FileNotFoundError("Index or chunk file missing. Rebuild required.")

    # Load FAISS index
    index = faiss.read_index(path)

    # Load chunks
    with open(path + ".pkl", "rb") as f:
        chunks = pickle.load(f)

    print("✅ Index and chunks loaded.")

    return index, chunks