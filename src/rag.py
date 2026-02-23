import pickle
import faiss
import numpy as np

from loader import load_pdf
from splitter import split_text
from embedder import embed_text


DATA_PATH = "data/book.pdf"
INDEX_PATH = "embeddings/index.faiss"
CHUNKS_PATH = "embeddings/index.pkl"


# ---------------- INGEST ---------------- #

def build_index():

    print("Loading PDF...")
    text = load_pdf(DATA_PATH)

    print("Splitting text...")
    chunks = split_text(text)

    print(f"Total chunks: {len(chunks)}")

    print("Embedding...")
    embeddings = embed_text(chunks)

    dim = embeddings.shape[1]

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Ingestion complete.")


# ---------------- LOAD ---------------- #

def load_index():

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# ---------------- RETRIEVE ---------------- #

def retrieve(query, index, chunks, top_k=5):

    q_vec = embed_text([query])

    scores, ids = index.search(q_vec, top_k)

    results = []

    for i in ids[0]:
        if i >= 0:
            results.append(chunks[i])

    return results


# ---------------- ASK ---------------- #

def ask(question):

    index, chunks = load_index()

    contexts = retrieve(question, index, chunks)

    print("\n--- Retrieved Context ---\n")

    for i, c in enumerate(contexts, 1):
        print(f"[{i}] {c[:500]}...\n")

    return contexts


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag.py ingest")
        print("  python rag.py ask \"your question\"")
        exit()

    cmd = sys.argv[1]

    if cmd == "ingest":
        build_index()

    elif cmd == "ask":

        question = " ".join(sys.argv[2:])
        ask(question)

    else:
        print("Unknown command.")