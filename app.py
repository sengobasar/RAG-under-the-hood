from src.loader import load_pdf
from src.splitter import split_text
from src.embedder import embed_text
from src.store import create_index, save_all, load_all
from src.retriever import search

import os


PDF_PATH = "data/book.pdf"
INDEX_PATH = "embeddings/index"


def build_index():

    print("Loading PDF (OCR will run once)...")
    text = load_pdf(PDF_PATH)

    print("Splitting text...")
    chunks = split_text(text)

    print("Embedding...")
    embeddings = embed_text(chunks)

    print("Creating index...")
    index = create_index(embeddings)

    print("Saving index and chunks...")
    save_all(index, chunks, INDEX_PATH)

    return index, chunks


def load_or_build():

    if not os.path.exists(INDEX_PATH + ".faiss"):
        return build_index()
    else:
        print("Loading saved index...")
        return load_all(INDEX_PATH)


def ask(question, index, chunks):

    q_vec = embed_text([question])[0]

    docs = search(index, q_vec, chunks)

    return docs


def main():

    index, chunks = load_or_build()

    while True:

        q = input("\nAsk: ")

        if q.lower() == "exit":
            break

        docs = ask(q, index, chunks)

        print("\nContext:\n")

        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d[:400]}...\n")


if __name__ == "__main__":
    main()