from src.loader import load_pdf
from src.splitter import split_text
from src.embedder import embed_text
from src.store import create_index, save_all, load_all
from src.retriever import search

import os
import subprocess


# ---------------- CONFIG ---------------- #

PDF_PATH = "data/book.pdf"
INDEX_PATH = "embeddings/index"

OLLAMA_MODEL = "mistral"

# Full path to ollama.exe
OLLAMA_PATH = r"C:\Users\bowjo\AppData\Local\Programs\Ollama\ollama.exe"


# ---------------- BUILD INDEX ---------------- #

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


# ---------------- OLLAMA LLM ---------------- #

def call_ollama(prompt: str) -> str:

    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            timeout=300,
            encoding="utf-8",   # 🔥 FIX: force UTF-8
            errors="ignore"     # ignore encoding errors safely
        )

        if result.returncode != 0:
            return f"❌ Ollama Error:\n{result.stderr}"

        return result.stdout.strip()

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


# ---------------- PROMPT ---------------- #

def build_prompt(question, contexts):

    context_text = "\n\n".join(
        [f"[{i+1}] {c}" for i, c in enumerate(contexts)]
    )

    prompt = f"""
You are a knowledgeable assistant.

Use ONLY the context below to answer the question.

The context comes from OCR and may contain noise.
Ignore spacing and formatting errors.

Context:
{context_text}

Question:
{question}

Instructions:
- Base your answer only on the context.
- Do not invent facts.
- If the answer is not present, say: "Not found in the text."
- Answer clearly and simply.

Answer:
"""

    return prompt.strip()


# ---------------- ASK ---------------- #

def ask(question, index, chunks):

    # Embed query
    q_vec = embed_text([question])[0]

    # Retrieve documents
    contexts = search(index, q_vec, chunks)

    # Build LLM prompt
    prompt = build_prompt(question, contexts)

    print("\n🤖 Thinking...\n")

    # Call LLM
    answer = call_ollama(prompt)

    return answer


# ---------------- MAIN ---------------- #

def main():

    index, chunks = load_or_build()

    print("\n✅ RAG + Open-Source LLM Ready")
    print("Type 'exit' to quit.\n")

    while True:

        q = input("Ask: ").strip()

        if not q:
            continue

        if q.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        answer = ask(q, index, chunks)

        print("\n📖 Answer:\n")
        print(answer)

        print("\n-----------------------------\n")


if __name__ == "__main__":
    main()