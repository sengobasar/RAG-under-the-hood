# 🧠 RAG — The Real Math Behind It

> **Retrieval-Augmented Generation** explained from first principles.
> No fluff. No "just use LangChain bro."
> Pure math, pure pipeline, pure understanding.
<img width="1919" height="1020" alt="image" src="https://github.com/user-attachments/assets/c69b9318-f91a-4f6f-b9aa-8f73afc8d4d1" />
RAW OCR OUTPUT:=
<img width="1919" height="987" alt="Screenshot 2026-02-24 013725" src="https://github.com/user-attachments/assets/d33a850e-249c-4f67-aedd-9060e2358b8c" />


---

## 🗺️ The Full Journey

```
PDF → [OCR if needed ⚠️] → Text → Numbers → Vectors → Similarity → Retrieval → Rerank → LLM → Answer
```

Every arrow is math. Let's walk through each one.

> ⚠️ **The OCR warning lives at the start because it should.**
> OCR is the most painful, most misunderstood step in the whole pipeline.
> See [Section 0](#0️⃣-ocr--the-step-you-want-to-avoid) before anything else.

---

## 🛠️ Tech Stack Reference

| Purpose        | Tool                                  | Notes                          |
|----------------|---------------------------------------|--------------------------------|
| Language       | Python 3.9+                           |                                |
| PDF Reading    | pdfplumber / PyPDF2                   | Use this first — no OCR needed |
| OCR (fallback) | Tesseract OCR 5.x via pytesseract     | ⚠️ Only if PDF has no text layer |
| OCR Engine     | LSTM neural engine (`--oem 1`)        | Tesseract's most accurate mode |
| Embeddings     | SentenceTransformers / OpenAI / HF    |                                |
| Vector DB      | FAISS or Chroma                       |                                |
| Reranking      | cross-encoder/ms-marco-MiniLM-L-12-v2 |                                |
| LLM (local)    | **Ollama** — Mistral / LLaMA 3        | Runs fully offline, no API key |
| LLM (cloud)    | OpenAI / HuggingFace                  | Optional alternative           |
| Framework      | LangChain (optional)                  |                                |
| Env Mgmt       | venv / pip                            |                                |

---

## 0️⃣ OCR — The Step You Want to Avoid

### What Is OCR?

**OCR (Optical Character Recognition)** is the process of converting an image of text
into machine-readable characters.

In the RAG pipeline, you need it when your PDF is a **scanned image** —
a photograph of a page — with no selectable text layer underneath.

```
Scanned image of page  →  OCR  →  "Rice grows in Assam during monsoon."
```

### What We Use: Tesseract OCR 5.x

In this project, OCR is handled by:

| Component     | Detail                                      |
|---------------|---------------------------------------------|
| Engine        | **Tesseract OCR 5.x**                       |
| Python bridge | **pytesseract**                             |
| OCR mode      | **LSTM neural engine** (`--oem 1`)          |
| Page mode     | `--psm 3` (fully automatic page segmentation) |

Tesseract 5.x replaced the older pattern-matching engine with an **LSTM-based neural network**,
giving significantly better accuracy on low-quality scans, rotated text, and noisy documents.

```python
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

def ocr_pdf(pdf_path: str) -> str:
    """
    Convert a scanned PDF to text using Tesseract LSTM engine.
    Only use this if pdfplumber returns empty or garbled text.
    """
    pages = convert_from_path(pdf_path, dpi=300)   # higher DPI = better accuracy
    full_text = []
    for page_img in pages:
        text = pytesseract.image_to_string(
            page_img,
            lang="eng",           # change to your language code
            config="--oem 1 --psm 3"  # oem 1 = LSTM only
        )
        full_text.append(text)
    return "\n".join(full_text)
```

**Tesseract OEM modes explained:**

| `--oem` | Engine           | Use when                          |
|---------|------------------|-----------------------------------|
| 0       | Legacy only      | Very old Tesseract installations  |
| 1       | LSTM only ✅     | **Default — best accuracy**       |
| 2       | Legacy + LSTM    | Experimental                      |
| 3       | Default (auto)   | Let Tesseract decide              |

---

### ❌ Is OCR Good for RAG? Honest Answer: No.

| Problem              | Why It Hurts RAG                                          |
|----------------------|-----------------------------------------------------------|
| **Slow**             | OCR a 100-page PDF can take 3–10 minutes                  |
| **Noisy output**     | Misread characters corrupt embeddings downstream          |
| **Layout breaks**    | Tables, columns, headers get scrambled into gibberish     |
| **No structure**     | Page numbers, headers, footnotes mix into the main text   |
| **Language gaps**    | Low-resource languages (like Galo) have no trained model  |

OCR errors are **silent**. The pipeline does not crash — it just embeds garbage,
and retrieval silently returns wrong chunks. This is the worst kind of failure.

---

### ✅ When Should You Use OCR?

```
Ask yourself: can pdfplumber extract clean text from this PDF?
```

```python
import pdfplumber

def has_text_layer(pdf_path: str) -> bool:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                return True   # ✅ native text layer exists — no OCR needed
    return False              # ❌ scanned image — OCR required
```

| PDF Type                        | Action                           |
|---------------------------------|----------------------------------|
| Born-digital PDF (Word → PDF)   | ✅ Use pdfplumber directly       |
| Scanned document (image PDF)    | ⚠️ Use Tesseract OCR 5.x + LSTM |
| Mixed (some pages are images)   | ⚠️ Detect per-page, use hybrid  |

---

### 🔑 The Golden Rule

> **Avoid OCR if you can. Use it only when forced.**
> Born-digital PDFs give you clean text for free.
> OCR is a last resort — not a default step.

OCR should live entirely inside **Phase 1 (Ingestion)** — run once, cached,
never touched again during query time.

```
❌ Wrong:  every query → OCR → embed → answer   (slow, broken)
✅ Right:  once → OCR → embed → save index → every query loads cached index
```

---

### 📊 OCR vs Direct Extraction — Speed Reality

| Method              | 100-page PDF   | Per query after caching |
|---------------------|----------------|-------------------------|
| pdfplumber (native) | ~2 seconds     | 0 ms (never re-runs)    |
| Tesseract OCR 5.x   | 3–10 minutes   | 0 ms (never re-runs)    |
| OCR every query ❌  | 3–10 min/query | 💀 Not viable           |

This is why caching the index after ingestion is not optional — it is the architecture.

---

## 1️⃣ Text → Numbers (Tokenization + Embeddings)

Suppose your document chunk is:

```
"Rice grows in Assam during monsoon."
```

### Step 1 — Tokenization

The text is split into tokens:

```python
["Rice", "grows", "in", "Assam", "during", "monsoon"]
```

Each token is mapped to an integer ID from a vocabulary:

```python
[5023, 9012, 321, 18492, 7721, 5512]
```

### Step 2 — Embedding Lookup

There is an **embedding matrix**:

$$E \in \mathbb{R}^{V \times d}$$

Where:
- $V$ = vocabulary size (e.g. 30,000–50,000 tokens)
- $d$ = embedding dimension (e.g. **768** for BERT)

Each token ID is a **row lookup** into this matrix, giving an initial vector:

$$x_i^{(0)} \in \mathbb{R}^{768}$$

> ⚠️ **Important precision note:** In modern transformers, this lookup is just the *starting point*.
> The real magic happens next.

### Step 3 — Positional Encoding

Transformers have no built-in sense of word order.
So positional information is added to each token vector:

$$x_i^{(0)} = \text{TokenEmbed}(t_i) + \text{PosEmbed}(i)$$

### Step 4 — Transformer Layers (Contextualization)

The token vectors pass through $L$ attention layers (e.g. 12 in BERT-base):

$$x_i^{(l)} = \text{TransformerBlock}^{(l)}\!\left(x_1^{(l-1)}, \ldots, x_n^{(l-1)}\right)$$

At each layer, every token **attends to every other token**.
The word "grows" sees "Rice", "Assam", "monsoon" — and updates its own vector accordingly.

So the **final vector for each token** is:

$$x_i = x_i^{(L)} \in \mathbb{R}^{768}$$

This is a **contextualized embedding** — not a static word lookup.
The vector for "bank" in "river bank" is numerically different from "bank" in "savings bank."
That is the key power of transformer-based embeddings.

---

## 2️⃣ Numbers → Vectors (Sentence Embedding)

We need **one vector** to represent the entire chunk.
Two main strategies:

### Method A — Mean Pooling

$$v_{\text{doc}} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

Where:
- $n$ = number of tokens
- $x_i$ = **final-layer** contextualized embedding of token $i$

For our 6-token sentence:

$$v_{\text{doc}} = \frac{x_1 + x_2 + x_3 + x_4 + x_5 + x_6}{6}$$

Result:

$$v_{\text{doc}} \in \mathbb{R}^{768}$$

> **Note:** SentenceTransformers may also use **weighted pooling** or **attention pooling**
> depending on the model — plain mean is the most common but not the only approach.

### Method B — CLS Token (BERT-style)

BERT prepends a special `[CLS]` token before the sequence.
After all transformer layers, its final hidden state is used:

$$v_{\text{doc}} = h_{[\text{CLS}]}^{(L)}$$

The model is trained so that $h_{[\text{CLS}]}$ aggregates full sentence-level meaning.

Either way, the result is one vector representing the entire chunk:

$$v_{\text{doc}} \in \mathbb{R}^{768}$$

---

## 3️⃣ Query → Vector (Same Process)

User asks:

```
"When does rice grow in Assam?"
```

The **exact same pipeline** runs:

```
Tokenize → Embed → Transformer Layers → Pool
```

Result:

$$v_{\text{query}} \in \mathbb{R}^{768}$$

Now we have two points in the same 768-dimensional space.
One for the document. One for the query.
RAG needs to find which document points are **closest** to the query point.

---

## 4️⃣ Vectors → Similarity

### Cosine Similarity (Most Common)

$$\text{cosine}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|}$$

Where:

$$q \cdot d = \sum_{i=1}^{768} q_i \, d_i \quad \text{(dot product)}$$

$$\|q\| = \sqrt{\sum_{i=1}^{768} q_i^2} \quad \text{(L2 norm)}$$

**Range:** $-1 \leq \text{cosine}(q, d) \leq 1$

| Score | Meaning                |
|-------|------------------------|
| 1.0   | Identical direction    |
| 0.0   | Orthogonal (unrelated) |
| -1.0  | Opposite direction     |

Cosine measures the **angle** between two vectors — ignoring their lengths.
This is useful because in practice, vector magnitude is not a reliable semantic signal.

> ⚠️ **Honest precision note:** In some models, magnitude can encode salience or
> confidence. This is an active research area. For practical RAG, treating cosine
> as the right default is correct.

### Dot Product (Faster, Used in FAISS)

$$\text{score}(q, d) = q^\top d = \sum_{i=1}^{768} q_i \, d_i$$

When all vectors are **pre-normalised** to unit length ($\|v\| = 1$),
dot product equals cosine similarity — with less computation.
This is why FAISS workflows always call `faiss.normalize_L2()` before indexing.

---

## 5️⃣ Similarity → Retrieval (Vector Database Search)

Suppose your vector database holds $N$ document chunks:

$$\mathcal{D} = \{d_1, d_2, \ldots, d_N\} \quad \text{where each } d_i \in \mathbb{R}^{768}$$

When query $q$ arrives, find the **Top-K most similar documents**:

$$\text{Top-K} = \underset{i}{\operatorname{arg\,topK}} \; \text{score}(q, d_i)$$

This is **nearest-neighbour search** in high-dimensional space.

### Computational Complexity

| Method           | Time Complexity     | Notes                                    |
|------------------|---------------------|------------------------------------------|
| Brute force      | $O(N \cdot d)$      | Exact, slow for large $N$                |
| FAISS IVF        | $\approx O(\sqrt{N})$ | Approximate — clusters vectors first   |
| FAISS HNSW       | $O(\log N)$         | Graph-based, very fast at query time     |

> ⚠️ **ANN complexity note:** Sub-linear holds in average case.
> Worst-case for approximate methods can still approach $O(N)$ depending on
> data distribution. For most real corpora, average-case performance dominates.

---

## 6️⃣ Why Vectors Capture Meaning

Embedding models are trained so that:

$$\text{similar meaning} \Rightarrow \text{vectors close in space}$$

The training objective is typically **Contrastive Loss** (InfoNCE / NT-Xent):

$$\mathcal{L} = -\log \frac{\exp\!\left(\text{sim}(q, d^+) / \tau\right)}{\sum_{j=1}^{B} \exp\!\left(\text{sim}(q, d_j) / \tau\right)}$$

Where:
- $d^+$ = the correct matching document (**positive pair**)
- $d_j$ = all documents in the training batch (**$B$ total, includes negatives**)
- $\tau$ = temperature hyperparameter (controls sharpness — lower = sharper peaks)
- $\text{sim}$ = cosine similarity

**What this does geometrically:**

```
Before training:  vectors are randomly scattered
After training:   similar-meaning vectors cluster together
```

- Pulls correct pairs $(q, d^+)$ → **closer together** ↓ distance
- Pushes wrong pairs $(q, d_j)$ → **farther apart** ↑ distance

After training, the vector space becomes a **semantic map**.
"Rice grows in monsoon" and "Paddy cultivation in rainy season" land near each other
even though they share **zero words in common.**

---

## 6.5️⃣ Reranking — The Precision Upgrade

Dense retrieval (Top-K cosine search) is **fast but approximate**.
It finds vectors that are directionally close — but similarity in embedding space
does not always perfectly match actual relevance to the question.

**The fix:** run a **cross-encoder** on the retrieved Top-K results.

### Bi-encoder (what retrieval does)

$$\text{score}(q, d) = E_q(q)^\top E_d(d)$$

Query and document are encoded **independently**, then compared.
Fast. Scalable. Used for retrieval over millions of chunks.

### Cross-encoder (what reranking does)

$$\text{score}(q, d) = \text{Transformer}([q \,;\, d])$$

Query and document are **concatenated and processed together**.
Every attention head sees both at once — capturing fine-grained interaction.

$$\text{Reranked Top-K} = \underset{k}{\operatorname{arg\,topK}} \; \text{CrossEncoder}(q, d_k)$$

**In practice:**

```
Dense Retrieval (Top-50) → Cross-Encoder Rerank → Top-3 → LLM
```

| Stage          | Model example                          | Speed   | Precision |
|----------------|----------------------------------------|---------|-----------|
| Retrieval      | all-MiniLM-L6-v2                       | ⚡ Fast  | Good      |
| Reranking      | ms-marco-MiniLM-L-12-v2               | 🐢 Slower | Excellent |

Reranking typically improves answer precision by **20–40%** with no change to the LLM.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# After dense retrieval gives you top_k_chunks:
pairs = [[question, chunk] for chunk in top_k_chunks]
scores = reranker.predict(pairs)

# Sort by reranker score and take the best
ranked = sorted(zip(scores, top_k_chunks), reverse=True)
final_context = [chunk for _, chunk in ranked[:3]]
```

---

## 7️⃣ Retrieved Text → LLM (Full RAG Equation)

The original RAG formulation (Lewis et al., 2020):

$$P(y \mid x) = \sum_{z \in \mathcal{D}} P(y \mid x, z) \cdot P(z \mid x)$$

Where:
- $x$ = the user's query
- $z$ = a retrieved document chunk
- $y$ = the generated answer

**Interpretation:**
1. $P(z \mid x)$ — retrieval probability: how likely is chunk $z$ relevant to query $x$?
2. $P(y \mid x, z)$ — generation probability: given query + chunk, how likely is answer $y$?

### Practical Top-K Approximation

Marginalising over all $\mathcal{D}$ is intractable.
In practice, approximate with the Top-K retrieved (and reranked) chunks:

$$P(y \mid x) \approx \sum_{k=1}^{K} P(y \mid x, z_k) \cdot P(z_k \mid x)$$

### How Production Systems Do It

Most systems skip the probabilistic sum and simply **concatenate** top-K chunks:

$$\text{Prompt} = x \oplus z_1 \oplus z_2 \oplus \cdots \oplus z_K$$

The LLM then computes:

$$P(y \mid x, z_1, z_2, \ldots, z_K)$$

The LLM performs implicit weighting through **attention over the concatenated context**
and **positional bias** — tokens appearing earlier in context tend to receive more weight.
This is why chunk ordering in the prompt matters.

---

## 8️⃣ Geometric Intuition — What Is Actually Happening

```
1. Map every document chunk  →  a point in ℝ⁷⁶⁸
2. Map the user query        →  a point in ℝ⁷⁶⁸
3. Find nearest document points to the query point
4. (Optional) Rerank those points with a cross-encoder
5. Hand the best chunks (as text) to the LLM
6. LLM generates an answer conditioned on query + context
```

**The formal equation:**

$$\boxed{f(x) = \text{LLM}\!\left(x \oplus \text{Rerank}\!\left(\text{NN}(E(x),\, E(\mathcal{D}))\right)\right)}$$

Without reranking (simpler form):

$$f(x) = \text{LLM}\!\left(x \oplus \text{NN}(E(x),\, E(\mathcal{D}))\right)$$

Where:
- $E(\cdot)$ = embedding function (transformer encoder)
- $\text{NN}(\cdot)$ = nearest-neighbour search (FAISS)
- $\text{Rerank}(\cdot)$ = cross-encoder rescoring (optional but powerful)
- $\oplus$ = concatenation into prompt

**RAG = High-dimensional geometry + optional cross-encoder rescoring + autoregressive decoding.**

---

## ⚡ The Speed Problem — And How to Fix It

### ❌ Prototype Mode (What You Probably Do Now)

Every single query re-runs everything:

```
PDF → OCR → Chunk → Embed corpus → Search → Answer
```

| Step              | Cost         | Runs every query? |
|-------------------|--------------|-------------------|
| OCR               | Very slow ❌  | Yes ❌             |
| Chunking          | Slow ❌       | Yes ❌             |
| Embed whole corpus | Slow ❌      | Yes ❌             |
| Search index      | Fast ✅       | Yes ✅             |
| LLM call          | Medium       | Yes ✅             |

Fine for learning. Not scalable for production.

---

### ✅ The Two-Phase Architecture

RAG has two completely separate phases.
Run them separately. **Cache the expensive one.**

---

#### Phase 1 — Ingestion (Run ONCE)

```
PDF → [OCR if scanned ⚠️] → Text → Chunk → Embed → FAISS Index → Save to Disk
```

Slow ❌ — but only runs **once** (or when documents change).
The OCR step uses **Tesseract 5.x with the LSTM engine** when the PDF has no native text layer.

```python
# ingestion.py  —  run this ONCE, not every query

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

model = SentenceTransformer("all-MiniLM-L6-v2")


def has_text_layer(pdf_path: str, min_chars: int = 50) -> bool:
    """
    Returns True if at least one page has a real text layer.
    min_chars guards against PDFs that have a text layer but it is
    just whitespace or a handful of stray characters.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) >= min_chars:
                return True
    return False


def extract_with_pdfplumber(pdf_path: str) -> str:
    """Fast path — born-digital PDF with a native text layer."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [
            page.extract_text()
            for page in pdf.pages
            if page.extract_text()
        ]
    return "\n".join(pages_text)


def extract_with_tesseract(pdf_path: str) -> str:
    """
    Slow path — scanned/image PDF with no text layer.
    Uses Tesseract OCR 5.x with LSTM neural engine (--oem 1).
    DPI=300 gives the best accuracy for typical document scans.
    """
    print("⚠️  No text layer found. Running Tesseract OCR 5.x (LSTM engine)...")
    pages = convert_from_path(pdf_path, dpi=300)
    return "\n".join(
        pytesseract.image_to_string(
            page,
            lang="eng",
            config="--oem 1 --psm 3",  # oem 1 = LSTM only, psm 3 = auto layout
        )
        for page in pages
    )


def extract_text(pdf_path: str) -> str:
    """
    Smart extractor: tries pdfplumber first.
    Only falls back to Tesseract OCR when no text layer is detected.
    This is the function that makes README and code agree.
    """
    if has_text_layer(pdf_path):
        print("✅ Text layer detected. Using pdfplumber (fast, no OCR).")
        return extract_with_pdfplumber(pdf_path)
    else:
        return extract_with_tesseract(pdf_path)


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# --- Main ingestion ---
raw_text  = extract_text("your_document.pdf")   # smart: pdfplumber or Tesseract
chunks    = chunk_text(raw_text)

# Embed entire corpus — slow, but runs ONCE
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Normalise so dot product == cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS flat index (swap for IndexIVFFlat on large corpora)
dimension = embeddings.shape[1]   # 384 for MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save everything — OCR and embedding never run again after this
faiss.write_index(index, "vector_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ {len(chunks)} chunks saved. Index ready. Query phase is now instant.")
```

---

#### Phase 2 — Query (Every User Request)

```
Question → Embed query → FAISS Search → (Rerank) → Top-K Chunks → LLM → Answer
```

Fast ⚡ — no OCR, no re-embedding the corpus.

```python
# query.py  —  this runs on every user question

import faiss
import pickle
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Load everything once at startup ──────────────────────────────────
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

index = faiss.read_index("vector_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ── Ollama config ─────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"          # swap to "llama3" or any model you have pulled


def retrieve(question: str, top_k: int = 20) -> list[str]:
    """Dense retrieval: embed query → search FAISS index."""
    q_vec = bi_encoder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    _, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0]]


def rerank(question: str, candidates: list[str], top_n: int = 3) -> list[str]:
    """Cross-encoder reranking: scores each candidate against the question."""
    pairs  = [[question, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Assemble the RAG prompt: context first, then question."""
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"You are a helpful assistant. "
        f"Use ONLY the context below to answer the question. "
        f"If the answer is not in the context, say 'I don't know based on the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Send the prompt to a locally running Ollama instance.
    Ollama streams tokens by default — we collect them all and return
    the complete answer as a single string.
    """
    payload  = {"model": model, "prompt": prompt, "stream": True}
    response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()

    full_answer = []
    for line in response.iter_lines():
        if line:
            token_data = json.loads(line)
            full_answer.append(token_data.get("response", ""))
            if token_data.get("done"):
                break

    return "".join(full_answer).strip()


def answer(
    question: str,
    show_context: bool = False,   # set True to inspect retrieved chunks
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve top-20 candidates from FAISS
      2. Rerank → keep top-3
      3. Build prompt
      4. Send to Ollama (Mistral / LLaMA 3)
      5. Return clean answer
    """
    candidates  = retrieve(question, top_k=20)
    best_chunks = rerank(question, candidates, top_n=3)
    prompt      = build_prompt(question, best_chunks)

    if show_context:
        print("\n── Retrieved context ──────────────────────────────")
        for i, chunk in enumerate(best_chunks, 1):
            print(f"\n[Chunk {i}]\n{chunk}")
        print("\n── Sending to Ollama ───────────────────────────────\n")

    return ask_ollama(prompt)


# ── Example usage ─────────────────────────────────────────────────────
if __name__ == "__main__":
    question = "When does rice grow in Assam?"

    print(f"Question: {question}\n")
    result = answer(question, show_context=True)   # flip to False for clean output only
    print(f"\nAnswer:\n{result}")
```

---

### 🦙 Ollama — Running LLMs Locally

This project uses **Ollama** to run the LLM fully offline.
No API key. No cloud. No cost per token.

```
Prompt  →  Ollama (local)  →  Mistral / LLaMA 3  →  Answer
```

**What Ollama does in this pipeline:**

| Step | What happens |
|------|-------------|
| ✅ Uses your existing FAISS pipeline | Retrieval is unchanged |
| ✅ Retrieves context via FAISS + reranker | Best 3 chunks selected |
| ✅ Builds the RAG prompt | Context + question concatenated |
| ✅ Sends prompt to Ollama | Via `http://localhost:11434/api/generate` |
| ✅ Streams tokens, returns clean answer | Full response assembled |
| ✅ Optional context inspection | Set `show_context=True` to debug |

**Setup (one time):**

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (pick one)
ollama pull mistral       # recommended — fast, accurate, 4GB
ollama pull llama3        # stronger, 8GB
ollama pull phi3          # lightweight, 2GB — good for low-RAM machines

# 3. Ollama runs as a background service automatically
# Confirm it is alive:
curl http://localhost:11434
```

**Switching models** — change one line in `query.py`:

```python
OLLAMA_MODEL = "mistral"    # default
OLLAMA_MODEL = "llama3"     # stronger
OLLAMA_MODEL = "phi3"       # smaller / faster
```

**How `ask_ollama()` works:**

Ollama streams responses token by token over HTTP.
The function collects every token and joins them into a single clean string:

```
HTTP POST /api/generate
    → stream of {"response": "Rice", "done": false}
    → stream of {"response": " grows", "done": false}
    → ...
    → {"response": ".", "done": true}
    → join all → "Rice grows in Assam during monsoon."
```

This is why `stream=True` is set on the requests call —
it is faster than waiting for the whole response at once.

---

### 📊 Speed Comparison

| Step                           | Ingestion        | Query       |
|--------------------------------|------------------|-------------|
| PDF text extraction (native)   | ~2 sec ✅ once   | ❌ Never    |
| Tesseract OCR 5.x (if needed)  | 3–10 min ⚠️ once | ❌ Never    |
| Chunk text                     | ✅ Once          | ❌ Never    |
| Embed corpus                   | ✅ Once          | ❌ Never    |
| Embed single query             | —                | ~5 ms ✅    |
| FAISS search (10k chunks)      | —                | ~2 ms ✅    |
| Cross-encoder rerank           | —                | ~50 ms ✅   |
| LLM generation                 | —                | varies ✅   |

Your query phase becomes **near-instant** after ingestion is cached.
OCR — however slow — runs **exactly once** and is never seen again.

---

## 9️⃣ Complete Pipeline — Visual

```
┌──────────────────────────────────────────────────────────────────┐
│                   INGESTION  (run once)                          │
│                                                                  │
│  PDF ──► has text layer? ──YES──► pdfplumber ──► raw text        │
│               │                                      │           │
│               NO                                     │           │
│               │                                      │           │
│               ▼                                      │           │
│        Tesseract OCR 5.x                             │           │
│        (LSTM engine --oem 1) ───────────────────► raw text       │
│                                                      │           │
│                                                      ▼           │
│                                                   Chunking       │
│                                                      │           │
│                                                      ▼           │
│                                              Bi-Encoder embed    │
│                                                      │           │
│                                                      ▼           │
│                                    FAISS Index + chunks.pkl      │
│                                         saved to disk ──────►    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│               QUERY PHASE  (every user request)                  │
│                                                                  │
│  Question                                                        │
│     │                                                            │
│     ▼                                                            │
│  Bi-Encoder (embed query ~5ms)                                   │
│     │                                                            │
│     ▼                                                            │
│  FAISS Search → Top-20 candidates (~2ms)                         │
│     │                                                            │
│     ▼                                                            │
│  Cross-Encoder Rerank → Top-3 chunks (~50ms)   ← optional       │
│     │                                                            │
│     ▼                                                            │
│  build_prompt()  =  Question ⊕ Top-3 chunks                     │
│     │                                                            │
│     ▼                                                            │
│  ask_ollama()  →  Ollama (localhost:11434)                       │
│                      │                                           │
│              Mistral / LLaMA 3 / Phi-3                          │
│                      │                                           │
│                      ▼                                           │
│                   Answer ✅                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Summary of All Key Equations

| What | Equation |
|------|----------|
| Token embedding | $x_i^{(0)} = \text{TokenEmbed}(t_i) + \text{PosEmbed}(i)$ |
| Contextualized vector | $x_i = \text{TransformerBlock}^{(1..L)}(x_1^{(0)}, \ldots, x_n^{(0)})$ |
| Document vector (mean pool) | $v_{\text{doc}} = \frac{1}{n}\sum_{i=1}^n x_i$ |
| Cosine similarity | $\text{sim}(q,d) = \frac{q \cdot d}{\|q\|\|d\|}$ |
| Top-K retrieval | $\operatorname{arg\,topK}_i\ \text{sim}(q, d_i)$ |
| Contrastive loss | $\mathcal{L} = -\log\frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_j \exp(\text{sim}(q,d_j)/\tau)}$ |
| Cross-encoder score | $\text{score}(q,d) = \text{Transformer}([q\,;\,d])$ |
| RAG (full) | $P(y\|x) = \sum_z P(y\|x,z)\,P(z\|x)$ |
| RAG (practical) | $f(x) = \text{LLM}(x \oplus \text{Rerank}(\text{NN}(E(x), E(\mathcal{D}))))$ |

---

## 🔥 What to Explore Next

| Topic | What It Answers |
|---|---|
| Why cosine works geometrically | Why angle matters more than magnitude |
| Curse of dimensionality | Why high-dim vectors don't collapse into noise |
| IVF indexing | How FAISS clusters space to skip most comparisons |
| HNSW indexing | Graph-based ANN — how it achieves $O(\log N)$ search |
| Gradient flow in RAG | Why gradients don't flow through retrieval in basic RAG |
| HyDE | Hypothetical Document Embeddings — query augmentation trick |
| RAG-Fusion | Query expansion + reciprocal rank fusion |
| RAGAS | How to evaluate RAG system quality automatically |

---

*Built for the Galo Language AI Project.*
*Because the math should be as clear as the mission.*
*OCR is a necessary evil — respect it, cache it, never run it twice.*
