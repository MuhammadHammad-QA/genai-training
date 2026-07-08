# Day 5 — Embeddings & Open-Source Models

**Topic:** Vector DB selection and open-source LLM integration.
**Task:** Ingest documents into Chroma DB using word-based chunking, then compare a
basic RAG pipeline (retrieval) against a full-text (no-retrieval) baseline using an
open-source LLM.

---

## 1. Setup

| Component | Choice |
|---|---|
| Document | `sample_pdf.pdf` (~28,393 characters) |
| Vector DB | Chroma (persistent, `chroma_db/`) |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| LLM | `llama3.2` via __Ollama__ (self-hosted, local API) |
| Test set | `golden_test_set.csv` (26 Q&A pairs) |
| Retrieval depth | `TOP_K = 3` |

Two scripts:

- `build_vector_db.py` — loads the PDF, chunks it, embeds, stores in Chroma.
- `rag_vs_fulltext.py` — runs both approaches over the test set and scores them.

---

## 2. Chunking Strategy

Chunking is **word-based** (not character-based): the text is split into words and
grouped into fixed-size windows with overlap.

```sh
CHUNK_SIZE    = 80 words
CHUNK_OVERLAP = 20 words
```

### Why 80 words?

The initial version used 200-word chunks and retrieval performed poorly. Root cause:
`all-MiniLM-L6-v2` accepts a maximum of **256 tokens**, but a 200-word chunk from this
PDF tokenized to **618 tokens** — so ~60% of every chunk was silently truncated before
embedding. The chunk text was stored in full, but its **search vector only represented
the first ~85 words**, making the rest of the text unretrievable by meaning.

Reducing to 80 words (~250 tokens) keeps each chunk under the model's limit, so the
embedding represents the whole chunk. This raised the chunk count from 27 → 71.

---

## 3. Retrieval Quality Fixes

Two additional issues were corrected in the ingestion pipeline:

| Fix | Reason |
|---|---|
| `normalize_embeddings=True` (index __and__ query side) | `all-MiniLM-L6-v2` is trained for cosine similarity; vectors must be normalized. |
| `metadata={"hnsw:space": "cosine"}` on the collection | Chroma defaults to __L2__ distance; MiniLM ranks best under __cosine__. |

The query embedding in `rag_vs_fulltext.py` is normalized to match the index — otherwise
indexing and querying would use different vector scales.

---

## 4. Evaluation Metrics

Each question is answered by both approaches and scored on:

- **Semantic Similarity (%)** — cosine similarity between the expected and candidate
   answer embeddings (replaced exact string match, which is too strict for free-form LLM output).
- **Keyword Match (%)** — overlap of expected keywords found in the answer.
- **Avg Length Difference** — how far answer length deviates from the expected (lower is better).
- **LLM Judge (/5)** — the LLM grades each answer against the expected answer.

---

## 5. Results

```sh
======================================================================
RAG vs FULL-TEXT EVALUATION
======================================================================
Metric                                 Full-Text               RAG
----------------------------------------------------------------------
Semantic Similarity (%)                    52.68             43.85
Keyword Match (%)                          61.04             40.81
Avg Length Difference                      4.874             2.321
LLM Judge (/5)                              3.32              3.40
======================================================================

🏆 Better Approach : Full-Text
```

| Metric | Full-Text | RAG | Better |
|---|---|---|---|
| Semantic Similarity (%) | 52.68 | 43.85 | Full-Text |
| Keyword Match (%) | 61.04 | 40.81 | Full-Text |
| Avg Length Difference | 4.874 | **2.321** | RAG |
| LLM Judge (/5) | 3.32 | **3.40** | RAG |

Full-text wins the weighted total (≈131.4 vs ≈129.5) — a narrow margin.

---

## 6. Interpretation

**Full-text wins on this document, and that is the expected result.**

The PDF is only ~28k characters (~7k tokens), which fits *entirely* inside the LLM's
context window. Full-text therefore hands the model 100% of the information on every
question, while RAG deliberately supplies only the top-3 chunks (~240 words). On a
document this small, retrieval can only *lose* information relative to sending
everything — so full-text scores higher on similarity and keyword overlap.

Notably, RAG still wins on **conciseness** (length difference) and the **LLM judge**,
showing its answers are tighter and on-target despite seeing far less text.

### The trade-off the metrics don't show

- **Full-text** sends ~7,000 tokens *per question*. A 300-page corpus (~500k tokens)
   would exceed the context window entirely — full-text does not scale.
- **RAG** sends ~240 tokens per question **regardless of corpus size**, keeping cost and
   latency flat as the knowledge base grows.

**Conclusion:** on small documents, full-text is simpler and more accurate. RAG's
advantage appears once the corpus is too large to fit in the context window — which is
the situation RAG is designed for.

---

## 7. Files

| File | Purpose |
|---|---|
| `build_vector_db.py` | Word-based chunking + Chroma ingestion |
| `rag_vs_fulltext.py` | RAG vs. full-text comparison and evaluation |
| `sample_pdf.pdf` | Source document |
| `golden_test_set.csv` | Evaluation questions |
| `chroma_db/` | Generated vector store |
| `results.csv` | Per-question scores (generated) |
