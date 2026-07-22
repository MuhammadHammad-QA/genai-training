# Day 6 — Chunking Strategies

**Topic:** Fixed-size vs. overlapping vs. recursive/hierarchical chunking and their
impact on recall and precision.
**Task:** Index the same corpus with three different chunking strategies and compare
their **hit-rate@5** retrieval performance.

---

## 1. Setup

| Component | Choice |
|---|---|
| Corpus | `sample_pdf.pdf` (~28,393 characters, a document about the PDF format) |
| Vector DB | Chroma (persistent, `chroma_db/`, one collection per strategy) |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers), cosine space, normalized |
| Test set | `golden_test_set.csv` (25 Q&A pairs) |
| Retrieval depth | `TOP_K = 5` (hit@1 / hit@3 / hit@5 reported) |

One script, `chunking_comparison.py`, does everything: load the corpus → build three
independent Chroma collections (one per chunking strategy) → run every golden question
against all three → score hit-rate@K.

---

## 2. The Three Chunking Strategies

**Chunk size is held constant (400 characters) across all three strategies**, so the
*only* variable being compared is the chunking **strategy** itself, not the size. 400
characters (~70–90 words, ~100–130 tokens) also stays safely under the **256-token
limit** of `all-MiniLM-L6-v2` — the day-5 lesson that an oversized chunk gets silently
truncated at embedding time.

| # | Strategy | How it splits | Trade-off |
|---|---|---|---|
| 1 | **Fixed-size** | Cuts the text every 400 chars, no overlap. | Simplest. A fact that straddles a boundary is split and can become unretrievable. |
| 2 | **Overlapping** | 400-char windows with an 80-char sliding overlap. | Boundary-straddling facts survive whole in at least one chunk, at the cost of redundant stored text. |
| 3 | **Recursive / hierarchical** | LangChain `RecursiveCharacterTextSplitter`, splitting on `\n\n` → `\n` → `. ` → ` ` → `""`. | Tries to keep whole sentences/paragraphs together, so chunks are semantically cleaner. |

Resulting index sizes:

| Strategy | Chunks | Avg chars/chunk |
|---|---|---|
| fixed | 71 | 399.9 |
| overlapping | 89 | 398.1 |
| recursive | 88 | 346.6 |

Overlapping produces more chunks (each window advances only 320 chars). Recursive
produces slightly shorter chunks because it breaks early at natural boundaries rather
than filling every window to 400.

---

## 3. How hit-rate@K is Measured

**hit-rate@K is a pure *retrieval* metric — no LLM generation is involved.** For each
question we retrieve the top-K chunks and ask: *did at least one relevant chunk make the
cut?*

`hit-rate@K = (questions with ≥1 relevant chunk in the top-K) / (total questions)`

Judging whether a retrieved chunk is "relevant" is the hard part, because the golden
answers are **paraphrases** — they rarely appear verbatim in the source text. A chunk is
counted as relevant if **either** signal fires:

- **Semantic** — cosine similarity between the expected answer and the chunk ≥ `0.5`.
   Catches reworded facts.
- **Keyword** — the chunk contains ≥ `0.5` of the answer's significant keywords
   (stopwords removed). Catches exact-term facts (acronyms, ISO numbers, proper nouns)
   that a short answer scores low on semantically.

Using keyword overlap alone (the first attempt) was too strict: many questions returned
0 hits across *all three* strategies, which hid the differences between them. Adding the
semantic signal made the metric reflect real retrieval quality.

---

## 4. Results

```sh
======================================================================
CHUNKING STRATEGY COMPARISON — HIT-RATE@K
======================================================================
Strategy        Chunks  AvgChars    Hit@1    Hit@3    Hit@5
----------------------------------------------------------------------
fixed               71     399.9    48.0%    52.0%    56.0%
overlapping         89     398.1    52.0%    64.0%    64.0%
recursive           88     346.6    36.0%    64.0%    64.0%
======================================================================

🏆 Best hit-rate@5 : overlapping (64.0%)
```

| Strategy | Hit@1 | Hit@3 | **Hit@5** |
|---|---|---|---|
| fixed | 48.0% | 52.0% | 56.0% |
| overlapping | **52.0%** | **64.0%** | **64.0%** |
| recursive | 36.0% | **64.0%** | **64.0%** |

Overlapping and recursive **tie at 64% hit@5**; the script breaks the tie in favour of
overlapping, which also leads at hit@1 and hit@3.

---

## 5. Interpretation

**Both boundary-aware strategies (overlapping, recursive) beat naïve fixed-size** — by
8 percentage points at hit@5 (64% vs 56%). This is the expected result and the whole
point of the exercise:

- **Fixed-size** cuts blindly every 400 characters, so a fact sitting on a boundary is
   split across two chunks and neither chunk embeds it cleanly — it drops out of
   retrieval. It trails at every cutoff.
- **Overlapping** repeats the last 80 characters of each chunk, so a boundary-straddling
   fact still appears whole somewhere. It recovers exactly those lost hits and leads
   overall.
- **Recursive** is the most interesting: it is the **weakest at hit@1 (36%)** but
   catches up completely by hit@3 (64%). Breaking on sentence/paragraph boundaries makes
   chunks cleaner, but the shorter chunks (avg 347 chars) split a fact's context across
   neighbours, so the single best chunk is less often a full hit — yet the right chunk is
   reliably somewhere in the top 3.

### Recall vs. precision (the day's topic)

hit@1 is a **precision-like** signal (is the *very top* result right?); hit@5 is a
**recall-like** signal (is a right answer *anywhere* in the returned set?).

- Overlapping wins on **both** — the best all-round choice here.
- Recursive shows the classic pattern: **lower precision, competitive recall.** If your
   pipeline feeds the top-1 chunk to the LLM, recursive would hurt; if it feeds the top-5,
   recursive is just as good as overlapping.
- Fixed-size is dominated on every axis — there is little reason to prefer it except
   implementation simplicity.

---

## 6. Caveats

- **Small corpus, small test set** (~28k chars, 25 questions). Each question is worth
   4 percentage points, so the overlapping-vs-recursive tie at hit@5 is within noise —
   the safe conclusion is "both boundary-aware strategies clearly beat fixed-size," not a
   precise ranking between the two.
- The relevance judgment is an automated proxy (semantic + keyword), not a human label,
   so absolute hit-rates should be read as *relative* comparisons between strategies
   under one consistent judge.

---

## 7. Files

| File | Purpose |
|---|---|
| `chunking_comparison.py` | Builds all three indexes and computes hit-rate@K |
| `sample_pdf.pdf` | Source corpus |
| `golden_test_set.csv` | Evaluation questions |
| `chroma_db/` | Generated vector store (one collection per strategy) |
| `results.csv` | Per-question hit@1/@3/@5 for all three strategies (generated) |

### Running it

```sh
cd week-1/day-6
../../venv/bin/python chunking_comparison.py
```

> The embedding model is loaded from the local HuggingFace cache. If the machine is
> offline, prefix the command with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` so
> SentenceTransformers doesn't try to reach the Hub.
