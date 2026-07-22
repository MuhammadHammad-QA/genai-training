from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import pandas as pd

import re

# =====================================================
# Configuration
# =====================================================

PDF_PATH = "sample_pdf.pdf"

CHROMA_DB_PATH = "chroma_db"

TEST_SET = "golden_test_set.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval depth. hit-rate@5 is the headline metric; @1 and @3 are also
# reported to show the recall/precision trade-off across strategies.
TOP_K = 5

# Chunk sizes are held constant across all three strategies so the chunking
# STRATEGY is the only variable being compared. 400 characters is ~70-90 words
# (~100-130 tokens) which stays safely under the 256-token limit of
# all-MiniLM-L6-v2 — a chunk larger than that would be silently truncated at
# embedding time (the lesson learned on day 5).
CHUNK_SIZE = 400

CHUNK_OVERLAP = 80

# A retrieved chunk counts as relevant if EITHER signal fires:
#   - its cosine similarity to the expected answer >= SIM_THRESHOLD, or
#   - it contains at least KEYWORD_THRESHOLD of the answer's significant keywords.
# Two signals are used because the expected answers are paraphrases: semantic
# similarity catches reworded facts, keyword overlap catches exact-term facts
# (names, IDs, acronyms) that a short answer may score low on semantically.
SIM_THRESHOLD = 0.5

KEYWORD_THRESHOLD = 0.5


# =====================================================
# Load PDF
# =====================================================

def load_pdf(path):

    print("Loading PDF...")

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    print("PDF Loaded Successfully.")
    print(f"Characters: {len(text)}")

    return text


# =====================================================
# Chunking Strategy 1 — Fixed-Size (no overlap)
# =====================================================

def chunk_fixed(text, chunk_size=CHUNK_SIZE):

    # Contiguous, non-overlapping windows. Simplest possible strategy: the
    # document is cut every `chunk_size` characters regardless of content, so a
    # sentence or fact can be split across a boundary and lost to retrieval.
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size

    return chunks


# =====================================================
# Chunking Strategy 2 — Overlapping (sliding window)
# =====================================================

def chunk_overlapping(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    # Same fixed window, but each chunk repeats the last `overlap` characters of
    # the previous one. This keeps facts that straddle a boundary intact in at
    # least one chunk, at the cost of storing more (redundant) text.
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# =====================================================
# Chunking Strategy 3 — Recursive / Hierarchical
# =====================================================

def chunk_recursive(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    # Splits on a hierarchy of natural boundaries — paragraphs, then lines, then
    # sentences, then words — falling back to the next separator only when a
    # piece is still too large. This tries to keep semantically whole units
    # (whole sentences / paragraphs) inside a single chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    return splitter.split_text(text)


STRATEGIES = {
    "fixed": chunk_fixed,
    "overlapping": chunk_overlapping,
    "recursive": chunk_recursive,
}


# =====================================================
# Load Embedding Model
# =====================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# =====================================================
# Read PDF
# =====================================================

document = load_pdf(PDF_PATH)


# =====================================================
# Build One Chroma Collection Per Strategy
# =====================================================

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collections = {}


def build_collection(name, chunks):

    print(f"\nIndexing '{name}' ({len(chunks)} chunks)...")

    # all-MiniLM-L6-v2 is trained for cosine similarity, so vectors are
    # normalized and the collection is told to rank under cosine distance
    # (Chroma defaults to L2) — same fixes applied on day 5.
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    try:
        client.delete_collection(name)
    except:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings]
    )

    return collection


chunk_stats = {}

for name, chunker in STRATEGIES.items():

    chunks = chunker(document)

    chunk_stats[name] = {
        "num_chunks": len(chunks),
        "avg_chars": round(
            sum(len(c) for c in chunks) / len(chunks), 1
        ),
    }

    collections[name] = build_collection(name, chunks)


# =====================================================
# Load Test Set
# =====================================================

df = pd.read_csv(TEST_SET)

print(f"\nLoaded {len(df)} questions.")


# =====================================================
# Relevance Judgment (keyword-based)
# =====================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "by", "with", "as", "at", "it", "its",
    "that", "this", "these", "those", "such", "which", "from", "into",
    "can", "used", "using", "use", "different", "across", "within",
}


def answer_keywords(answer):

    # Significant content words from the expected answer: lower-cased, no
    # stopwords, at least 3 characters.
    words = re.findall(r"\w+", answer.lower())

    return set(
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    )


def keyword_overlap(chunk, keywords):

    if not keywords:
        return 0.0

    chunk_words = set(
        re.findall(r"\w+", chunk.lower())
    )

    overlap = keywords & chunk_words

    return len(overlap) / len(keywords)


def is_relevant(chunk, keywords, answer_embedding):

    # Semantic signal: cosine similarity between the expected answer and the
    # chunk (embeddings are normalized, so this is a dot product).
    chunk_embedding = embedding_model.encode(
        chunk,
        normalize_embeddings=True
    )

    similarity = util.cos_sim(
        answer_embedding,
        chunk_embedding
    ).item()

    if similarity >= SIM_THRESHOLD:
        return True

    # Keyword signal: exact-term overlap, for facts a short answer scores low
    # on semantically (acronyms, ISO numbers, proper nouns).
    return keyword_overlap(chunk, keywords) >= KEYWORD_THRESHOLD


# =====================================================
# Hit-Rate@K Evaluation
# =====================================================

def hits_at_k(retrieved, keywords, answer_embedding):

    # Judge each retrieved chunk once, in rank order, then read off @1/@3/@5.
    relevance = [
        is_relevant(chunk, keywords, answer_embedding)
        for chunk in retrieved
    ]

    result = {}

    for k in (1, 3, 5):

        # Hit@K = 1 if ANY of the first K retrieved chunks is relevant.
        result[k] = int(any(relevance[:k]))

    return result


def retrieve(collection, question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# =====================================================
# Evaluate Every Strategy Over the Test Set
# =====================================================

rows = []

for index, row in df.iterrows():

    question = row["question"]
    expected = row["expected_answer"]

    keywords = answer_keywords(expected)

    answer_embedding = embedding_model.encode(
        expected,
        normalize_embeddings=True
    )

    print(f"\nQuestion {index+1}/{len(df)}: {question}")

    record = {
        "question": question,
        "expected_answer": expected,
    }

    for name in STRATEGIES:

        retrieved = retrieve(collections[name], question)

        hits = hits_at_k(retrieved, keywords, answer_embedding)

        for k in (1, 3, 5):
            record[f"{name}_hit@{k}"] = hits[k]

        print(
            f"  {name:12} "
            f"hit@1={hits[1]}  hit@3={hits[3]}  hit@5={hits[5]}"
        )

    rows.append(record)


# =====================================================
# Save Per-Question Results
# =====================================================

results_df = pd.DataFrame(rows)

results_df.to_csv(
    "results.csv",
    index=False
)

print("\nPer-question results saved to results.csv")


# =====================================================
# Summary — Hit-Rate@K Per Strategy
# =====================================================

print("\n")
print("=" * 70)
print("CHUNKING STRATEGY COMPARISON — HIT-RATE@K")
print("=" * 70)

print(
    f"{'Strategy':14}{'Chunks':>8}{'AvgChars':>10}"
    f"{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}"
)
print("-" * 70)

summary = {}

for name in STRATEGIES:

    hit1 = results_df[f"{name}_hit@1"].mean() * 100
    hit3 = results_df[f"{name}_hit@3"].mean() * 100
    hit5 = results_df[f"{name}_hit@5"].mean() * 100

    summary[name] = hit5

    print(
        f"{name:14}"
        f"{chunk_stats[name]['num_chunks']:>8}"
        f"{chunk_stats[name]['avg_chars']:>10}"
        f"{hit1:>8.1f}%"
        f"{hit3:>8.1f}%"
        f"{hit5:>8.1f}%"
    )

print("=" * 70)


# =====================================================
# Winner (by hit-rate@5)
# =====================================================

best = max(summary, key=summary.get)

print()
print(f"🏆 Best hit-rate@5 : {best} ({summary[best]:.1f}%)")

print("\nEvaluation Complete!")
print("Results saved to results.csv")
