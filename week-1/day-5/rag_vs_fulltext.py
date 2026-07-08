import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import chromadb
import pandas as pd

import re
import time

# =====================================================
# Configuration
# =====================================================

# Open-source LLM served locally by Ollama.
OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "llama3.2"

PDF_PATH = "sample_pdf.pdf"

PDF_COLLECTION = "pdf_collection"

CHROMA_DB = "chroma_db"

TEST_SET = "golden_test_set.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3

# =====================================================
# Load Embedding Model
# =====================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")

# =====================================================
# Open Chroma Database
# =====================================================

print("\nOpening Chroma database...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB
)

collection = chroma_client.get_collection(
    PDF_COLLECTION
)

print("Connected successfully.")

# =====================================================
# Load Full Document Text (for the full-text baseline)
# =====================================================

def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


FULL_DOCUMENT = load_pdf(PDF_PATH)

print(f"\nLoaded full document ({len(FULL_DOCUMENT)} characters).")

# =====================================================
# Load Test Set
# =====================================================

df = pd.read_csv(TEST_SET)

print(f"\nLoaded {len(df)} questions.")

# =====================================================
# Ollama Call + Retry Wrapper
# =====================================================

def ask_ollama(messages):

    while True:

        try:

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            return data["message"]["content"].strip()

        except Exception as e:

            print("\nOllama Error:")
            print(e)

            print("\nRetrying in 10 seconds...")

            time.sleep(10)

# =====================================================
# Retrieve Relevant Chunks
# =====================================================

def retrieve(question):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=TOP_K

    )

    return results["documents"][0]

# =====================================================
# Build Context
# =====================================================

def build_context(question):

    documents = retrieve(question)

    context = "\n\n".join(documents)

    return context


# =====================================================
# RAG (retrieve top-K chunks)
# =====================================================

def ask_rag(question):

    context = build_context(question)

    prompt = f"""
You are answering questions using ONLY the provided context.

If the answer is not present in the context, reply exactly:

Not found in context.

Context:

{context}

Question:

{question}

Answer briefly in one or two sentences.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return ask_ollama(messages)


# =====================================================
# Full-Text Baseline (entire document, no retrieval)
# =====================================================

def ask_fulltext(question):

    prompt = f"""
You are answering questions using ONLY the provided document.

If the answer is not present in the document, reply exactly:

Not found in context.

Document:

{FULL_DOCUMENT}

Question:

{question}

Answer briefly in one or two sentences.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return ask_ollama(messages)


# =====================================================
# Test
# =====================================================

question = df.iloc[0]["question"]

print("\nQuestion:")
print(question)

print("\nFull-Text Answer:\n")

print(
    ask_fulltext(question)
)

print("\nRAG Answer:\n")

print(
    ask_rag(question)
)

# =====================================================
# Evaluation Metrics
# =====================================================

def semantic_similarity(expected, actual):

    # Embed both answers and compare their meaning with cosine similarity,
    # instead of requiring an exact string match.
    embeddings = embedding_model.encode(
        [expected, actual]
    )

    score = util.cos_sim(
        embeddings[0],
        embeddings[1]
    ).item()

    # Cosine similarity is in [-1, 1]; clamp negatives to 0.
    return max(0.0, score)


def keyword_match(expected, actual):

    expected_words = set(
        re.findall(r"\w+", expected.lower())
    )

    actual_words = set(
        re.findall(r"\w+", actual.lower())
    )

    if len(expected_words) == 0:
        return 0

    overlap = expected_words.intersection(actual_words)

    return len(overlap) / len(expected_words)


def length_difference(expected, actual):

    expected_len = len(expected.split())
    actual_len = len(actual.split())

    if expected_len == 0:
        return 0

    return abs(actual_len - expected_len) / expected_len

# =====================================================
# LLM Judge
# =====================================================

def llm_judge(question, expected, answer):

    prompt = f"""
You are evaluating answers.

Question:
{question}

Expected Answer:
{expected}

Candidate Answer:
{answer}

Give a score from 1 to 5.

5 = Perfect
4 = Mostly Correct
3 = Partially Correct
2 = Mostly Incorrect
1 = Incorrect

Return ONLY the number.
"""

    result = ask_ollama([
        {
            "role": "user",
            "content": prompt
        }
    ])

    numbers = re.findall(r"\d+", result)

    if numbers:
        score = int(numbers[0])
        return max(1, min(5, score))

    return 0

# =====================================================
# Evaluate Full-Text vs RAG
# =====================================================

results = []

for index, row in df.iterrows():

    question = row["question"]
    expected = row["expected_answer"]

    print(f"\nQuestion {index+1}/{len(df)}")
    print(question)

    fulltext_answer = ask_fulltext(question)

    rag_answer = ask_rag(question)

    fulltext_similarity = semantic_similarity(expected, fulltext_answer)
    rag_similarity = semantic_similarity(expected, rag_answer)

    fulltext_keyword = keyword_match(expected, fulltext_answer)
    rag_keyword = keyword_match(expected, rag_answer)

    fulltext_length = length_difference(expected, fulltext_answer)
    rag_length = length_difference(expected, rag_answer)

    fulltext_score = llm_judge(
        question,
        expected,
        fulltext_answer
    )

    rag_score = llm_judge(
        question,
        expected,
        rag_answer
    )

    results.append({

        "question": question,

        "expected_answer": expected,

        "fulltext_answer": fulltext_answer,

        "rag_answer": rag_answer,

        "fulltext_similarity": round(fulltext_similarity, 3),

        "rag_similarity": round(rag_similarity, 3),

        "fulltext_keyword_match": round(fulltext_keyword, 3),

        "rag_keyword_match": round(rag_keyword, 3),

        "fulltext_length_difference": round(fulltext_length, 3),

        "rag_length_difference": round(rag_length, 3),

        "fulltext_judge": fulltext_score,

        "rag_judge": rag_score

    })

    print("\nExpected:")
    print(expected)

    print("\nFull-Text:")
    print(fulltext_answer)

    print("\nRAG:")
    print(rag_answer)

    print("-" * 80)

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results.csv",
    index=False
)

print("\nResults saved to results.csv")

# =====================================================
# Summary Statistics
# =====================================================

fulltext_similarity = results_df["fulltext_similarity"].mean() * 100
rag_similarity = results_df["rag_similarity"].mean() * 100

fulltext_keyword = results_df["fulltext_keyword_match"].mean() * 100
rag_keyword = results_df["rag_keyword_match"].mean() * 100

fulltext_length = results_df["fulltext_length_difference"].mean()
rag_length = results_df["rag_length_difference"].mean()

fulltext_judge = results_df["fulltext_judge"].mean()
rag_judge = results_df["rag_judge"].mean()

# =====================================================
# Final Results
# =====================================================

print("\n")
print("=" * 70)
print("RAG vs FULL-TEXT EVALUATION")
print("=" * 70)

print(f"{'Metric':30}{'Full-Text':>18}{'RAG':>18}")
print("-" * 70)

print(
    f"{'Semantic Similarity (%)':30}"
    f"{fulltext_similarity:>18.2f}"
    f"{rag_similarity:>18.2f}"
)

print(
    f"{'Keyword Match (%)':30}"
    f"{fulltext_keyword:>18.2f}"
    f"{rag_keyword:>18.2f}"
)

print(
    f"{'Avg Length Difference':30}"
    f"{fulltext_length:>18.3f}"
    f"{rag_length:>18.3f}"
)

print(
    f"{'LLM Judge (/5)':30}"
    f"{fulltext_judge:>18.2f}"
    f"{rag_judge:>18.2f}"
)

print("=" * 70)

# =====================================================
# Overall Winner
# =====================================================

fulltext_total = (
    fulltext_similarity
    + fulltext_keyword
    + (fulltext_judge * 20)
    - (fulltext_length * 10)
)

rag_total = (
    rag_similarity
    + rag_keyword
    + (rag_judge * 20)
    - (rag_length * 10)
)

print()

if rag_total > fulltext_total:

    print("🏆 Better Approach : RAG")

elif fulltext_total > rag_total:

    print("🏆 Better Approach : Full-Text")

else:

    print("🏆 Both approaches performed equally.")


print("\nEvaluation Complete!")

print("Results saved to results.csv")
