from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd

import os
import re
import time

load_dotenv()

# =====================================================
# Configuration
# =====================================================

MODEL = "llama-3.3-70b-versatile"

PDF_COLLECTION = "pdf_collection"

CHROMA_DB = "chroma_db"

TEST_SET = "golden_test_set.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

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
# Load Test Set
# =====================================================

df = pd.read_csv(TEST_SET)

print(f"\nLoaded {len(df)} questions.")

# =====================================================
# Retry Wrapper
# =====================================================

def ask_groq(messages):

    while True:

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=messages,

                temperature=0

            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            print("\nGroq Error:")
            print(e)

            print("\nRetrying in 30 seconds...")

            time.sleep(30)

# =====================================================
# Retrieve Relevant Chunks
# =====================================================

def retrieve(question):

    query_embedding = embedding_model.encode(
        question
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
# LLM Only
# =====================================================

def ask_llm(question):

    messages = [
        {
            "role": "user",
            "content": f"""
Answer the following question briefly.

Question:
{question}
"""
        }
    ]

    return ask_groq(messages)


# =====================================================
# RAG
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

    return ask_groq(messages)


# =====================================================
# Test
# =====================================================

question = df.iloc[0]["question"]

print("\nQuestion:")
print(question)

print("\nLLM Only Answer:\n")

print(
    ask_llm(question)
)

print("\nRAG Answer:\n")

print(
    ask_rag(question)
)

# =====================================================
# Evaluation Metrics
# =====================================================

def exact_match(expected, actual):

    return int(
        expected.lower().strip()
        ==
        actual.lower().strip()
    )


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

    result = ask_groq([
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
# Evaluate LLM vs RAG
# =====================================================

results = []

for index, row in df.iterrows():

    question = row["question"]
    expected = row["expected_answer"]

    print(f"\nQuestion {index+1}/{len(df)}")
    print(question)

    llm_answer = ask_llm(question)

    rag_answer = ask_rag(question)

    llm_exact = exact_match(expected, llm_answer)
    rag_exact = exact_match(expected, rag_answer)

    llm_keyword = keyword_match(expected, llm_answer)
    rag_keyword = keyword_match(expected, rag_answer)

    llm_length = length_difference(expected, llm_answer)
    rag_length = length_difference(expected, rag_answer)

    llm_score = llm_judge(
        question,
        expected,
        llm_answer
    )

    rag_score = llm_judge(
        question,
        expected,
        rag_answer
    )

    results.append({

        "question": question,

        "expected_answer": expected,

        "llm_answer": llm_answer,

        "rag_answer": rag_answer,

        "llm_exact_match": llm_exact,

        "rag_exact_match": rag_exact,

        "llm_keyword_match": round(llm_keyword, 3),

        "rag_keyword_match": round(rag_keyword, 3),

        "llm_length_difference": round(llm_length, 3),

        "rag_length_difference": round(rag_length, 3),

        "llm_judge": llm_score,

        "rag_judge": rag_score

    })

    print("\nExpected:")
    print(expected)

    print("\nLLM:")
    print(llm_answer)

    print("\nRAG:")
    print(rag_answer)

    print("-" * 80)

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "rag_results.csv",
    index=False
)

print("\nResults saved to rag_results.csv")

# =====================================================
# Summary Statistics
# =====================================================

llm_exact = results_df["llm_exact_match"].mean() * 100
rag_exact = results_df["rag_exact_match"].mean() * 100

llm_keyword = results_df["llm_keyword_match"].mean() * 100
rag_keyword = results_df["rag_keyword_match"].mean() * 100

llm_length = results_df["llm_length_difference"].mean()
rag_length = results_df["rag_length_difference"].mean()

llm_judge = results_df["llm_judge"].mean()
rag_judge = results_df["rag_judge"].mean()

# =====================================================
# Final Results
# =====================================================

print("\n")
print("=" * 70)
print("FINAL RAG EVALUATION")
print("=" * 70)

print(f"{'Metric':30}{'LLM Only':>18}{'RAG':>18}")
print("-" * 70)

print(
    f"{'Exact Match (%)':30}"
    f"{llm_exact:>18.2f}"
    f"{rag_exact:>18.2f}"
)

print(
    f"{'Keyword Match (%)':30}"
    f"{llm_keyword:>18.2f}"
    f"{rag_keyword:>18.2f}"
)

print(
    f"{'Avg Length Difference':30}"
    f"{llm_length:>18.3f}"
    f"{rag_length:>18.3f}"
)

print(
    f"{'LLM Judge (/5)':30}"
    f"{llm_judge:>18.2f}"
    f"{rag_judge:>18.2f}"
)

print("=" * 70)

# =====================================================
# Overall Winner
# =====================================================

llm_score = (
    llm_exact
    + llm_keyword
    + (llm_judge * 20)
    - (llm_length * 10)
)

rag_score = (
    rag_exact
    + rag_keyword
    + (rag_judge * 20)
    - (rag_length * 10)
)

print()

if rag_score > llm_score:

    print("🏆 Better Approach : RAG")

elif llm_score > rag_score:

    print("🏆 Better Approach : LLM Only")

else:

    print("🏆 Both approaches performed equally.")


print("\nEvaluation Complete!")

print("Results saved to rag_results.csv")