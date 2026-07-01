from groq import Groq
import pandas as pd
import pdfplumber
import os
import re
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# Configuration
# =====================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_A = "llama-3.3-70b-versatile"
MODEL_B = "openai/gpt-oss-120b"

# =====================================================
# PDF Extraction
# =====================================================

def extract_pdf_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


DOCUMENT_TEXT = extract_pdf_text(
    "leave_policy.pdf"
)

print(
    f"Document loaded: "
    f"{len(DOCUMENT_TEXT.split())} words"
)

# =====================================================
# Load Golden Test Set
# =====================================================

df = pd.read_csv(
    "golden_test_set.csv"
)

results = []

# =====================================================
# Helper Functions
# =====================================================

def ask_model(model, question):

    prompt = f"""
Answer ONLY using information from the document.

If the answer is not present in the document,
return:

NOT FOUND

Document:

{DOCUMENT_TEXT}

Question:

{question}

Return a short answer only.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


def exact_match(expected, actual):

    return int(
        expected.lower().strip()
        ==
        actual.lower().strip()
    )


def keyword_match(expected, actual):

    expected_words = set(
        re.findall(
            r"\w+",
            expected.lower()
        )
    )

    actual_words = set(
        re.findall(
            r"\w+",
            actual.lower()
        )
    )

    if len(expected_words) == 0:
        return 0

    overlap = expected_words.intersection(
        actual_words
    )

    return len(overlap) / len(expected_words)


def length_difference(expected, actual):

    expected_len = len(
        expected.split()
    )

    actual_len = len(
        actual.split()
    )

    if expected_len == 0:
        return 0

    return abs(
        actual_len - expected_len
    ) / expected_len


def llm_judge(question, expected, actual):

    prompt = f"""
You are evaluating a candidate answer.

Question:
{question}

Expected Answer:
{expected}

Candidate Answer:
{actual}

Score from 1 to 5.

5 = Fully correct
4 = Mostly correct
3 = Partially correct
2 = Mostly incorrect
1 = Incorrect

Return ONLY the number.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:

        score = int(
            re.findall(
                r"\d+",
                text
            )[0]
        )

        return max(
            1,
            min(5, score)
        )

    except:

        return 0


# =====================================================
# Evaluation Loop
# =====================================================

for index, row in df.iterrows():

    question = row["question"]
    expected = row["expected_answer"]

    print(
        f"Question {index + 1}/{len(df)}"
    )

    answer_a = ask_model(
        MODEL_A,
        question
    )

    answer_b = ask_model(
        MODEL_B,
        question
    )

    a_exact = exact_match(
        expected,
        answer_a
    )

    b_exact = exact_match(
        expected,
        answer_b
    )

    a_keyword = keyword_match(
        expected,
        answer_a
    )

    b_keyword = keyword_match(
        expected,
        answer_b
    )

    a_length = length_difference(
        expected,
        answer_a
    )

    b_length = length_difference(
        expected,
        answer_b
    )

    a_judge = llm_judge(
        question,
        expected,
        answer_a
    )

    b_judge = llm_judge(
        question,
        expected,
        answer_b
    )

    results.append({

        "question": question,
        "expected_answer": expected,

        "model_a_answer": answer_a,
        "model_b_answer": answer_b,

        "a_exact_match": a_exact,
        "b_exact_match": b_exact,

        "a_keyword_match": a_keyword,
        "b_keyword_match": b_keyword,

        "a_length_difference": a_length,
        "b_length_difference": b_length,

        "a_llm_judge": a_judge,
        "b_llm_judge": b_judge
    })

# =====================================================
# Save Results
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

# =====================================================
# Final Scores
# =====================================================

print("\n==============================")
print("FINAL RESULTS")
print("==============================\n")

print(
    f"Model A Exact Match: "
    f"{results_df['a_exact_match'].mean()*100:.2f}%"
)

print(
    f"Model B Exact Match: "
    f"{results_df['b_exact_match'].mean()*100:.2f}%"
)

print()

print(
    f"Model A Keyword Match: "
    f"{results_df['a_keyword_match'].mean()*100:.2f}%"
)

print(
    f"Model B Keyword Match: "
    f"{results_df['b_keyword_match'].mean()*100:.2f}%"
)

print()

print(
    f"Model A Avg Length Difference: "
    f"{results_df['a_length_difference'].mean()*100:.2f}%"
)

print(
    f"Model B Avg Length Difference: "
    f"{results_df['b_length_difference'].mean()*100:.2f}%"
)

print()

print(
    f"Model A LLM Judge Score: "
    f"{results_df['a_llm_judge'].mean():.2f}/5"
)

print(
    f"Model B LLM Judge Score: "
    f"{results_df['b_llm_judge'].mean():.2f}/5"
)

print(
    "\nDetailed results saved to evaluation_results.csv"
)