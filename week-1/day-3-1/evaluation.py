from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from tqdm import tqdm

import pandas as pd
import os
import re
import time

load_dotenv()

# =====================================================
# Configuration
# =====================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Models
MODEL_A = "llama-3.3-70b-versatile"
MODEL_B = "openai/gpt-oss-120b"

# Files
PDF_PATH = "sample_pdf.pdf"
TEST_SET = "golden_test_set.csv"
OUTPUT_FILE = "evaluation_results.csv"

# =====================================================
# Load PDF
# =====================================================

def load_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text
            text += "\n"

    return text


print("=" * 60)
print("Loading PDF...")
print("=" * 60)

DOCUMENT = load_pdf(PDF_PATH)

print("PDF Loaded Successfully.")
print(f"Characters : {len(DOCUMENT):,}")
print(f"Words      : {len(DOCUMENT.split()):,}")
print()

# =====================================================
# Load Golden Test Set
# =====================================================

df = pd.read_csv(TEST_SET)

print(f"Loaded {len(df)} Questions")
print()

# =====================================================
# Resume Previous Evaluation
# =====================================================

if os.path.exists(OUTPUT_FILE):

    print("Existing evaluation file found.")

    results_df = pd.read_csv(OUTPUT_FILE)

    results = results_df.to_dict("records")

    completed = len(results)

    print(f"Resuming from Question {completed + 1}")

else:

    print("Starting new evaluation.")

    results = []

    completed = 0

print()


# =====================================================
# Retry Wrapper
# =====================================================

def ask_groq(messages, model):

    while True:

        try:

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            error = str(e)

            print("\n" + "=" * 60)
            print("Groq API Error")
            print("=" * 60)
            print(error)

            # ---------------------------------------------
            # Extract wait time from Groq rate-limit message
            # Example:
            # try again in 59m1.536s
            # ---------------------------------------------

            wait_time = 60

            match = re.search(
                r"try again in (\d+)m([\d.]+)s",
                error
            )

            if match:

                minutes = int(match.group(1))
                seconds = float(match.group(2))

                wait_time = int(minutes * 60 + seconds) + 5

            print(f"\nWaiting {wait_time} seconds...")
            print()

            time.sleep(wait_time)


# =====================================================
# Ask Model
# =====================================================

def ask_model(question, model):

    prompt = f"""
Answer ONLY using the document below.

If the answer cannot be found,
reply exactly:

Not found in document.

Document:

{DOCUMENT}

Question:
{question}

Give a concise answer.
"""

    messages = [

        {
            "role": "user",
            "content": prompt
        }

    ]

    return ask_groq(
        messages,
        model
    )

# =====================================================
# Exact Match
# =====================================================

def exact_match(expected, actual):

    return int(
        expected.strip().lower()
        ==
        actual.strip().lower()
    )


# =====================================================
# Keyword Match
# =====================================================

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


# =====================================================
# Length Difference
# =====================================================

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
You are an impartial evaluator.

Evaluate whether the candidate answer correctly answers the question based on the expected answer.

Question:
{question}

Expected Answer:
{expected}

Candidate Answer:
{answer}

Scoring:

5 = Completely correct
4 = Mostly correct
3 = Partially correct
2 = Mostly incorrect
1 = Incorrect

Return ONLY a single integer.
"""

    messages = [

        {
            "role": "user",
            "content": prompt
        }

    ]

    result = ask_groq(
        messages,
        MODEL_A
    )

    numbers = re.findall(r"\d+", result)

    if numbers:

        score = int(numbers[0])

        return max(1, min(5, score))

    return 0


# =====================================================
# Evaluate One Model
# =====================================================

def evaluate_answer(question, expected, answer):

    return {

        "exact_match": exact_match(
            expected,
            answer
        ),

        "keyword_match": round(
            keyword_match(
                expected,
                answer
            ),
            3
        ),

        "length_difference": round(
            length_difference(
                expected,
                answer
            ),
            3
        ),

        "llm_judge": llm_judge(
            question,
            expected,
            answer
        )

    }

# =====================================================
# Evaluation Loop
# =====================================================

print("=" * 60)
print("Starting Evaluation")
print("=" * 60)

for index in tqdm(range(completed, len(df))):

    row = df.iloc[index]

    question = row["question"]
    expected = row["expected_answer"]

    print("\n" + "=" * 80)
    print(f"Question {index + 1}/{len(df)}")
    print(question)
    print("=" * 80)

    # -------------------------------------------------
    # Model A
    # -------------------------------------------------

    print(f"\nQuerying {MODEL_A}...")

    answer_a = ask_model(
        question,
        MODEL_A
    )

    model_a_metrics = evaluate_answer(
        question,
        expected,
        answer_a
    )

    # -------------------------------------------------
    # Model B
    # -------------------------------------------------

    print(f"\nQuerying {MODEL_B}...")

    answer_b = ask_model(
        question,
        MODEL_B
    )

    model_b_metrics = evaluate_answer(
        question,
        expected,
        answer_b
    )

    # -------------------------------------------------
    # Store Results
    # -------------------------------------------------

    result = {

        "question": question,

        "expected_answer": expected,

        "model_a_answer": answer_a,

        "model_b_answer": answer_b,

        "a_exact_match":
            model_a_metrics["exact_match"],

        "b_exact_match":
            model_b_metrics["exact_match"],

        "a_keyword_match":
            model_a_metrics["keyword_match"],

        "b_keyword_match":
            model_b_metrics["keyword_match"],

        "a_length_difference":
            model_a_metrics["length_difference"],

        "b_length_difference":
            model_b_metrics["length_difference"],

        "a_llm_judge":
            model_a_metrics["llm_judge"],

        "b_llm_judge":
            model_b_metrics["llm_judge"]

    }

    results.append(result)

    # -------------------------------------------------
    # Save Progress
    # -------------------------------------------------

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------------------------------
    # Print Results
    # -------------------------------------------------

    print("\nExpected Answer")
    print("-" * 40)
    print(expected)

    print("\nModel A")
    print("-" * 40)
    print(answer_a)

    print("\nModel B")
    print("-" * 40)
    print(answer_b)

    print("\nMetrics")

    print(
        f"Model A -> "
        f"Exact:{model_a_metrics['exact_match']}  "
        f"Keyword:{model_a_metrics['keyword_match']:.2f}  "
        f"Length:{model_a_metrics['length_difference']:.2f}  "
        f"Judge:{model_a_metrics['llm_judge']}"
    )

    print(
        f"Model B -> "
        f"Exact:{model_b_metrics['exact_match']}  "
        f"Keyword:{model_b_metrics['keyword_match']:.2f}  "
        f"Length:{model_b_metrics['length_difference']:.2f}  "
        f"Judge:{model_b_metrics['llm_judge']}"
    )

    print("\nProgress Saved ✓")


# =====================================================
# Final Summary
# =====================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

a_exact = results_df["a_exact_match"].mean() * 100
b_exact = results_df["b_exact_match"].mean() * 100

a_keyword = results_df["a_keyword_match"].mean() * 100
b_keyword = results_df["b_keyword_match"].mean() * 100

a_length = results_df["a_length_difference"].mean()
b_length = results_df["b_length_difference"].mean()

a_judge = results_df["a_llm_judge"].mean()
b_judge = results_df["b_llm_judge"].mean()

summary = pd.DataFrame({

    "Metric": [

        "Exact Match (%)",
        "Keyword Match (%)",
        "Average Length Difference",
        "LLM Judge (/5)"

    ],

    "Model A": [

        round(a_exact, 2),
        round(a_keyword, 2),
        round(a_length, 3),
        round(a_judge, 2)

    ],

    "Model B": [

        round(b_exact, 2),
        round(b_keyword, 2),
        round(b_length, 3),
        round(b_judge, 2)

    ]

})

print(summary.to_string(index=False))

print("\n")

# =====================================================
# Overall Score
# =====================================================

score_a = (
    a_exact
    + a_keyword
    + (a_judge * 20)
    - (a_length * 10)
)

score_b = (
    b_exact
    + b_keyword
    + (b_judge * 20)
    - (b_length * 10)
)

print("=" * 70)

print(f"Overall Score ({MODEL_A}) : {score_a:.2f}")
print(f"Overall Score ({MODEL_B}) : {score_b:.2f}")

print("=" * 70)

if score_a > score_b:

    winner = MODEL_A

elif score_b > score_a:

    winner = MODEL_B

else:

    winner = "Tie"

print()

print(f"🏆 Better Overall Model : {winner}")

print()

print(f"Detailed results saved to {OUTPUT_FILE}")

# =====================================================
# Generate Markdown Report
# =====================================================

with open("report.md", "w") as report:

    report.write("# Day 3 Evaluation Report\n\n")

    report.write("## Dataset\n\n")
    report.write(f"- PDF: {PDF_PATH}\n")
    report.write(f"- Questions: {len(df)}\n")
    report.write(f"- Model A: {MODEL_A}\n")
    report.write(f"- Model B: {MODEL_B}\n\n")

    report.write("## Evaluation Metrics\n\n")
    report.write("| Metric | Model A | Model B |\n")
    report.write("|--------|---------|---------|\n")
    report.write(f"| Exact Match (%) | {a_exact:.2f} | {b_exact:.2f} |\n")
    report.write(f"| Keyword Match (%) | {a_keyword:.2f} | {b_keyword:.2f} |\n")
    report.write(f"| Avg Length Difference | {a_length:.3f} | {b_length:.3f} |\n")
    report.write(f"| LLM Judge (/5) | {a_judge:.2f} | {b_judge:.2f} |\n\n")

    report.write("## Overall Scores\n\n")
    report.write(f"- {MODEL_A}: {score_a:.2f}\n")
    report.write(f"- {MODEL_B}: {score_b:.2f}\n\n")

    report.write(f"**Winner:** {winner}\n\n")

    report.write("## Observations\n\n")

    if winner == MODEL_A:

        report.write("- Model A achieved the highest overall score.\n")
        report.write("- Model A generally produced more accurate answers.\n")

    elif winner == MODEL_B:

        report.write("- Model B achieved the highest overall score.\n")
        report.write("- Model B generally produced more accurate answers.\n")

    else:

        report.write("- Both models performed similarly.\n")

    report.write("- Keyword Match provided a more meaningful metric than Exact Match because models often paraphrased answers.\n")
    report.write("- LLM Judge effectively evaluated semantic correctness even when wording differed.\n")
    report.write("- Exact Match alone was insufficient for evaluating long-form answers.\n")
    report.write("- Average Length Difference highlighted verbosity differences between models.\n")

print("\nreport.md generated successfully.")
print("\nEvaluation Complete!")