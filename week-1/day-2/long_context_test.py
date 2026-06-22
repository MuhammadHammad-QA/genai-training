from groq import Groq
from pypdf import PdfReader
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# Configuration
# =====================================================

PDF_FILE = "leave_policy.pdf"

QUESTIONS = [
    "What is the purpose of this leave policy?",
    "Who is eligible for leave under this policy?",
    "How many annual leave days are employees entitled to?",
    "What types of leave are available?",
    "What is the process for requesting leave?",
    "How far in advance should leave be requested?",
    "Can unused leave be carried forward?",
    "What happens if an employee takes leave without approval?",
    "Are there any exceptions or special cases mentioned in the policy?",
    "Summarize the leave policy in 5 bullet points."
]

# =====================================================
# Initialize Clients
# =====================================================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

encoder = tiktoken.get_encoding("cl100k_base")

# =====================================================
# Extract PDF Text
# =====================================================

print("Reading PDF...")

reader = PdfReader(PDF_FILE)

document_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        document_text += page_text + "\n"

document_tokens = len(
    encoder.encode(document_text)
)

print(f"Document Tokens: {document_tokens}")

# =====================================================
# Create Summary
# =====================================================

print("Generating document summary...")

summary_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"""
You are analyzing a company leave policy.

Create a detailed summary covering:

1. Purpose of the policy
2. Employee eligibility
3. Leave types
4. Leave entitlement
5. Approval process
6. Carry-forward rules
7. Special conditions
8. Restrictions and exceptions

Document:

{document_text}
"""
        }
    ],
    temperature=0
)

summary = summary_response.choices[0].message.content

summary_tokens = len(
    encoder.encode(summary)
)

print(f"Summary Tokens: {summary_tokens}")

reduction = (
    (document_tokens - summary_tokens)
    / document_tokens
) * 100

print(
    f"Token Reduction: {reduction:.2f}%"
)

# Save summary
with open("document_summary.txt", "w") as f:
    f.write(summary)

# =====================================================
# Compare Both Approaches
# =====================================================

results = []

for i, question in enumerate(QUESTIONS, start=1):

    print(
        f"Processing Question {i}/{len(QUESTIONS)}"
    )

    # -------------------------------------------------
    # Naive Stuffing
    # -------------------------------------------------

    naive_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
You are answering questions about a leave policy.

Document:

{document_text}

Question:
{question}
"""
            }
        ],
        temperature=0
    )

    naive_answer = (
        naive_response
        .choices[0]
        .message
        .content
    )

    # -------------------------------------------------
    # Summarize-Then-Answer
    # -------------------------------------------------

    summary_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""
You are answering questions using a summarized leave policy.

Summary:

{summary}

Question:
{question}
"""
            }
        ],
        temperature=0
    )

    summary_answer = (
        summary_response
        .choices[0]
        .message
        .content
    )

    results.append(
        {
            "question": question,
            "naive": naive_answer,
            "summary": summary_answer
        }
    )

# =====================================================
# Save Comparison Results
# =====================================================

with open(
    "comparison_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("LEAVE POLICY COMPARISON\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        f"Document Tokens: {document_tokens}\n"
    )
    f.write(
        f"Summary Tokens: {summary_tokens}\n"
    )
    f.write(
        f"Token Reduction: {reduction:.2f}%\n\n"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        f.write("=" * 80 + "\n")
        f.write(f"QUESTION {i}\n")
        f.write("=" * 80 + "\n\n")

        f.write("QUESTION:\n")
        f.write(result["question"] + "\n\n")

        f.write("NAIVE STUFFING ANSWER:\n")
        f.write(result["naive"] + "\n\n")

        f.write(
            "SUMMARIZE-THEN-ANSWER:\n"
        )
        f.write(
            result["summary"] + "\n\n"
        )

print("\nDone!")
print(
    "Generated files:"
)
print(
    " - document_summary.txt"
)
print(
    " - comparison_results.txt"
)