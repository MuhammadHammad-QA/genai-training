# Day 4 – Retrieval-Augmented Generation (RAG) using ChromaDB

## Objective

The objective of this task was to build a simple Retrieval-Augmented Generation (RAG) pipeline using ChromaDB and compare its performance against a Large Language Model (LLM) answering questions without retrieval.

---

# Dataset

**Source Document**

* `leave_policy.pdf`

**Question Set**

* `golden_test_set.csv`
* Total Questions: **25**

The evaluation used the same 25-question dataset for both the LLM-only approach and the RAG pipeline to ensure a fair comparison.

---

# Technologies Used

* Python
* ChromaDB
* Sentence Transformers
* Groq API
* Llama-3.3-70B-Versatile
* PyPDF
* Pandas

---

# RAG Pipeline

The following steps were implemented:

1. Load the PDF document.
2. Extract text from all pages.
3. Split the document into overlapping text chunks.
4. Generate embeddings using the **all-MiniLM-L6-v2** Sentence Transformer model.
5. Store embeddings in a persistent ChromaDB collection.
6. For each question:

   * Generate the question embedding.
   * Retrieve the Top-K most relevant chunks.
   * Build a context from the retrieved chunks.
   * Send the context and question to the LLM.
7. Compare the generated answer with the expected answer.

---

# Evaluation Metrics

The following metrics were used:

* Exact Match
* Keyword Match
* Average Length Difference
* LLM-as-a-Judge (1–5)

---

# Results

| Metric                    | LLM Only |      RAG |
| ------------------------- | -------: | -------: |
| Exact Match               |    0.00% |    0.00% |
| Keyword Match             |   31.61% |   77.26% |
| Average Length Difference |    7.131 |    4.503 |
| LLM Judge                 | 3.28 / 5 | 4.40 / 5 |

**Overall Winner:** **RAG**

---

# Analysis

The RAG pipeline significantly improved answer quality compared to the LLM-only approach.

The retrieved document context enabled the model to produce answers that were more relevant and closer to the expected answers. This improvement is reflected in both the Keyword Match score and the LLM Judge evaluation.

Although the Exact Match score remained 0%, this is due to the strict string comparison used during evaluation. Many generated answers were semantically correct but differed slightly in wording from the expected answers.

The lower Average Length Difference also indicates that the RAG-generated answers were closer in length to the reference answers.

---

# Challenges

During implementation, the following challenges were encountered:

* Groq API rate limits during repeated evaluations.
* Selecting an appropriate chunk size and overlap.
* Ensuring relevant chunks were retrieved from ChromaDB.
* Strict Exact Match evaluation leading to 0% scores despite correct answers.

---

# Conclusion

The experiment demonstrates that Retrieval-Augmented Generation improves the quality of answers by grounding the language model in the provided document.

Compared with the LLM-only approach, the RAG pipeline achieved:

* Higher Keyword Match
* Better LLM Judge scores
* More concise and relevant answers

These results confirm that providing retrieved context before generation improves response quality when answering document-specific questions.

---

# Project Files

* `leave_policy.pdf`
* `golden_test_set.csv`
* `build_database.py`
* `rag_pipeline.py`
* `rag_results.csv`
* `report.md`

---

# Future Improvements

* Replace Exact Match with semantic similarity metrics.
* Experiment with different embedding models.
* Tune chunk size and overlap.
* Retrieve more relevant chunks using reranking techniques.
* Evaluate on larger and more diverse document collections.
