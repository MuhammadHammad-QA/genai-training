# Day 3 Evaluation Report

## Dataset

- PDF: sample_pdf.pdf
- Questions: 25
- Model A: llama-3.3-70b-versatile
- Model B: openai/gpt-oss-120b

## Evaluation Metrics

| Metric | Model A | Model B |
|--------|---------|---------|
| Exact Match (%) | 0.00 | 4.00 |
| Keyword Match (%) | 45.38 | 53.82 |
| Avg Length Difference | 0.883 | 1.348 |
| LLM Judge (/5) | 3.92 | 4.28 |

## Overall Scores

- llama-3.3-70b-versatile: 114.95
- openai/gpt-oss-120b: 129.94

**Winner:** openai/gpt-oss-120b

## Observations

- Model B achieved the highest overall score.
- Model B generally produced more accurate answers.
- Keyword Match provided a more meaningful metric than Exact Match because models often paraphrased answers.
- LLM Judge effectively evaluated semantic correctness even when wording differed.
- Exact Match alone was insufficient for evaluating long-form answers.
- Average Length Difference highlighted verbosity differences between models.
