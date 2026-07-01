# Day 3 Report

## Documents

1. Emumba Leave Policy (Pages 1–5)
2. Emumba Leave Policy (Pages 6–9)

> Note: Only one PDF was available, so the 25-question test set was created from different sections of the same document.

## Models Evaluated

* **LLM-A:** llama-3.3-70b-versatile
* **LLM-B:** openai/gpt-oss-120b

## Evaluation Metrics

1. Exact Match
2. Keyword Match
3. Length Difference
4. LLM Judge Score

## Results

| Metric                    |   LLM-A   |    LLM-B   |
| ------------------------- | :-------: | :--------: |
| Exact Match               | 8% (2/25) | 28% (7/25) |
| Average Keyword Match     |   83.2%   |    85.4%   |
| Average Length Difference |    0.57   |    0.71    |
| Average LLM Judge Score   |  4.72 / 5 |  4.88 / 5  |

## Observations

* Both models answered most factual questions correctly using the Leave Policy document.
* Exact Match scores were relatively low because the models often paraphrased the expected answers instead of using identical wording.
* Keyword Match scores remained high, indicating that the important information was generally preserved.
* LLM-B achieved a slightly higher Exact Match, Keyword Match, and LLM Judge score than LLM-A.
* Both models occasionally produced longer responses than the expected answers, which increased the Length Difference metric.

## Conclusion

A 25-question golden test set was created from the Leave Policy document and used to evaluate two LLMs. The evaluation showed that Exact Match alone is not sufficient for assessing LLM responses, as semantically correct answers may differ in wording. Keyword Match and LLM Judge scores provided a more reliable measure of answer quality. Overall, both models performed well, with LLM-B showing slightly better overall performance on this test set.
