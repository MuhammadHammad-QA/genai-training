import tiktoken
from transformers import AutoTokenizer
import pandas as pd


# -----------------------------
# 1. Create 30 diverse strings
# -----------------------------

texts = [
    "Hello world",
    "I love Python programming",
    "Generative AI is interesting",
    "Machine learning is powerful",
    "OpenAI creates AI models",

    "def add(a, b): return a + b",
    "print('Hello World')",
    "SELECT * FROM users WHERE id=1",
    "for i in range(10): print(i)",
    "git commit -m 'update code'",

    "😀😀😀😀😀",
    "🚀🔥🎉",
    "😀 Hello AI 🤖",
    "👍👍👍",

    "こんにちは",
    "こんにちは世界",
    "你好世界",
    "我喜欢人工智能",

    "مرحبا",
    "السلام علیکم",
    "أنا أتعلم الذكاء الاصطناعي",

    "नमस्ते दुनिया",
    "मैं Python सीख रहा हूँ",

    "1234567890",
    "100 + 200 = 300",

    "Python 😀 مرحبا",
    "AI未来科技",
    "hello123@test.com",
    "https://www.example.com",
    "The quick brown fox jumps over the lazy dog"
]


# -----------------------------
# 2. Load Tokenizers
# -----------------------------

tiktoken_encoder = tiktoken.get_encoding(
    "cl100k_base"
)
hf_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)



# -----------------------------
# 3. Compare token counts
# -----------------------------

results = []


for text in texts:

    # Count tokens using tiktoken
    tiktoken_count = len(
        tiktoken_encoder.encode(text)
    )


    # Count tokens using Hugging Face
    hf_tokens = hf_tokenizer.tokenize(text)
    hf_count = len(hf_tokens)


    results.append(
        {
            "text": text,
            "tiktoken_count": tiktoken_count,
            "bert_token_count": hf_count
        }
    )



# -----------------------------
# 4. Save results
# -----------------------------

df = pd.DataFrame(results)

df.to_csv(
    "token_counts.csv",
    index=False
)


print("Token comparison completed!")
print(df)