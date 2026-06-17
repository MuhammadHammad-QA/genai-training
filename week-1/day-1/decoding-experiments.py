from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

prompt = "Invent a new smartphone product and write a marketing pitch for it"

# 10 decoding configurations
configs = [
    {"temperature": 0.0, "top_p": 1.0},
    {"temperature": 0.4, "top_p": 1.0},
    {"temperature": 0.8, "top_p": 1.0},
    {"temperature": 1.2, "top_p": 1.0},
    {"temperature": 1.6, "top_p": 1.0},
    {"temperature": 2.0, "top_p": 1.0},
    {"temperature": 1.0, "top_p": 0.5},
    {"temperature": 0.8, "top_p": 0.7},
    {"temperature": 0.6, "top_p": 0.9},
    {"temperature": 0.2, "top_p": 0.5},
]

with open("decoding_results.txt", "w", encoding="utf-8") as f:
    for i, config in enumerate(configs, start=1):

        print("=" * 80)
        print(
            f"Run {i} | Temperature={config['temperature']} | Top-p={config['top_p']}"
        )
        print("=" * 80)
        
        f.write("=" * 80 + "\n")
        f.write(f"Run {i} | Temperature={config['temperature']} | Top-p={config['top_p']}\n")
        f.write("=" * 80 + "\n")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )

        output = response.choices[0].message.content

        print(output)
        print("\n\n")
        
        f.write(output + "\n\n")
