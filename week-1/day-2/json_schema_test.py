from groq import Groq
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()


# ======================================
# GROQ Client
# ======================================

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ======================================
# Read prompts from prompt_pack.md
# ======================================

with open(
    "prompt_pack.md",
    "r",
    encoding="utf-8"
) as file:

    content = file.read()


# Split prompts:
# # Prompt 1
# # Prompt 2
# ...

prompts = re.split(
    r"# Prompt \d+.*",
    content
)


# Remove empty entries

prompts = [
    p.strip()
    for p in prompts
    if p.strip()
]


print(
    f"Found {len(prompts)} prompts"
)


# ======================================
# Run prompts
# ======================================

with open(
    "prompt_results.txt",
    "w",
    encoding="utf-8"
) as result_file:


    for index, prompt in enumerate(
        prompts,
        start=1
    ):


        print(
            f"\nRunning Prompt {index}"
        )


        # Add JSON instruction
        final_prompt = f"""

You must follow the instructions exactly.

Return ONLY valid JSON.

No markdown.
No explanation.
No extra text.

Task:

{prompt}

"""


        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],

            temperature=0

        )


        output = (
            response
            .choices[0]
            .message
            .content
        )


        # Save raw output

        result_file.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        result_file.write(
            f"PROMPT {index}\n"
        )

        result_file.write(
            "=" * 80
            + "\n\n"
        )

        result_file.write(
            output
            + "\n"
        )


        # ======================================
        # JSON Validation
        # ======================================


        clean_output = (
            output
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )


        try:

            json_data = json.loads(
                clean_output
            )


            print(
                f"Prompt {index}: JSON PASSED"
            )


            result_file.write(
                "\nJSON VALIDATION: PASSED\n"
            )


        except Exception as e:


            print(
                f"Prompt {index}: JSON FAILED"
            )

            print(e)


            result_file.write(
                "\nJSON VALIDATION: FAILED\n"
            )


print(
    "\nCompleted all prompts"
)

print(
    "Results saved: prompt_results.txt"
)