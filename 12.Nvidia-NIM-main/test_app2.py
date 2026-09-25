import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Create NVIDIA OpenAI-compatible client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Create streaming response
stream = client.chat.completions.create(
    model="deepseek-ai/deepseek-v4-flash-0731",
    messages=[
        {
            "role": "user",
            "content": "Write a limerick about the wonders of GPU computing."
        }
    ],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": "high"
        }
    },
    stream=True,
)

# Print response as it arrives
for chunk in stream:

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta

    if delta.content:
        print(delta.content, end="", flush=True)

print()