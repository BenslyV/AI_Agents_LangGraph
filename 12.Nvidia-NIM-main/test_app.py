from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)


completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-v4-pro-0813",
  messages=[{"role":"user","content":"Write a Python Program to add two numbers."}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  seed=42,
  extra_body={"chat_template_kwargs":{"thinking":False}},
  stream=True
)

print(completion.choices[0].message.content)