import os

import gradio as gr

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# MODEL
# ============================================================

model = "codellama/CodeLlama-7b-Instruct-hf"


# ============================================================
# CHAT HISTORY
# ============================================================

history = []


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(prompt):

    history.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=history,
        max_tokens=500,
        temperature=0.7
    )

    actual_response = response.choices[0].message.content

    history.append(
        {
            "role": "assistant",
            "content": actual_response
        }
    )

    return actual_response


# ============================================================
# GRADIO INTERFACE
# ============================================================

interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter your Prompt"
    ),
    outputs="text"
)


# ============================================================
# LAUNCH
# ============================================================

interface.launch()