import os

import streamlit as st

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
# PAGE
# ============================================================

st.title("Code Llama - Hugging Face")


# ============================================================
# CHAT HISTORY
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# USER INPUT
# ============================================================

prompt = st.text_area(
    "Enter your Prompt"
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

if st.button("Generate"):

    if prompt:

        st.session_state.history.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        response = client.chat.completions.create(
            model=model,
            messages=st.session_state.history,
            max_tokens=500,
            temperature=0.7
        )

        actual_response = response.choices[0].message.content

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": actual_response
            }
        )

        st.write(actual_response)