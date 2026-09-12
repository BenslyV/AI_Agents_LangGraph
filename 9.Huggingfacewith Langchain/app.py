import os
import validators
import streamlit as st

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

hf_api_key = os.getenv("HF_TOKEN")

if not hf_api_key:
    raise ValueError(
        "HF_TOKEN is not set in the environment variables."
    )


# ============================================================
# Streamlit APP
# ============================================================

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# ============================================================
# URL Input
# ============================================================

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Enter YouTube or Website URL"
)


# ============================================================
# Hugging Face Model
# ============================================================

repo_id = "deepseek-ai/DeepSeek-V3-0324"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=hf_api_key
)

# Convert HuggingFaceEndpoint into Chat Model
chat_model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# Summarization Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert summarization assistant.

            Summarize the provided content in approximately 300 words.

            Focus on:
            - Main ideas
            - Important facts
            - Key points
            - Important conclusions

            Do not add information that is not present
            in the provided content.
            """
        ),
        (
            "human",
            """
            Here is the content to summarize:

            {context}
            """
        ),
    ]
)


# ============================================================
# NEW LANGCHAIN 1.x CHAIN
# ============================================================

def format_documents(docs):
    """
    Convert LangChain Document objects into plain text.
    """
    return "\n\n".join(
        document.page_content
        for document in docs
    )


# LCEL / Runnable chain
document_chain = prompt | chat_model


# ============================================================
# Summarize Button
# ============================================================

if st.button("Summarize the Content from YT or Website"):

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not generic_url.strip():

        st.error(
            "Please provide a URL."
        )

    elif not validators.url(generic_url):

        st.error(
            "Please enter a valid URL. "
            "It can be a YouTube video URL or website URL."
        )

    else:

        try:

            with st.spinner(
                "Loading and summarizing the content..."
            ):

                # ====================================================
                # Load YouTube or Website
                # ====================================================

                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 "
                                "(Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                # ====================================================
                # Load Documents
                # ====================================================

                docs = loader.load()

                if not docs:

                    st.error(
                        "No content could be extracted from the URL."
                    )

                    st.stop()

                # ====================================================
                # Convert Documents -> Text
                # ====================================================

                context = format_documents(docs)

                # ====================================================
                # NEW LANGCHAIN 1.x INVOCATION
                # ====================================================

                response = document_chain.invoke(
                    {
                        "context": context
                    }
                )

                # ====================================================
                # Display Summary
                # ====================================================

                st.subheader("Summary")

                st.write(response.content)

        except Exception as e:

            st.error(
                "An error occurred while processing the URL."
            )

            st.exception(e)