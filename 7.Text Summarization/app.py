import os
import validators
import streamlit as st

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
)


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# Streamlit App
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
    label_visibility="collapsed"
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


# ============================================================
# Prompt
# ============================================================

prompt_template = """
Provide a summary of the following content in approximately 300 words.

Content:
{context}
"""

prompt = PromptTemplate.from_template(prompt_template)


# ============================================================
# Summarization Chain
# ============================================================

summarization_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# ============================================================
# Button
# ============================================================

if st.button("Summarize the Content from YT or Website"):

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not groq_api_key:
        st.error("GROQ_API_KEY is not set in the .env file.")

    elif not generic_url.strip():
        st.error("Please provide a URL.")

    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid URL. "
            "It can be a YouTube video URL or a website URL."
        )

    else:

        try:

            with st.spinner("Loading and summarizing..."):

                # ------------------------------------------------
                # Load YouTube content
                # ------------------------------------------------

                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True
                    )

                # ------------------------------------------------
                # Load normal website
                # ------------------------------------------------

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 "
                                "(Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 "
                                "Safari/537.36"
                            )
                        }
                    )

                # ------------------------------------------------
                # Load documents
                # ------------------------------------------------

                docs = loader.load()

                # ------------------------------------------------
                # Generate summary
                # ------------------------------------------------

                output_summary = summarization_chain.invoke(
                    {
                        "context": docs
                    }
                )

                # ------------------------------------------------
                # Display result
                # ------------------------------------------------

                st.success(output_summary)

        except Exception as e:

            st.exception(e)