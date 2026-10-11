import os
import time
import streamlit as st

from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    NVIDIAEmbeddings
)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    st.error("NVIDIA_API_KEY is not configured.")
    st.stop()


# ============================================================
# Streamlit UI
# ============================================================

st.title("NVIDIA NIM RAG Demo")


# ============================================================
# NVIDIA LLM
# ============================================================

llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-v4-flash-0731",
    temperature=0
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based only on the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

<context>
{context}
</context>

Question:
{question}

Answer:
"""
)


# ============================================================
# Create Vector Store
# ============================================================

def vector_embedding():

    if "vectors" not in st.session_state:

        # ----------------------------------------------------
        # NVIDIA Embeddings
        # ----------------------------------------------------

        embeddings = NVIDIAEmbeddings()

        # ----------------------------------------------------
        # Load PDF documents
        # ----------------------------------------------------

        loader = PyPDFDirectoryLoader("./us_census")

        documents = loader.load()

        # ----------------------------------------------------
        # Split documents
        # ----------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        final_documents = text_splitter.split_documents(
            documents[:30]
        )

        # ----------------------------------------------------
        # Create FAISS vector store
        # ----------------------------------------------------

        vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        # ----------------------------------------------------
        # Save in Streamlit session
        # ----------------------------------------------------

        st.session_state.vectors = vectors


# ============================================================
# Create embeddings button
# ============================================================

if st.button("Documents Embedding"):

    with st.spinner("Creating vector store..."):

        vector_embedding()

    st.success("Vector Store DB is Ready")


# ============================================================
# Question
# ============================================================

question = st.text_input(
    "Enter Your Question From Documents"
)


# ============================================================
# RAG
# ============================================================

if question:

    if "vectors" not in st.session_state:

        st.warning(
            "Please click 'Documents Embedding' first."
        )

    else:

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        retriever = st.session_state.vectors.as_retriever()

        # ----------------------------------------------------
        # Retrieve relevant documents
        # ----------------------------------------------------

        start = time.perf_counter()

        retrieved_docs = retriever.invoke(question)

        # ----------------------------------------------------
        # Combine document contents
        # ----------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # ----------------------------------------------------
        # Modern LangChain pipeline
        # ----------------------------------------------------

        rag_chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response_time = time.perf_counter() - start

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.write(answer)

        st.caption(
            f"Response time: {response_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Document similarity search
        # ----------------------------------------------------

        with st.expander("Document Similarity Search"):

            for i, doc in enumerate(retrieved_docs):

                st.write(
                    f"Document Chunk {i + 1}"
                )

                st.write(doc.page_content)

                st.write("--------------------------------")