# ============================================================
# Conversational RAG with PDF + Chat History
# ============================================================
#
# Flow:
#
# PDF Upload
#     ↓
# Load PDF
#     ↓
# Split PDF into chunks
#     ↓
# Create embeddings
#     ↓
# Store embeddings in Chroma
#     ↓
# Retrieve relevant chunks
#     ↓
# Use chat history to understand follow-up questions
#     ↓
# Send retrieved context + question to Groq LLM
#     ↓
# Generate answer
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import os

import streamlit as st
from dotenv import load_dotenv

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Prompt related classes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Manage chat history with a runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# Groq LLM
from langchain_groq import ChatGroq

# Hugging Face embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma vector database
from langchain_chroma import Chroma

# RAG chains
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


# ============================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ============================================================
#
# The .env file should contain:
#
# GROQ_API_KEY=your_groq_api_key
#
# HF_TOKEN=your_huggingface_token
#
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# STEP 3: CHECK GROQ API KEY
# ============================================================

if not groq_api_key:
    st.error("GROQ_API_KEY is not configured in the .env file.")
    st.stop()


# ============================================================
# STEP 4: CREATE THE GROQ LLM
# ============================================================

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


# ============================================================
# STEP 5: CREATE THE EMBEDDING MODEL
# ============================================================
#
# The embedding model converts text into numerical vectors.
#
# Example:
#
# "What is machine learning?"
#
#       ↓
#
# [0.12, -0.34, 0.87, ...]
#
# These vectors are stored in Chroma and used for
# similarity search.
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ============================================================
# STEP 6: STREAMLIT USER INTERFACE
# ============================================================

st.title("Conversational RAG with PDF and Chat History")

st.write(
    "Upload PDF files and ask questions about their content."
)


# ============================================================
# STEP 7: CREATE SESSION ID
# ============================================================
#
# Session ID allows us to maintain separate chat histories.
#
# Example:
#
# Session 1 → default_session
# Session 2 → user_123
#
# Each session can have its own conversation history.
# ============================================================

session_id = st.text_input(
    "Session ID",
    value="default_session"
)


# ============================================================
# STEP 8: CREATE CHAT HISTORY STORAGE
# ============================================================
#
# st.session_state keeps data while the Streamlit session
# is running.
#
# "store" will contain:
#
# {
#     "default_session": ChatMessageHistory()
# }
#
# ============================================================

if "store" not in st.session_state:
    st.session_state.store = {}


# ============================================================
# STEP 9: UPLOAD PDF FILES
# ============================================================

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)


# ============================================================
# STEP 10: PROCESS THE PDF FILES
# ============================================================

if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:

        # ----------------------------------------------------
        # Save the uploaded PDF temporarily
        # ----------------------------------------------------

        temp_pdf = "./temp.pdf"

        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        # ----------------------------------------------------
        # Load the PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_pdf)

        docs = loader.load()

        # ----------------------------------------------------
        # Add pages to the complete document list
        # ----------------------------------------------------

        documents.extend(docs)


    # ========================================================
    # STEP 11: SPLIT DOCUMENTS INTO CHUNKS
    # ========================================================
    #
    # Large documents are divided into smaller chunks.
    #
    # chunk_size = 5000
    # chunk_overlap = 500
    #
    # Overlap helps preserve context between chunks.
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )

    splits = text_splitter.split_documents(documents)


    # ========================================================
    # STEP 12: CREATE CHROMA VECTOR DATABASE
    # ========================================================
    #
    # Each chunk is converted into an embedding and stored
    # in Chroma.
    # ========================================================

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )


    # ========================================================
    # STEP 13: CREATE RETRIEVER
    # ========================================================
    #
    # Retriever searches Chroma for documents relevant to
    # the user's question.
    # ========================================================

    retriever = vectorstore.as_retriever()


    # ========================================================
    # STEP 14: CREATE CONTEXTUALIZATION PROMPT
    # ========================================================
    #
    # This is important for follow-up questions.
    #
    # Example:
    #
    # User: What is RAG?
    #
    # User: How does it work?
    #
    # "How does it work?" by itself is ambiguous.
    #
    # Chat history allows the system to understand that
    # "it" refers to RAG.
    # ========================================================

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be "
        "understood without the chat history. "
        "Do NOT answer the question. "
        "Just reformulate it if needed, otherwise return it as is."
    )


    # ========================================================
    # STEP 15: CREATE CONTEXTUALIZATION PROMPT TEMPLATE
    # ========================================================

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),

            MessagesPlaceholder(
                "chat_history"
            ),

            ("human", "{input}")
        ]
    )


    # ========================================================
    # STEP 16: CREATE HISTORY-AWARE RETRIEVER
    # ========================================================
    #
    # Instead of directly searching using the user's question,
    # the LLM first understands the question using chat history.
    #
    # User question
    #       +
    # Chat history
    #       ↓
    # Standalone question
    #       ↓
    # Retriever
    #       ↓
    # Relevant PDF chunks
    # ========================================================

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )


    # ========================================================
    # STEP 17: CREATE QA SYSTEM PROMPT
    # ========================================================
    #
    # This prompt tells the LLM how to answer questions using
    # the retrieved PDF content.
    # ========================================================

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )


    # ========================================================
    # STEP 18: CREATE QA PROMPT
    # ========================================================

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),

            MessagesPlaceholder(
                "chat_history"
            ),

            ("human", "{input}")
        ]
    )


    # ========================================================
    # STEP 19: CREATE DOCUMENT QA CHAIN
    # ========================================================
    #
    # This chain takes:
    #
    # Retrieved documents
    #        +
    # User question
    #        +
    # Chat history
    #
    # and asks the LLM to generate an answer.
    # ========================================================

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )


    # ========================================================
    # STEP 20: CREATE COMPLETE RAG CHAIN
    # ========================================================
    #
    # Complete flow:
    #
    # User question
    #       ↓
    # History-aware retriever
    #       ↓
    # Relevant documents
    #       ↓
    # QA chain
    #       ↓
    # LLM
    #       ↓
    # Answer
    # ========================================================

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )


    # ========================================================
    # STEP 21: FUNCTION TO GET SESSION CHAT HISTORY
    # ========================================================
    #
    # Each session ID gets its own ChatMessageHistory object.
    # ========================================================

    def get_session_history(session_id: str) -> BaseChatMessageHistory:

        if session_id not in st.session_state.store:

            st.session_state.store[session_id] = (
                ChatMessageHistory()
            )

        return st.session_state.store[session_id]


    # ========================================================
    # STEP 22: ADD MESSAGE HISTORY TO THE RAG CHAIN
    # ========================================================
    #
    # RunnableWithMessageHistory automatically:
    #
    # 1. Reads previous chat history
    # 2. Sends it to the chain
    # 3. Stores the new user question
    # 4. Stores the assistant answer
    #
    # ========================================================

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,

        input_messages_key="input",

        history_messages_key="chat_history",

        output_messages_key="answer"
    )


    # ========================================================
    # STEP 23: GET USER QUESTION
    # ========================================================

    user_input = st.text_input(
        "Your question:"
    )


    # ========================================================
    # STEP 24: PROCESS USER QUESTION
    # ========================================================

    if user_input:

        response = conversational_rag_chain.invoke(
            {"input": user_input},

            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )


        # ====================================================
        # STEP 25: DISPLAY ANSWER
        # ====================================================

        st.write(
            "Assistant:",
            response["answer"]
        )


        # ====================================================
        # STEP 26: DISPLAY CHAT HISTORY
        # ====================================================

        session_history = get_session_history(session_id)

        st.write(
            "Chat History:",
            session_history.messages
        )