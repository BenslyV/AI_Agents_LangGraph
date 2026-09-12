# ============================================================
# rag_service.py
# Conversational RAG Service
# ============================================================
#
# This file contains the complete RAG logic:
#
# PDF
#   ↓
# Document Loader
#   ↓
# Text Splitter
#   ↓
# Hugging Face Embeddings
#   ↓
# Chroma Vector Database
#   ↓
# History-Aware Retriever
#   ↓
# Groq LLM
#   ↓
# Answer
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import os

from dotenv import load_dotenv

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Prompt classes
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Message history wrapper
from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

# Groq LLM
from langchain_groq import ChatGroq

# Hugging Face embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Text splitter
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# Chroma vector database
from langchain_chroma import Chroma

# RAG chains
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)


# ============================================================
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured in the .env file."
    )


# ============================================================
# STEP 3: CREATE GROQ LLM
# ============================================================

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY
)


# ============================================================
# STEP 4: CREATE HUGGING FACE EMBEDDING MODEL
# ============================================================
#
# Converts text into numerical vectors.
#
# These vectors are stored in Chroma and later used
# for similarity search.
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ============================================================
# STEP 5: CHAT HISTORY STORAGE
# ============================================================
#
# FastAPI does not have Streamlit's st.session_state.
#
# Therefore, we maintain our own Python dictionary.
#
# Example:
#
# {
#     "user1": ChatMessageHistory(),
#     "user2": ChatMessageHistory()
# }
#
# Each session_id gets separate conversation history.
# ============================================================

store = {}


def get_session_history(
    session_id: str
) -> BaseChatMessageHistory:

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


# ============================================================
# STEP 6: CREATE THE RAG CHAIN
# ============================================================
#
# The chain will be created after PDF documents are uploaded.
# ============================================================

conversational_rag_chain = None


# ============================================================
# STEP 7: PROCESS PDF DOCUMENTS
# ============================================================

def process_pdfs(pdf_paths: list[str]):

    global conversational_rag_chain

    documents = []


    # --------------------------------------------------------
    # Load every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        documents.extend(docs)


    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )

    splits = text_splitter.split_documents(documents)


    # --------------------------------------------------------
    # Create Chroma vector database
    # --------------------------------------------------------

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )


    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever()


    # ========================================================
    # STEP 8: CREATE HISTORY-AWARE RETRIEVER
    # ========================================================

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be "
        "understood without the chat history. "
        "Do NOT answer the question. "
        "Just reformulate it if needed, otherwise return it as is."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                contextualize_q_system_prompt
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            )
        ]
    )


    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )


    # ========================================================
    # STEP 9: CREATE QA PROMPT
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


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),

            MessagesPlaceholder(
                "chat_history"
            ),

            (
                "human",
                "{input}"
            )
        ]
    )


    # ========================================================
    # STEP 10: CREATE DOCUMENT QA CHAIN
    # ========================================================

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )


    # ========================================================
    # STEP 11: CREATE COMPLETE RAG CHAIN
    # ========================================================

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )


    # ========================================================
    # STEP 12: ADD CHAT HISTORY TO RAG CHAIN
    # ========================================================

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,

        input_messages_key="input",

        history_messages_key="chat_history",

        output_messages_key="answer"
    )


    return len(documents), len(splits)


# ============================================================
# STEP 13: ASK QUESTION
# ============================================================

def ask_question(
    question: str,
    session_id: str
):

    if conversational_rag_chain is None:
        raise ValueError(
            "Please upload a PDF before asking a question."
        )


    response = conversational_rag_chain.invoke(
        {"input": question},

        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )


    return response["answer"]


# ============================================================
# STEP 14: GET CHAT HISTORY
# ============================================================

def get_chat_history(
    session_id: str
):

    history = get_session_history(session_id)

    return history.messages