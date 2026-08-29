import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b"
)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Provide the most accurate response based on the context.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


# --------------------------------------------------
# Create embeddings + vector database
# --------------------------------------------------

def create_vector_embedding():

    if "vectors" not in st.session_state:

        # Hugging Face embedding model
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader(
            "research_papers"
        )

        st.session_state.docs = (
            st.session_state.loader.load()
        )

        # Split documents
        st.session_state.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter
            .split_documents(st.session_state.docs[:50])
        )

        # Create FAISS vector database
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("RAG Document Q&A with Groq + Hugging Face")

user_prompt = st.text_input(
    "Enter your query from the research paper"
)


# --------------------------------------------------
# Create vector database
# --------------------------------------------------

if st.button("Document Embedding"):

    create_vector_embedding()

    st.write("Vector Database is ready")


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if user_prompt:

    if "vectors" not in st.session_state:

        st.warning(
            "Please click 'Document Embedding' first."
        )

    else:

        # Document chain
        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        # Retriever
        retriever = (
            st.session_state.vectors
            .as_retriever()
        )

        # Retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        # Start timer
        start = time.process_time()

        # Run RAG
        response = retrieval_chain.invoke(
            {"input": user_prompt}
        )

        # Response time
        print(
            f"Response time: "
            f"{time.process_time() - start}"
        )

        # Display answer
        st.write(response["answer"])


        # --------------------------------------------------
        # Show retrieved documents
        # --------------------------------------------------

        with st.expander(
            "Document Similarity Search"
        ):

            for i, doc in enumerate(
                response["context"]
            ):

                st.write(
                    f"Document {i + 1}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "------------------------"
                )