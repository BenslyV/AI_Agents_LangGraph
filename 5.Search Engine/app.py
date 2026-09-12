import streamlit as st
import os

from dotenv import load_dotenv

# ============================================================
# LLM
# ============================================================

from langchain_groq import ChatGroq


# ============================================================
# Arxiv
# ============================================================

from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


# ============================================================
# Wikipedia
# ============================================================

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# ============================================================
# DuckDuckGo
# ============================================================

from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# Custom RAG Tool
# ============================================================

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import create_retriever_tool


# ============================================================
# Agent
# ============================================================

from langchain.agents import create_agent


# ============================================================
# Streamlit Callback
# ============================================================

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# ============================================================
# Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Arxiv and Wikipedia Tools
# ============================================================

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200
)

arxiv = ArxivQueryRun(
    api_wrapper=arxiv_wrapper
)


api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper
)


# ============================================================
# DuckDuckGo Search
# ============================================================

search = DuckDuckGoSearchRun(
    name="Search"
)


# ============================================================
# Custom RAG Tool
# ============================================================

# Load LangSmith documentation
loader = WebBaseLoader(
    "https://docs.smith.langchain.com/"
)

docs = loader.load()


# Split documents
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Vector database
vectordb = FAISS.from_documents(
    documents,
    embeddings
)


# Retriever
retriever = vectordb.as_retriever()


# Convert retriever into a tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith-search",
    "Search any information about LangSmith."
)


print(retriever_tool.name)


# ============================================================
# Final Tool List
# ============================================================

tools = [
    search,
    arxiv,
    wiki,
    retriever_tool
]

print(tools)


# ============================================================
# Streamlit UI
# ============================================================

st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler`
to display the thoughts and actions of an agent in an
interactive Streamlit app.
"""


# ============================================================
# Sidebar for settings
# ============================================================

st.sidebar.title("Settings")


# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?"
        }
    ]


for msg in st.session_state.messages:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )


# ============================================================
# User Input
# ============================================================

if prompt := st.chat_input(
    placeholder="What is machine learning?"
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)


    # ========================================================
    # Groq LLM
    # ========================================================

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        streaming=True
    )


    # ========================================================
    # Create Agent
    # ========================================================

    search_agent = create_agent(
        model=llm,
        tools=tools
    )


    # ========================================================
    # Agent Response
    # ========================================================

    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )


        response = search_agent.invoke(
            {
                "messages": st.session_state.messages
            },
            config={
                "callbacks": [st_cb]
            }
        )


        # Get final response
        response = response["messages"][-1].content


        # Save response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


        # Display response
        st.write(response)