import os
import sqlite3
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

from langchain_groq import ChatGroq

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain.agents import create_agent


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="LangChain: Chat with SQL DB",
    page_icon="🦜"
)

st.title("🦜 LangChain: Chat with SQL DB")


# ============================================================
# Database options
# ============================================================

LOCALDB = "USE_LOCALDB"
POSTGRES = "USE_POSTGRES"


radio_opt = [
    "Use SQLite 3 Database - student.db",
    "Connect to PostgreSQL - engineering_students"
]


selected_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat",
    options=radio_opt
)


# ============================================================
# Determine selected database
# ============================================================

if radio_opt.index(selected_opt) == 1:
    db_uri = POSTGRES
else:
    db_uri = LOCALDB


# ============================================================
# Check Groq API Key
# ============================================================

if not GROQ_API_KEY:
    st.sidebar.error("GROQ_API_KEY is not configured in .env")
    st.stop()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


# ============================================================
# Configure Database
# ============================================================

@st.cache_resource(ttl="2h")
def configure_db(db_uri):

    # ========================================================
    # SQLite
    # ========================================================

    if db_uri == LOCALDB:

        dbfilepath = (
            Path(__file__).parent / "student.db"
        ).absolute()

        creator = lambda: sqlite3.connect(
            f"file:{dbfilepath}?mode=ro",
            uri=True
        )

        engine = create_engine(
            "sqlite:///",
            creator=creator
        )

        return SQLDatabase(engine)


    # ========================================================
    # PostgreSQL
    # ========================================================

    elif db_uri == POSTGRES:

        if not POSTGRES_PASSWORD:
            st.error(
                "POSTGRES_PASSWORD is not configured in .env"
            )
            st.stop()

        postgres_uri = (
            f"postgresql+psycopg://postgres:"
            f"{POSTGRES_PASSWORD}"
            f"@localhost:5432/"
            f"engineering_students"
        )

        engine = create_engine(postgres_uri)

        return SQLDatabase(engine)


# ============================================================
# Create database connection
# ============================================================

db = configure_db(db_uri)


# ============================================================
# SQL Database Toolkit
# ============================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# ============================================================
# Create SQL Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools
)


# ============================================================
# Initialize Chat History
# ============================================================

if "messages" not in st.session_state:

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]


# ============================================================
# Clear Message History
# ============================================================

if st.sidebar.button("Clear message history"):

    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]


# ============================================================
# Display Chat History
# ============================================================

for msg in st.session_state["messages"]:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )


# ============================================================
# User Query
# ============================================================

user_query = st.chat_input(
    placeholder="Ask anything from the database"
)


# ============================================================
# Process User Query
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": user_query
        }
    )

    st.chat_message("user").write(user_query)


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        streamlit_callback = StreamlitCallbackHandler(
            st.container()
        )

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            },
            config={
                "callbacks": [streamlit_callback]
            }
        )


        # ----------------------------------------------------
        # Get final response
        # ----------------------------------------------------

        final_response = response["messages"][-1].content

        st.write(final_response)


        # ----------------------------------------------------
        # Store assistant response
        # ----------------------------------------------------

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": final_response
            }
        )