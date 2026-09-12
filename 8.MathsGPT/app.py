import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from langchain.agents import create_agent


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set in the .env file.")
    st.stop()


# ============================================================
# Set up the Streamlit app
# ============================================================

st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)

st.title("Text To Math Problem Solver Using openai/gpt-oss-120b")


# ============================================================
# Initialize LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


# ============================================================
# Initialize the tools
# ============================================================

wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned"
)


# ============================================================
# Initialize the Math tool
# ============================================================

math_prompt = PromptTemplate.from_template(
    """
You are a mathematical assistant.

Solve the following mathematical problem and provide the answer.

Question:
{question}

Answer:
"""
)

math_chain = math_prompt | llm

calculator = Tool(
    name="Calculator",
    func=lambda question: math_chain.invoke({"question": question}).content,
    description="A tool for answering math related questions."
)


# ============================================================
# Reasoning tool
# ============================================================

prompt = """
You are an agent tasked with solving users mathematical questions.
Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below.

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = prompt_template | llm

reasoning_tool = Tool(
    name="reasoning_tool",
    func=lambda question: chain.invoke({"question": question}).content,
    description="A tool for answering logic-based and reasoning questions."
)


# ============================================================
# Initialize the agent
# ============================================================

assistant_agent = create_agent(
    model=llm,
    tools=[
        wikipedia_tool,
        calculator,
        reasoning_tool
    ]
)


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math chatbot who can answer all your maths questions"
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ============================================================
# Start the interaction
# ============================================================

question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. "
    "Each pack of blueberries contains 25 berries. "
    "How many total pieces of fruit do I have at the end?"
)


if st.button("Find my answer"):

    if question:

        with st.spinner("Generate response.."):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            st.chat_message("user").write(question)

            # New LangChain agent invocation
            response = assistant_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )

            # Get the final response
            response = response["messages"][-1].content

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the question")