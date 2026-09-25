import os

from dotenv import load_dotenv
from crewai import Agent
from crewai.llm import LLM

from tools import web_search


load_dotenv()


# --------------------------------
# Groq LLM
# --------------------------------

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# --------------------------------
# Blog Researcher
# --------------------------------

blog_researcher = Agent(
    role="Senior Web Researcher",

    goal=(
        "Search the web and find accurate and relevant "
        "information about {topic}."
    ),

    verbose=True,

    memory=True,

    backstory=(
        "You are an expert technology researcher specializing "
        "in AI, Data Science, Machine Learning and Generative AI. "
        "You research topics using reliable web sources."
    ),

    tools=[web_search],

    allow_delegation=True,

    llm=llm
)


# --------------------------------
# Blog Writer
# --------------------------------

blog_writer = Agent(
    role="Technical Blog Writer",

    goal=(
        "Create an engaging and technically accurate blog "
        "about {topic} using the research provided."
    ),

    verbose=True,

    memory=True,

    backstory=(
        "You are an experienced technical writer who can simplify "
        "complex AI and Data Science topics into clear and engaging "
        "technical content."
    ),

    tools=[web_search],

    allow_delegation=False,

    llm=llm
)