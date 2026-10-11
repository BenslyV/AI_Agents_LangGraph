import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM


# --------------------------------
# Load environment variables
# --------------------------------
load_dotenv()


# --------------------------------
# Create Groq LLM
# --------------------------------
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# --------------------------------
# Create Agent
# --------------------------------
agent = Agent(
    role="AI Assistant",
    goal="Answer questions accurately and clearly",
    backstory=(
        "You are a knowledgeable AI assistant "
        "who provides clear and accurate answers."
    ),
    llm=llm,
    verbose=True
)


# --------------------------------
# Create Task
# --------------------------------
task = Task(
    description=(
        "Explain what LangChain is and how it differs "
        "from CrewAI"
    ),
    expected_output=(
        "A clear and concise explanation of LangChain "
        "and its differences from CrewAI."
    ),
    agent=agent
)


# --------------------------------
# Create Crew
# --------------------------------
crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)


# --------------------------------
# Run Crew
# --------------------------------
result = crew.kickoff()


# --------------------------------
# Print Result
# --------------------------------
print("\n\n===== FINAL ANSWER =====\n")
print(result)