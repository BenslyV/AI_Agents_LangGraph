import os

from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool

from ddgs import DDGS


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Check Groq API Key
# ============================================================

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is not set in the .env file"
    )


# ============================================================
# 3. Create Groq LLM
# ============================================================

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ============================================================
# 4. Create Web Search Tool
# ============================================================

@tool("Web Search")
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Returns titles, URLs and descriptions of search results.
    """

    try:

        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No search results found."

        output = []

        for result in results:

            title = result.get("title", "")
            url = result.get("href", "")
            description = result.get("body", "")

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return f"Web search failed: {str(e)}"


# ============================================================
# 5. Researcher Agent
# ============================================================

blog_researcher = Agent(

    role="Senior Web Researcher",

    goal=(
        "Research {topic} using the Web Search tool and "
        "provide accurate, relevant and well-organized "
        "information."
    ),

    verbose=True,

    # IMPORTANT:
    # Disable CrewAI memory.
    # This avoids Chroma/OpenAI embedding requirements.
    memory=False,

    backstory=(
        "You are an expert technology researcher specializing "
        "in Artificial Intelligence, Machine Learning, "
        "Deep Learning and Data Science. "
        "You research information from multiple web sources "
        "and identify the most important facts."
    ),

    tools=[web_search],

    allow_delegation=False,

    llm=llm
)


# ============================================================
# 6. Blog Writer Agent
# ============================================================

blog_writer = Agent(

    role="Technical Blog Writer",

    goal=(
        "Write a clear, engaging and technically accurate "
        "blog about {topic} using the research provided "
        "by the researcher."
    ),

    verbose=True,

    # Disable CrewAI memory
    memory=False,

    backstory=(
        "You are an experienced technical writer specializing "
        "in AI, Machine Learning, Data Science and Generative AI. "
        "You simplify complex technical concepts so that "
        "learners can understand them easily."
    ),

    # The writer doesn't need web search.
    # It receives the research from the previous task.
    tools=[],

    allow_delegation=False,

    llm=llm
)


# ============================================================
# 7. Research Task
# ============================================================

research_task = Task(

    description=(
        "Research the topic {topic} using the Web Search tool. "
        "Search multiple sources and identify the important "
        "concepts, definitions, relationships, differences "
        "and practical examples."
    ),

    expected_output=(
        "A detailed research report about {topic}. "
        "The report should contain accurate definitions, "
        "important concepts, differences, relationships "
        "and practical examples."
    ),

    tools=[web_search],

    agent=blog_researcher
)


# ============================================================
# 8. Writing Task
# ============================================================

write_task = Task(

    description=(
        "Using the research produced by the previous task, "
        "write a clear and technically accurate blog about "
        "{topic}. "
        "Use headings, explanations, comparisons and examples. "
        "The blog should be useful for someone learning "
        "AI and Data Science."
    ),

    expected_output=(
        "A complete technical blog post about {topic} "
        "with a title, introduction, clear sections, "
        "comparisons, examples and conclusion."
    ),

    agent=blog_writer,

    async_execution=False,

    output_file="new-blog-post.md"
)


# ============================================================
# 9. Create Crew
# ============================================================

crew = Crew(

    agents=[
        blog_researcher,
        blog_writer
    ],

    tasks=[
        research_task,
        write_task
    ],

    process=Process.sequential,

    # IMPORTANT:
    # Disable CrewAI memory to avoid Chroma/OpenAI embeddings.
    memory=False,

    cache=True,

    max_rpm=100
)


# ============================================================
# 10. Start Crew
# ============================================================

result = crew.kickoff(

    inputs={
        "topic": "AI VS ML VS DL vs Data Science"
    }
)


# ============================================================
# 11. Print Final Result
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)
print()

print(result)

print()
print("=" * 70)
print("Blog saved to: new-blog-post.md")
print("=" * 70)