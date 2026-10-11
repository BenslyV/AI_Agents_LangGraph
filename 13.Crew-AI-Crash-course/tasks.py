from crewai import Task

from tools import web_search
from agents import blog_researcher, blog_writer


# --------------------------------
# Research Task
# --------------------------------

research_task = Task(
    description=(
        "Search the web for detailed and reliable information "
        "about {topic}. Research the topic from multiple sources "
        "and identify the most important concepts and facts."
    ),

    expected_output=(
        "A comprehensive research report about {topic}, "
        "covering the key concepts, important facts and "
        "relevant information found through web research."
    ),

    tools=[web_search],

    agent=blog_researcher
)


# --------------------------------
# Writing Task
# --------------------------------

write_task = Task(
    description=(
        "Using the research about {topic}, create a clear, "
        "engaging and technically accurate blog post."
    ),

    expected_output=(
        "A well-structured technical blog post about {topic} "
        "with clear explanations and examples."
    ),

    tools=[web_search],

    agent=blog_writer,

    async_execution=False,

    output_file="new-blog-post.md"
)