import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is missing from .env")


# ============================================================
# 2. YOUTUBE SEARCH FUNCTION
# ============================================================

def search_youtube(query: str, max_results: int = 5) -> str:

    youtube = build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )

    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )

    response = request.execute()

    results = []

    for item in response.get("items", []):

        video_id = item["id"]["videoId"]
        snippet = item["snippet"]

        title = snippet.get("title", "")
        channel = snippet.get("channelTitle", "")
        published = snippet.get("publishedAt", "")
        description = snippet.get("description", "")

        url = f"https://www.youtube.com/watch?v={video_id}"

        results.append(
            f"""
Video ID: {video_id}
Title: {title}
Channel: {channel}
Published: {published}
URL: {url}
Description: {description}
----------------------------------------
"""
        )

    if not results:
        return "No YouTube videos found."

    return "\n".join(results)


# ============================================================
# 3. YOUTUBE TRANSCRIPT FUNCTION
# ============================================================

def get_youtube_transcript(video_id: str) -> str:

    try:

        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        text = " ".join(
            snippet.text
            for snippet in transcript
        )

        return text

    except Exception as e:

        return (
            f"Unable to retrieve transcript for video "
            f"{video_id}.\nError: {str(e)}"
        )


# ============================================================
# 4. YOUTUBE SEARCH TOOL
# ============================================================

class YouTubeSearchTool(BaseTool):

    name: str = "YouTube Search"

    description: str = """
Search YouTube for videos.

Input:
A search query such as:
"AI vs ML vs DL vs Data Science"

Returns:
Video ID, title, channel, publication date,
URL and description.
"""

    def _run(self, query: str) -> str:

        return search_youtube(query)


# ============================================================
# 5. YOUTUBE TRANSCRIPT TOOL
# ============================================================

class YouTubeTranscriptTool(BaseTool):

    name: str = "YouTube Transcript"

    description: str = """
Retrieve the transcript of a YouTube video.

Input:
The YouTube Video ID.

Example:
vNc2z2u_nh0

Returns:
The complete transcript text.
"""

    def _run(self, video_id: str) -> str:

        return get_youtube_transcript(video_id)


# ============================================================
# 6. CREATE TOOLS
# ============================================================

youtube_search_tool = YouTubeSearchTool()

youtube_transcript_tool = YouTubeTranscriptTool()


# ============================================================
# 7. GROQ LLM
# ============================================================

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)


# ============================================================
# 8. CREATE AGENT
# ============================================================

research_agent = Agent(

    role="Senior YouTube Researcher",

    goal="""
Research technical topics using YouTube videos and transcripts.
Create accurate, structured research reports.
""",

    backstory="""
You are an experienced technical researcher.

You specialize in:
Artificial Intelligence
Machine Learning
Deep Learning
Data Science
Generative AI

You carefully inspect YouTube search results,
select a relevant video, retrieve its transcript,
and produce a structured research report.
""",

    llm=llm,

    tools=[
        youtube_search_tool,
        youtube_transcript_tool
    ],

    verbose=True,

    allow_delegation=False,

    max_iter=5,

    max_retry_limit=2,
)


# ============================================================
# 9. CREATE TASK
# ============================================================

research_task = Task(

    description="""
Research the following topic:

AI vs ML vs DL vs Data Science

Follow this workflow:

STEP 1
Use the YouTube Search tool.

Search for:

AI vs ML vs DL vs Data Science explanation

STEP 2
Review the returned YouTube videos.

STEP 3
Select ONE highly relevant video.

Prefer a video that directly explains:
AI
Machine Learning
Deep Learning
Data Science

STEP 4
Extract the Video ID from the selected result.

STEP 5
Use the YouTube Transcript tool with that Video ID.

STEP 6
Analyze the transcript.

STEP 7
Create a detailed research report.

The report must contain:

1. Video Title

2. Channel

3. Video URL

4. Main Concepts

5. Important Definitions

6. AI explanation

7. Machine Learning explanation

8. Deep Learning explanation

9. Data Science explanation

10. Relationship between AI, ML, DL and Data Science

11. Examples

12. Important conclusions

IMPORTANT:

Use only ONE YouTube search.

Select ONE video.

Retrieve ONE transcript.

Do not repeatedly call the YouTube Search tool.

Do not invent information that is not supported
by the selected video transcript.
""",

    expected_output="""
A detailed technical research report in Markdown format.

Include:

# Video Information

# Main Concepts

# Important Definitions

# AI

# Machine Learning

# Deep Learning

# Data Science

# Relationship Between AI, ML, DL and Data Science

# Examples

# Key Takeaways

# Conclusion
""",

    agent=research_agent,
)


# ============================================================
# 10. CREATE CREW
# ============================================================

crew = Crew(

    agents=[research_agent],

    tasks=[research_task],

    process=Process.sequential,

    verbose=True,
)


# ============================================================
# 11. RUN CREW
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("STARTING YOUTUBE RESEARCH CREW")
    print("=" * 70)
    print("\n")

    result = crew.kickoff()

    print("\n")
    print("=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    print("\n")

    print(result)


    # ========================================================
    # 12. SAVE FINAL REPORT TO FILE
    # ========================================================

    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "youtube_blogpost.md"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(str(result))


    print("\n")
    print("=" * 70)
    print("RESEARCH REPORT SAVED")
    print("=" * 70)
    print(f"\nFile: {os.path.abspath(output_file)}")