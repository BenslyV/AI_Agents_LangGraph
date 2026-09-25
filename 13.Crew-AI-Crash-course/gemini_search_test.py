import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load .env
load_dotenv()


# Check API key
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set")


# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# Google Search tool
google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)


# Configuration
config = types.GenerateContentConfig(
    tools=[google_search_tool]
)


# Ask a current-information question
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="What are the latest developments in Generative AI?",
    config=config
)


# Print response
print("\n===== GEMINI GOOGLE SEARCH RESULT =====\n")

print(response.text)