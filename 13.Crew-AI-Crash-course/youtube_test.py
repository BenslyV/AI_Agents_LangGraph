import os

from dotenv import load_dotenv
from googleapiclient.discovery import build


# --------------------------------------------------
# Load .env
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    raise ValueError(
        "YOUTUBE_API_KEY is not set in the .env file"
    )


# --------------------------------------------------
# Create YouTube API client
# --------------------------------------------------

youtube = build(
    "youtube",
    "v3",
    developerKey=api_key,
    cache_discovery=False
)


# --------------------------------------------------
# Search YouTube
# --------------------------------------------------

request = youtube.search().list(
    part="snippet",
    q="AI Machine Learning",
    type="video",
    maxResults=5
)

response = request.execute()


# --------------------------------------------------
# Display results
# --------------------------------------------------

for item in response["items"]:

    video_id = item["id"]["videoId"]

    title = item["snippet"]["title"]

    channel = item["snippet"]["channelTitle"]

    description = item["snippet"]["description"]

    url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    print()
    print("=" * 70)
    print("Title:", title)
    print("Channel:", channel)
    print("URL:", url)
    print("Description:", description)