from crewai.tools import tool
from ddgs import DDGS


@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web for information about a given topic."""

    results = DDGS().text(
        query,
        max_results=5
    )

    if not results:
        return "No search results found."

    output = []

    for result in results:
        output.append(
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('href', '')}\n"
            f"Description: {result.get('body', '')}"
        )

    return "\n\n".join(output)