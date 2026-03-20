import wikipedia
import requests
from mcp.server.fastmcp import FastMCP
from datetime import datetime
from pathlib import Path

mcp = FastMCP("WikipediaSearch")

# MediaWiki 1.46+: parse&prop=sections with `titles=` often returns [].
# Using `page=` (resolved article title) returns the real TOC.
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "wiki-agent-mcp/1.0 (https://github.com/wiki-agent; MCP Wikipedia tools)"


def _section_titles_for_page(resolved_title: str) -> list[str]:
    r = requests.get(
        WIKI_API,
        params={
            "action": "parse",
            "page": resolved_title,
            "prop": "sections",
            "format": "json",
        },
        headers={"User-Agent": WIKI_UA},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", str(data["error"])))
    return [s["line"] for s in data["parse"]["sections"]]


@mcp.tool()
def fetch_wikipedia_info(query: str) -> dict:
    """
    Search Wikipedia for a topic and return title, summary, and URL of the best match.
    """
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return {"error": "No results found for your query."}

        best_match = search_results[0]
        page = wikipedia.page(best_match)

        return {"title": page.title, "summary": page.summary, "url": page.url}

    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }

    except wikipedia.PageError:
        return {"error": "No Wikipedia page could be loaded for this query."}


@mcp.tool()
def list_wikipedia_sections(topic: str) -> dict:
    """
    Return a list of section titles from the Wikipedia page of a given topic.
    """
    try:
        page = wikipedia.page(topic)
        sections = _section_titles_for_page(page.title)
        return {"sections": sections}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_section_content(topic: str, section_title: str) -> dict:
    """
    Return the content of a specific section in a Wikipedia article.
    """
    try:
        page = wikipedia.page(topic)
        content = page.section(section_title)
        if content:
            return {"content": content}
        else:
            return {
                "error": f"Section '{section_title}' not found in article '{topic}'."
            }
    except Exception as e:
        return {"error": str(e)}


@mcp.prompt()
def highlight_sections_prompt(topic: str) -> str:
    """
    Identifies the most important sections from a Wikipedia article on the given topic
    """
    return f"""
    The user is exploring the Wikipedia article on "{topic}"

    Given the list of section title from  the article, choose the 3-5 most important or interesting sections that are likely to help someone learn about the topic.

    Return a bullet list of these section title, along with 1-line explanations of why each one matters
    """


@mcp.resource("file://suggested_titles")
def suggested_titles() -> list[str]:
    """
    Return a list of suggested titles for Wikipedia articles from a local file
    """
    try:
        path = Path("suggested_titles.txt")
        if not path.exists():
            return ["File not found"]
        return path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"Error reading file: {str(e)}"]


# Run the MCP server
if __name__ == "__main__":
    print(
        f"{__name__}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting MCP Wikipedia Server..."
    )
    mcp.run(transport="stdio")
