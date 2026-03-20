# Wiki Agent

CLI assistant that answers questions using **Wikipedia**, wired through **MCP** (stdio) and **LangGraph** + **OpenAI**. The client spawns the MCP server as a subprocess; you only run the client.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or another venv + installer
- **OpenAI API key** (model defaults to `gpt-4o-mini` in code)

## Setup

```bash
git clone <your-fork-or-remote> wiki-agent
cd wiki-agent
uv sync
```

Create `.env` in the project root (same folder as `pyproject.toml`):

```env
OPENAI_API_KEY=sk-...
# Optional: max LangGraph steps per user message (default 25)
# GRAPH_RECURSION_LIMIT=25
```

**Run from the repo root** so `mcp_server.py` can resolve files like `suggested_titles.txt` for MCP resources.

## Usage

Start the interactive client (this also starts `mcp_server.py` over stdio):

```bash
uv run mcp_client.py
```

### Plain chat

Type a question; the model picks MCP **tools** (Wikipedia search, sections, section body) as needed.

```text
You: Who is Kahlil Gibran?
You: Summarize the "Career" section for Python (programming language)
```

Exit: `exit`, `quit`, or `q`.

### Slash commands

| Command | What it does |
|--------|----------------|
| `/prompts` | List MCP **prompts** and their argument names. |
| `/prompt <name> <args...>` | Fetch prompt text from the server and run it through the **same** LangGraph agent (quoted args if spaces). |
| `/resources` | List MCP **resources**. |
| `/resource <name-or-index>` | Print resource body (e.g. `1` for first item). |

**Prompt example** (arguments must match what `/prompts` shows; use **one shell-style token per argument**, quotes allowed):

```text
You: /prompts
You: /prompt highlight_sections_prompt "Kahlil Gibran"
```

**Resource example** (`suggested_titles` comes from `suggested_titles.txt` in the cwd):

```text
You: /resources
You: /resource 1
```

## Architecture (short)

| Piece | Role |
|-------|------|
| `mcp_server.py` | FastMCP server: Wikipedia **tools**, **prompts**, **resources** (stdio). |
| `mcp_client.py` | MCP **client**, `load_mcp_tools`, LangGraph `StateGraph` (chat ↔ `ToolNode`), `MemorySaver` checkpointer. |

Server launch is configured in code as roughly: `uv run mcp_server.py` (see `StdioServerParameters` in `mcp_client.py`).

## MCP surface (server)

**Tools**

- `fetch_wikipedia_info` — search + summary + URL  
- `list_wikipedia_sections` — section titles (TOC via MediaWiki `parse` + `page=` — avoids empty TOCs from title-only API quirks)  
- `get_section_content` — plain text for one section  

**Prompts**

- `highlight_sections_prompt(topic)` — instructs the model to prioritize important sections (still relies on tools for real section lists)

**Resources**

- `file://suggested_titles` — lines from `suggested_titles.txt`

## Configuration notes

- **`GRAPH_RECURSION_LIMIT`** — Caps LangGraph steps **per `ainvoke`** (one user message / one `/prompt` run), not “per tool call” and not cumulative across the whole REPL session. If the agent keeps calling tools, you’ll see `CallToolRequest` spam until the limit trips.
- **Interrupted tool turns** — If a run stops at the step limit mid–tool-call, the client repairs message history before the next model call so OpenAI doesn’t reject the thread.

## Troubleshooting

- **`FileNotFoundError` for subprocess** — `StdioServerParameters` must use a real executable + `args` (e.g. `command="uv"`, `args=["run", "mcp_server.py"]`), not a single `"uv run ..."` string.
- **Empty Wikipedia sections from older code paths** — Server uses direct API calls for section titles where the `wikipedia` PyPI package’s `titles=` parse returns an empty list.
- **Resource file missing** — Run `uv run mcp_client.py` from the directory that contains `suggested_titles.txt`.

## Dev

```bash
uv run ruff check .
```
