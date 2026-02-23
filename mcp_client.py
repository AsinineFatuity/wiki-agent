from typing import Annotated, List 
from typing_extensions import TypedDict
from decouple import config
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters

server_params = StdioServerParameters(command="uv run mcp_server.py")
client = ClientSession(server_params)

# Langgraph state definition
class LangGraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

async def create_graph(session):
    # Load tools from MCP server
    tools  = await load_mcp_tools(session)
    # LLM Configuration (system prompt can be added later)
    api_key = config("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)
    # Prompt template with user/assistant chat only 
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to search Wikipedia"),
        MessagesPlaceholder(variable_name="messages")
    ])
    chat_llm = prompt_template | llm_with_tools