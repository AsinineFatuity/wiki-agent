from typing import Annotated, List 
from typing_extensions import TypedDict
from decouple import config
from mcp import ClientSession, StdioServerParameters
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

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


    def chat_node(state: LangGraphState) -> LangGraphState:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state 
    
    #Build Langraph with tool routing 
    graph=StateGraph(LangGraphState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
    graph.add_edge("tool_node", "chat_node")
    return graph.compile(checkpointer=MemorySaver())