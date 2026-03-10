import asyncio
import logging
import shlex
from typing import Annotated, List
from typing_extensions import TypedDict
from decouple import config
from mcp import ClientSession, StdioServerParameters
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="uv", args=["run", "mcp_server.py"])


# Langgraph state definition
class LangGraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    # Load tools from MCP server
    tools = await load_mcp_tools(session)
    # LLM Configuration (system prompt can be added later)
    api_key = config("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)
    # Prompt template with user/assistant chat only
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that uses tools to search Wikipedia",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: LangGraphState) -> LangGraphState:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Build Langraph with tool routing
    graph = StateGraph(LangGraphState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges(
        "chat_node", tools_condition, {"tools": "tool_node", "__end__": END}
    )
    graph.add_edge("tool_node", "chat_node")
    return graph.compile(checkpointer=MemorySaver())

async def list_prompts(session):
    prompt_response = await session.list_prompts()

    if not prompt_response or not prompt_response.prompts:
        print("No prompts found on the server.")
        return

    print("\nAvailable Prompts and Argument Structure:")
    for p in prompt_response.prompts:
        print(f"\nPrompt: {p.name}")
        if p.arguments:
            for arg in p.arguments:
                print(f"  - {arg.name}")
        else:
            print("  - No arguments required.")
    print("\nUse: /prompt <prompt_name> \"arg1\" \"arg2\" ...")

async def handle_prompt(session, tools, command, agent):
    parts = shlex.split(command.strip())
    if len(parts) < 2:
        print("Usage: /prompt <name> \"args>\"")
        return

    prompt_name = parts[1]
    args = parts[2:]

    try:
        # Get available prompts
        prompt_def = await session.list_prompts()
        match = next((p for p in prompt_def.prompts if p.name == prompt_name), None)
        if not match:
            print(f"Prompt '{prompt_name}' not found.")
            return

        # Check arg count
        if len(args) != len(match.arguments):
            expected = ", ".join([a.name for a in match.arguments])
            print(f"Expected {len(match.arguments)} arguments: {expected}")
            return

        # Build argument dict
        arg_values = {arg.name: val for arg, val in zip(match.arguments, args)}
        response = await session.get_prompt(prompt_name, arg_values)
        prompt_text = response.messages[0].content.text
        
        # Execute the prompt via the agent
        agent_response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt_text)]},
            config={"configurable": {"thread_id": "wiki-session"}}
        )
        print("\n=== Prompt Result ===")
        print(agent_response["messages"][-1].content)

    except Exception:
        logging.error(f"{__name__}: Prompt invocation failed:", exc_info=True)
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print("Wikipedia MCP agent is ready.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config={"configurable": {"thread_id": "wiki-session"}},
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception:
                    logging.error(f"{__name__}:Error:", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
