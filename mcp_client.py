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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="uv", args=["run", "mcp_server.py"])

# One graph "step" ≈ one node (chat or tools). chat→tools→chat is ~3 steps.
# Hard cap so a bad tool loop can't spin forever (shows as many CallToolRequest lines).
GRAPH_RECURSION_LIMIT = int(config("GRAPH_RECURSION_LIMIT", default=25))


def agent_invoke_config(thread_id: str = "wiki-session") -> dict:
    return {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }


def repair_openai_tool_messages(
    messages: List[AnyMessage],
) -> tuple[List[AnyMessage], bool]:
    """
    OpenAI rejects history if an assistant message has tool_calls but not every id has
    a following ToolMessage (e.g. GraphRecursionError aborts before tool_node runs).
    Insert synthetic ToolMessages for any missing ids so the next completion call works.
    """
    out: List[AnyMessage] = []
    pending: List[str] = []
    changed = False

    def flush() -> None:
        nonlocal changed, pending, out
        if not pending:
            return
        for tid in pending:
            out.append(
                ToolMessage(
                    content=(
                        "[Aborted] This tool call did not finish (e.g. step/recursion limit). "
                        "Do not assume tool output; use prior messages or call tools again if needed."
                    ),
                    tool_call_id=tid,
                )
            )
        changed = True
        pending = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            flush()
            out.append(msg)
            pending = [str(tc["id"]) for tc in msg.tool_calls if tc.get("id")]
        elif isinstance(msg, ToolMessage):
            out.append(msg)
            tid = msg.tool_call_id
            if tid in pending:
                pending = [x for x in pending if x != tid]
        else:
            flush()
            out.append(msg)

    flush()
    return out, changed


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
                "You are a helpful assistant that uses tools to search Wikipedia. "
                "Use tools efficiently: prefer a small number of calls, reuse prior tool "
                "results in the same turn, and stop calling tools once you can answer. "
                "Do not repeatedly call the same tool with the same arguments.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: LangGraphState) -> dict:
        msgs, did_repair = repair_openai_tool_messages(state["messages"])
        ai_msg = chat_llm.invoke({"messages": msgs})
        if did_repair:
            # add_messages is append-only; mid-history inserts require full replace.
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *msgs,
                    ai_msg,
                ]
            }
        return {"messages": ai_msg}

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
    print('\nUse: /prompt <prompt_name> "arg1" "arg2" ...')


async def handle_prompt(session, tools, command, agent):
    parts = shlex.split(command.strip())
    if len(parts) < 2:
        print('Usage: /prompt <name> "args>"')
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
            config=agent_invoke_config(),
        )
        print("\n=== Prompt Result ===")
        print(agent_response["messages"][-1].content)

    except GraphRecursionError:
        print(
            "Stopped: agent hit the step limit "
            f"(GRAPH_RECURSION_LIMIT={GRAPH_RECURSION_LIMIT})."
        )
    except Exception:
        logging.error(f"{__name__}: Prompt invocation failed:", exc_info=True)


async def list_resources(session):
    try:
        response = await session.list_resources()
        if not response or not response.resources:
            print("No resources found on the server.")
            return
        print("\nAvailable Resources:")
        for i, resource in enumerate(response.resources, start=1):
            print(f"{i}. {resource.name}")
        print("\nUse: /resource <name> to view its content")
    except Exception:
        logging.error(f"{__name__}: Resource listing failed:", exc_info=True)


async def handle_resource(session, command):
    parts = shlex.split(command.strip())
    if len(parts) < 2:
        print("Usage: /resource <name>")
        return
    resource_id = parts[1]

    try:
        response = await session.list_resources()
        resources = response.resources
        resource_map = {str(i + 1): r.name for i, r in enumerate(resources)}

        # resolve resource name or index
        resource_name = resource_map.get(resource_id, resource_id)
        match = next((r for r in resources if r.name == resource_name), None)
        if not match:
            print(f"Resource '{resource_id}' not found.")
            return
        # get resource content
        result = await session.read_resource(match.uri)
        for content in result.contents:
            if hasattr(content, "text"):
                print("\n===Resource Text===")
                print(content.text)

    except Exception:
        logging.error(f"{__name__}: Resource handling failed:", exc_info=True)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = await create_graph(session)

            print("Wikipedia MCP agent is ready.")
            print("Type a question or use the following templates:")
            print("  /prompts                - to list available prompts")
            print('  /prompt <name> "args"   - to run a specific prompt')
            print("  /resources               - to list available resources")
            print("  /resource <name>        - to run a specific resource")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                elif user_input.startswith("/prompts"):
                    await list_prompts(session)
                    continue
                elif user_input.startswith("/prompt"):
                    await handle_prompt(session, tools, user_input, agent)
                    continue
                elif user_input.startswith("/resources"):
                    await list_resources(session)
                    continue
                elif user_input.startswith("/resource"):
                    await handle_resource(session, user_input)
                    continue

                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config=agent_invoke_config(),
                    )
                    print("AI:", response["messages"][-1].content)
                except GraphRecursionError:
                    print(
                        "Stopped: agent hit the step limit "
                        f"(GRAPH_RECURSION_LIMIT={GRAPH_RECURSION_LIMIT}). "
                        "Raise the limit in .env or fix the prompt / tool loop."
                    )
                except Exception:
                    logging.error(f"{__name__}: Error:", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
