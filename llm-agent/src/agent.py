"""Main LLM agent for the pipeline."""

from typing import Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

from src.states import AgentState, AgentInputState
from src.tools import get_all_tools
from src.prompts import SYSTEM_PROMPT
from src.configs import Configuration


# Initialize configurable model
configurable_model = init_chat_model(
    configurable_fields=("model", "temperature", "max_tokens", "api_key"),
)


def get_api_key_for_model(model: str, config: RunnableConfig) -> str:
    """Get the appropriate API key for the model."""
    if "claude" in model or "anthropic" in model:
        return config.get("configurable", {}).get("anthropic_api_key", "")
    elif "gpt" in model or "openai" in model:
        return config.get("configurable", {}).get("openai_api_key", "")
    return ""


async def agent_node(state: AgentState, config: RunnableConfig):
    """Main agent node that decides which tools to call."""
    configurable = Configuration.from_runnable_config(config)

    # Get tools
    tools = get_all_tools()

    # Configure the model with tools
    model_config = {
        **configurable.agent_llm,
        "api_key": get_api_key_for_model(
            configurable.agent_llm.get("model", ""),
            config
        ),
    }

    model = configurable_model.bind_tools(tools).with_config(model_config)

    # Get messages
    messages = state.get("messages", [])

    # Add system prompt if this is the first message
    if len(messages) == 0 or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Invoke the model
    response = await model.ainvoke(messages)

    return {"messages": [response]}


async def route_after_agent(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """Route after the agent node - either call tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]

    # If the agent called tools, go to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return END


def create_agent_graph():
    """Create the LangGraph agent."""

    # Create the graph
    graph_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)

    # Add nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(get_all_tools()))

    # Set entry point
    graph_builder.set_entry_point("agent")

    # Add edges
    graph_builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    # After tools, always go back to agent
    graph_builder.add_edge("tools", "agent")

    # Compile the graph
    return graph_builder.compile()


# Create the compiled graph
agent_graph = create_agent_graph()
