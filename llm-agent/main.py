"""Main CLI interface for the LLM agent."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent import agent_graph
from src.configs import Configuration
from src.prompts import USER_GUIDANCE_PROMPT


# Load environment variables
load_dotenv()


async def run_agent_interactive():
    """Run the agent in interactive mode."""
    print("=" * 80)
    print("🤖 LLM Agent for Agricultural Data Analysis")
    print("=" * 80)
    print("\n" + USER_GUIDANCE_PROMPT)
    print("\n" + "=" * 80)
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("=" * 80 + "\n")

    # Get API key from environment
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_api_key and not openai_api_key:
        print("⚠️  Warning: No API keys found in environment variables.")
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file.")
        print("Create a .env file based on .env.example\n")
        return

    # Configuration
    config = {
        "configurable": {
            "anthropic_api_key": anthropic_api_key,
            "openai_api_key": openai_api_key,
            "agent_llm": {
                "model": "claude-3-5-sonnet-20241022",  # or "gpt-4o"
                "temperature": 0.7,
                "max_tokens": 4000,
            },
            "verbose": True,
            "stream_output": True,
        }
    }

    # Conversation history
    conversation_history = []

    while True:
        # Get user input
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))

        # Prepare state
        state = {
            "messages": conversation_history,
        }

        print("\n🤖 Agent: ", end="", flush=True)

        try:
            # Stream the response
            full_response = ""
            async for event in agent_graph.astream(state, config, stream_mode="values"):
                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]

                    # If it's an AI message (not a tool call)
                    if hasattr(last_message, "content") and isinstance(last_message.content, str):
                        # Only print new content
                        if last_message.content != full_response:
                            new_content = last_message.content[len(full_response):]
                            print(new_content, end="", flush=True)
                            full_response = last_message.content

                    # If it's a tool call, show what tool is being called
                    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            print(f"\n  [Calling tool: {tool_name}]", flush=True)

            print()  # New line after response

            # Update conversation history with all new messages
            conversation_history = event.get("messages", conversation_history)

        except Exception as e:
            print(f"\n\n❌ Error: {str(e)}")
            print("Please try again or check your API keys and configuration.\n")
            # Don't add the failed message to history
            conversation_history = conversation_history[:-1]


async def run_agent_single(query: str):
    """Run the agent with a single query."""
    # Get API key from environment
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_api_key and not openai_api_key:
        print("Error: No API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        return

    # Configuration
    config = {
        "configurable": {
            "anthropic_api_key": anthropic_api_key,
            "openai_api_key": openai_api_key,
            "agent_llm": {
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7,
                "max_tokens": 4000,
            },
        }
    }

    # Prepare state
    state = {
        "messages": [HumanMessage(content=query)],
    }

    print(f"\n🧑 Query: {query}\n")
    print("🤖 Agent: ", end="", flush=True)

    # Stream the response
    full_response = ""
    async for event in agent_graph.astream(state, config, stream_mode="values"):
        messages = event.get("messages", [])
        if messages:
            last_message = messages[-1]

            if hasattr(last_message, "content") and isinstance(last_message.content, str):
                if last_message.content != full_response:
                    new_content = last_message.content[len(full_response):]
                    print(new_content, end="", flush=True)
                    full_response = last_message.content

            elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    print(f"\n  [Calling tool: {tool_name}]", flush=True)

    print("\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM Agent for Agricultural Data Analysis"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run (non-interactive mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM model to use (default: claude-3-5-sonnet-20241022)",
    )

    args = parser.parse_args()

    if args.query:
        # Single query mode
        asyncio.run(run_agent_single(args.query))
    else:
        # Interactive mode
        asyncio.run(run_agent_interactive())


if __name__ == "__main__":
    main()
