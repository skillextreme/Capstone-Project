"""Configuration for the LLM agent."""

from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """Configuration for the LLM agent."""

    # LLM configuration
    agent_llm: dict = Field(
        default={
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "max_tokens": 4000,
        },
        description="Configuration for the main agent LLM"
    )

    # Tool execution settings
    max_tool_iterations: int = Field(
        default=15,
        description="Maximum number of tool calls before forcing completion"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed tool calls"
    )

    # Output settings
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging"
    )

    stream_output: bool = Field(
        default=True,
        description="Stream LLM responses to user"
    )

    @staticmethod
    def from_runnable_config(config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create Configuration from RunnableConfig."""
        if config is None:
            return Configuration()

        configurable = config.get("configurable", {})
        if not configurable:
            return Configuration()

        return Configuration(**configurable)
