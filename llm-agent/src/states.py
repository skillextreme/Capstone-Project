"""State definitions for the LLM agent."""

from typing import List, Dict, Any, Optional
from typing_extensions import Annotated
import operator
from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState


def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        if current_value is None:
            return new_value
        if isinstance(current_value, list) and isinstance(new_value, list):
            return current_value + new_value
        return new_value


#### Agent State Definitions ####
class AgentState(MessagesState):
    """Main state for the LLM agent managing the pipeline."""
    # Pipeline execution state
    summaries_generated: bool = False
    tasks_suggested: bool = False
    current_task_id: Optional[str] = None
    plan_created: bool = False
    analysis_executed: bool = False

    # File tracking
    available_files: Annotated[List[str], override_reducer] = []
    summary_files: Annotated[List[str], override_reducer] = []

    # Task tracking
    suggested_tasks: Annotated[List[Dict[str, Any]], override_reducer] = []

    # Results tracking
    analysis_results: Annotated[Dict[str, Any], override_reducer] = {}

    # Error tracking
    last_error: Optional[str] = None


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


#### Tool Input Schema Definitions ####

class ListFilesInput(BaseModel):
    """Input for listing available data files."""
    pattern: str = Field(
        default="*",
        description="Glob pattern to filter files (e.g., '*.csv', 'crop*')"
    )


class SummarizeDataInput(BaseModel):
    """Input for Stage 1: Data Summarization."""
    file_names: Optional[List[str]] = Field(
        default=None,
        description="Specific files to summarize. If None, summarizes all files in data/raw/"
    )
    force_rerun: bool = Field(
        default=False,
        description="Force re-summarization even if summaries exist"
    )


class SuggestTasksInput(BaseModel):
    """Input for Stage 2: Task Suggestion."""
    force_rerun: bool = Field(
        default=False,
        description="Force re-generation of task suggestions"
    )


class PlanAnalysisInput(BaseModel):
    """Input for Stage 3: Analysis Planning."""
    task_id: str = Field(
        ...,
        description="Task ID to plan for (e.g., 'T1', 'T2')"
    )
    force_rerun: bool = Field(
        default=False,
        description="Force re-planning even if plan exists"
    )


class ExecuteAnalysisInput(BaseModel):
    """Input for Stage 4: Analysis Execution."""
    task_id: str = Field(
        ...,
        description="Task ID to execute (e.g., 'T1', 'T2')"
    )
    force_rerun: bool = Field(
        default=False,
        description="Force re-execution even if results exist"
    )


class ViewSummaryInput(BaseModel):
    """Input for viewing a file summary."""
    file_name: str = Field(
        ...,
        description="Name of the file to view summary for (e.g., 'crop_yield.csv')"
    )


class ViewTasksInput(BaseModel):
    """Input for viewing suggested tasks."""
    task_id: Optional[str] = Field(
        default=None,
        description="Specific task ID to view. If None, shows all tasks."
    )


class ViewResultsInput(BaseModel):
    """Input for viewing analysis results."""
    task_id: str = Field(
        ...,
        description="Task ID to view results for (e.g., 'T1', 'T2')"
    )
    result_type: str = Field(
        default="summary",
        description="Type of result to view: 'summary', 'metrics', 'predictions', 'model_card'"
    )
