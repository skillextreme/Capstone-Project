# LLM Agent Architecture

This document explains the architecture and design decisions for the LLM-based agentic system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (CLI / Interactive)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Natural Language
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Agent                          │
│                     (ReAct Pattern)                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐         ┌──────────────┐                     │
│  │ Agent Node   │────────→│  Tool Node   │                     │
│  │ (LLM thinks) │←────────│ (Executes)   │                     │
│  └──────────────┘         └──────────────┘                     │
│         │                        │                              │
│         │                        │                              │
│    Decides which             Calls selected                     │
│    tool to call              tool function                      │
└─────────┼────────────────────────┼──────────────────────────────┘
          │                        │
          │ Tool Selection         │ Tool Execution
          ↓                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Tool Layer                              │
│                    (LangChain Tools)                            │
├─────────────────────────────────────────────────────────────────┤
│  Data Exploration          │    Pipeline Stages                 │
│  ├─ list_data_files        │    ├─ summarize_data (Stage 1)    │
│  ├─ view_summary           │    ├─ suggest_tasks (Stage 2)     │
│  ├─ view_tasks             │    ├─ plan_analysis (Stage 3)     │
│  └─ view_results           │    └─ execute_analysis (Stage 4)  │
└─────────┬────────────────────────┬──────────────────────────────┘
          │                        │
          │                        │
          ↓                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic Pipeline                             │
│                  (Existing 4-Stage System)                      │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Summarizer     │  Stage 2: Task Suggester            │
│  Stage 3: Planner        │  Stage 4: Executor                  │
└─────────────────────────────────────────────────────────────────┘
          │                        │
          ↓                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Data & Results                              │
│  data/raw/  →  data/summaries/  →  data/tasks.json             │
│  data/intermediate/  →  data/outputs/                           │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interface Layer

**File:** `main.py`

**Responsibilities:**
- Accept user input (interactive or single query)
- Display agent responses
- Show tool execution progress
- Handle streaming responses

**Key Functions:**
- `run_agent_interactive()` - Start conversational interface
- `run_agent_single(query)` - Execute single query
- `main()` - Entry point with argument parsing

### 2. LangGraph Agent

**File:** `src/agent.py`

**Architecture:** State machine with nodes and edges

```python
StateGraph:
  Nodes:
    - agent_node: LLM reasoning and tool selection
    - tools: Tool execution

  Edges:
    - START → agent_node
    - agent_node → [tools | END] (conditional)
    - tools → agent_node (loop back)
```

**Agent Node (`agent_node`):**
- Receives current state (messages, context)
- LLM analyzes user request and conversation history
- Decides: call a tool OR respond directly
- Returns: AI message (with or without tool calls)

**Tool Node (`tools`):**
- Executes the selected tool
- Returns tool result as message
- Loops back to agent for interpretation

**Routing Logic (`route_after_agent`):**
- If agent called tools → go to `tools` node
- If no tool calls → END (respond to user)

**State:** `AgentState` (extends `MessagesState`)
- `messages`: Conversation history
- `summaries_generated`, `tasks_suggested`, etc.: Pipeline state
- `available_files`, `summary_files`: File tracking
- `suggested_tasks`: Task tracking
- `analysis_results`: Results cache
- `last_error`: Error handling

### 3. Tool Layer

**File:** `src/tools.py`

All tools are decorated with `@tool` from LangChain, which:
- Automatically generates tool descriptions for the LLM
- Handles input validation via Pydantic schemas
- Provides error handling

**Tool Categories:**

**A. Data Exploration Tools**
- Purpose: Inspect data and results without modifying state
- Examples: `list_data_files`, `view_summary`, `view_tasks`, `view_results`

**B. Pipeline Stage Tools**
- Purpose: Execute pipeline stages
- Examples: `summarize_data`, `suggest_tasks`, `plan_analysis`, `execute_analysis`
- Side effects: Create files, modify pipeline state

**Tool Design Pattern:**
```python
@tool("tool_name")
def tool_function(param1: Type1, param2: Type2) -> str:
    """Tool description for LLM.

    Detailed explanation of what the tool does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Human-readable result string
    """
    try:
        # 1. Validate inputs
        # 2. Execute pipeline stage
        # 3. Format response
        # 4. Return user-friendly message
    except Exception as e:
        return f"Error: {str(e)}"
```

**Why return strings?**
- LLM needs human-readable results to interpret
- Structured data (JSON) is harder for LLM to reason about
- Formatted strings include context and guidance

### 4. State Management

**File:** `src/states.py`

**AgentState:**
```python
class AgentState(MessagesState):
    # Pipeline execution tracking
    summaries_generated: bool
    tasks_suggested: bool
    current_task_id: Optional[str]
    plan_created: bool
    analysis_executed: bool

    # Data tracking
    available_files: List[str]
    summary_files: List[str]
    suggested_tasks: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]

    # Error handling
    last_error: Optional[str]
```

**State Reducers:**
- `override_reducer`: Allows replacing list values instead of appending
- Used for `available_files`, `summary_files`, etc.

**Tool Input Schemas:**
- Each tool has a Pydantic schema for inputs
- Enables LLM to correctly format tool calls
- Provides validation and type checking

### 5. Configuration

**File:** `src/configs.py`

**Configuration Options:**
```python
class Configuration(BaseModel):
    agent_llm: dict  # Model, temperature, max_tokens
    max_tool_iterations: int  # Prevent infinite loops
    max_retries: int  # Retry failed tool calls
    verbose: bool  # Logging
    stream_output: bool  # Streaming responses
```

**Runtime Configuration:**
- Passed via `RunnableConfig` to agent
- Includes API keys (not stored in code)
- Can be overridden per invocation

### 6. Prompts

**File:** `src/prompts.py`

**SYSTEM_PROMPT:**
- Defines agent's role and capabilities
- Lists all available tools and their purposes
- Provides guidelines for agent behavior
- Sets conversational tone

**Design Principles:**
1. **Clear tool descriptions** - Help LLM select correct tool
2. **Workflow guidance** - Explain stage order (1→2→3→4)
3. **Error handling** - How to handle tool failures
4. **User experience** - Be helpful and proactive

## Data Flow

### Typical Interaction Flow

```
1. User: "Analyze my data"
   ↓
2. Agent Node (LLM):
   - Reads SYSTEM_PROMPT
   - Sees available tools
   - Decides: "I should first check what data files exist"
   - Calls: list_data_files()
   ↓
3. Tool Node:
   - Executes list_data_files()
   - Returns: "Found 3 files: crop_yield.csv, ..."
   ↓
4. Agent Node (LLM):
   - Receives tool result
   - Decides: "Now I should summarize these files"
   - Calls: summarize_data()
   ↓
5. Tool Node:
   - Executes summarize_data()
   - Calls Stage 1 Summarizer
   - Returns: "✓ Successfully summarized 3 files..."
   ↓
6. Agent Node (LLM):
   - Receives tool result
   - Decides: "I have enough info to respond"
   - Returns: AI message without tool calls
   ↓
7. User sees: "I've analyzed your data. You have 3 files with... Next, I can suggest analysis tasks."
```

### Message Flow in State

```python
# Initial state
{
    "messages": [
        SystemMessage("You are an expert..."),
        HumanMessage("Analyze my data")
    ]
}

# After agent calls tool
{
    "messages": [
        SystemMessage(...),
        HumanMessage("Analyze my data"),
        AIMessage(tool_calls=[{name: "list_data_files", ...}])
    ]
}

# After tool execution
{
    "messages": [
        SystemMessage(...),
        HumanMessage("Analyze my data"),
        AIMessage(tool_calls=[...]),
        ToolMessage(content="Found 3 files: ...", tool_call_id=...)
    ]
}

# Agent sees tool result, calls another tool
{
    "messages": [
        ...,
        ToolMessage(...),
        AIMessage(tool_calls=[{name: "summarize_data", ...}])
    ]
}

# Final response (no tool calls)
{
    "messages": [
        ...,
        AIMessage(content="I've analyzed your data...")
    ]
}
```

## Design Decisions

### Why LangGraph?

**Alternatives considered:**
- Direct OpenAI/Anthropic function calling
- LangChain's AgentExecutor
- Custom loop implementation

**Why LangGraph won:**
1. **State persistence** - Built-in state management
2. **Flexibility** - Full control over agent flow
3. **Debuggability** - Clear node/edge visualization
4. **Streaming** - Native streaming support
5. **Production-ready** - Designed for deployment

### Why ReAct Pattern?

**ReAct = Reasoning + Acting**

The agent alternates between:
- **Reasoning**: Thinking about what to do next
- **Acting**: Calling tools to gather information

**Benefits:**
- More reliable than pure chain-of-thought
- Self-correcting (can retry if tool fails)
- Transparent (we see the agent's reasoning)

### Why String Responses from Tools?

**Alternative:** Return structured data (JSON, dicts)

**Why strings:**
1. LLM interprets natural language better than JSON
2. Can include guidance ("Next step: ...")
3. User-friendly (can show directly to user)
4. Flexible formatting

**Trade-off:** Slightly more verbose, but much better UX

### Error Handling Strategy

**Principle:** Graceful degradation

1. **Tool errors** → Return error message as string
2. **LLM errors** → Catch and retry (up to max_retries)
3. **User errors** → Agent guides user to fix (via prompts)

**Example:**
```python
try:
    result = execute_stage()
    return format_success(result)
except FileNotFoundError:
    return "Error: Data files not found. Please run 'summarize_data' first."
except Exception as e:
    return f"Error: {str(e)}"
```

## Performance Considerations

### Token Usage

**Strategies to minimize tokens:**
1. **Tool responses** - Concise but informative
2. **System prompt** - Clear but not overly long
3. **State** - Only track essential information
4. **Streaming** - Show progress without bloating messages

**Typical conversation:**
- Initial prompt: ~1,000 tokens
- Per turn: ~500-1,000 tokens (query + response)
- Tool results: ~200-500 tokens each

### Latency

**Sources of latency:**
1. **LLM inference** - 2-5 seconds per agent node
2. **Tool execution** - Varies by stage:
   - list_data_files: <1 second
   - summarize_data: 5-30 seconds (depends on data size)
   - execute_analysis: 30-300 seconds (model training)

**Optimization:**
- Streaming responses (user sees partial output)
- Parallel tool calls (where possible)
- Caching (planned: cache summaries, tasks)

## Security Considerations

### API Key Management

- Loaded from environment variables (.env)
- Never stored in code or state
- Passed via RunnableConfig (runtime only)

### Sandboxing

**Current state:** Tools access filesystem directly

**Risks:**
- Tool could read/write arbitrary files
- Malicious prompts could cause unintended operations

**Mitigations:**
1. Tools only access specific directories (`data/`)
2. No `rm`, `delete`, or destructive operations
3. All writes go to designated output directories

**Future improvements:**
- Docker sandboxing (like local-deep-research)
- File access whitelist
- User confirmation for destructive operations

## Extensibility

### Adding New Tools

**Steps:**
1. Define tool function with `@tool` decorator
2. Add Pydantic schema for inputs
3. Implement tool logic
4. Add to `get_all_tools()` list
5. Update SYSTEM_PROMPT with tool description

**Example:**
```python
@tool("my_new_tool")
def my_new_tool(param: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return "Result"

# Add to get_all_tools()
def get_all_tools():
    return [
        ...,
        my_new_tool,
    ]
```

### Supporting New LLM Providers

**Steps:**
1. Add API key to Configuration
2. Update `get_api_key_for_model()` in agent.py
3. Add model name to docs

**Example (for Gemini):**
```python
def get_api_key_for_model(model: str, config: RunnableConfig) -> str:
    if "claude" in model:
        return config.get("configurable", {}).get("anthropic_api_key", "")
    elif "gpt" in model:
        return config.get("configurable", {}).get("openai_api_key", "")
    elif "gemini" in model:
        return config.get("configurable", {}).get("google_api_key", "")
    return ""
```

### Custom Agent Behaviors

**Modify:** `src/prompts.py` - SYSTEM_PROMPT

**Examples:**
- More technical/less conversational tone
- Domain-specific knowledge (beyond agriculture)
- Multi-lingual support
- Persona (e.g., "You are a data scientist...")

## Testing Strategy

### Unit Tests (Planned)

**Tool tests:**
```python
def test_list_data_files():
    result = list_data_files(pattern="*.csv")
    assert "crop_yield.csv" in result

def test_summarize_data():
    result = summarize_data(file_names=["test.csv"])
    assert "Successfully summarized" in result
```

**Agent tests:**
```python
async def test_agent_workflow():
    state = {"messages": [HumanMessage("List files")]}
    result = await agent_graph.ainvoke(state, config)
    assert len(result["messages"]) > 1
```

### Integration Tests (Planned)

**End-to-end workflow:**
```python
async def test_full_pipeline():
    # Simulate user conversation
    queries = [
        "List my data files",
        "Summarize the data",
        "Suggest tasks",
        "Run task T1"
    ]

    for query in queries:
        result = await agent_graph.ainvoke(...)
        assert no errors
```

## Monitoring & Debugging

### LangSmith Integration

**Setup:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=llm-agent
```

**Benefits:**
- Trace every LLM call
- See tool executions
- Debug failures
- Monitor token usage
- Analyze conversation patterns

### Logging

**Current:** Print statements in main.py

**Future:**
- Structured logging (JSON)
- Log levels (DEBUG, INFO, ERROR)
- Log to file for analysis

## Future Enhancements

**Short term:**
1. Add conversation persistence (save/load)
2. Implement caching for expensive operations
3. Better error messages with suggestions
4. Add confirmation for destructive operations

**Medium term:**
1. Web UI (Streamlit/Gradio)
2. Visualization preview in terminal
3. Multi-user support
4. API server mode

**Long term:**
1. Multi-agent collaboration (supervisor pattern)
2. Autonomous mode (agent runs full pipeline)
3. Human-in-the-loop approvals
4. Integration with data cataloging systems

---

**Questions or suggestions?** Open an issue or contribute!
