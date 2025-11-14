# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a capstone project containing three main components:

1. **Agentic Pipeline** (`Capstone-Project/agentic-pipeline/`) - A 4-stage agricultural data analysis pipeline
2. **LLM Agent** (`Capstone-Project/llm-agent/`) - **NEW!** Conversational LLM agent that orchestrates the pipeline through natural language
3. **Local Deep Research** (`Capstone-Project/local-deep-research/`) - A LangGraph-based deep research agent

## Agentic Pipeline

### Purpose
A stateless, verifiable 4-stage pipeline for analyzing agricultural datasets (crop yields, rainfall, inputs) with automated verification at each checkpoint.

### Development Commands

```bash
# Navigate to the pipeline directory
cd "Capstone-Project/agentic-pipeline"

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
pytest tests/test_stage1.py -v  # Test specific stage

# Run the full pipeline
python src/main.py --mode full

# Run individual stages
python src/main.py --mode stage1  # Summarize data
python src/main.py --mode stage2  # Generate task suggestions
python src/main.py --mode stage3 --task-id T1  # Plan for task T1
python src/main.py --mode stage4 --task-id T1  # Execute task T1

# Run with verbose logging
python src/main.py --mode full --verbose

# Run specific modules directly
python -m src.stage1.summarizer --data-dir data/raw --output-dir data/summaries
python -m src.stage2.task_suggester --summaries-dir data/summaries
```

### Architecture

The pipeline follows a stateless design where each stage writes artifacts to disk:

**Stage 1: Summarizer**
- Input: Raw CSV/JSON files in `data/raw/`
- Output: JSON summaries in `data/summaries/`
- Generates schema analysis, statistics, and candidate keys
- Code: `src/stage1/summarizer.py`

**Stage 2: Task Suggester**
- Input: All summaries from Stage 1
- Output: Task proposals in `data/tasks.json`
- Proposes prediction, descriptive, and clustering tasks
- Code: `src/stage2/task_suggester.py`

**Stage 3: Planner**
- Input: Selected task + raw data files
- Output: Merged data in `data/intermediate/<task_id>_merged.parquet` and join plan JSON
- Normalizes keys, merges files, engineers features (lags, rolling statistics)
- Code: `src/stage3/planner.py`, `src/stage3/normalizer.py`

**Stage 4: Executor**
- Input: Task definition + merged data from Stage 3
- Output: Metrics, predictions, model cards, and plots in `data/outputs/`
- Trains models (Ridge, XGBoost), evaluates on holdout set, generates visualizations
- Code: `src/stage4/executor.py`, `src/stage4/models.py`, `src/stage4/visualizer.py`

**Verification System**
- V1 (after Stage 1): Schema validation via `src/verifiers/schema_check.py`
- V2 (after Stage 2): Human task selection
- V3 (after Stage 3): Join cardinality and leakage checks via `src/verifiers/join_check.py`
- V4 (after Stage 4): Metrics validation via `src/verifiers/metrics_check.py`

### Key Design Principles

1. **Statelessness**: Stages don't maintain internal state; all results persist to disk
2. **Verification Gates**: Automated checks after each stage (configurable non-blocking)
3. **Modularity**: Each stage can run independently
4. **Reproducibility**: Same inputs always produce same outputs

### Configuration

- Pipeline settings: `config/pipeline_config.yaml`
- API keys (optional): `.env` file (create from `config/.env.example`)
- Application config: `src/config.py` (loads YAML + env vars)

### Data Flow

```
data/raw/ → Stage 1 → data/summaries/ → Stage 2 → data/tasks.json
                                              ↓
data/outputs/ ← Stage 4 ← data/intermediate/ ← Stage 3
```

### Important Implementation Details

- Time-based train/test splits for prediction tasks (train on ≤2020, test on >2020)
- Feature engineering creates lag features and rolling statistics via pandas groupby
- Join cardinality tracking prevents many-to-many explosions
- Models: Ridge regression and XGBoost (configurable in pipeline_config.yaml)
- Parquet format for intermediate storage (efficient compression)
- Stage 1 can sample large files (set `sample_size` in config)

## LLM Agent (NEW!)

### Purpose
A conversational AI agent that allows users to interact with the 4-stage agentic pipeline through natural language. Built with **LangGraph** and **LangChain**, users can chat with an LLM that autonomously calls the appropriate pipeline tools.

### Development Commands

```bash
# Navigate to the LLM agent directory
cd "Capstone-Project/llm-agent"

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY or OPENAI_API_KEY

# Run interactive mode
python main.py

# Run single query
python main.py --query "Analyze my data and suggest tasks"

# Use specific model
python main.py --model gpt-4o
```

### Architecture

LangGraph-based agent with ReAct (Reasoning + Acting) pattern:

**Main Components:**
- `src/agent.py`: LangGraph agent with tool calling
- `src/tools.py`: 8 tools wrapping pipeline stages
- `src/states.py`: State management (conversation + pipeline state)
- `src/prompts.py`: System prompts and guidance
- `src/configs.py`: Configuration management
- `main.py`: CLI interface (interactive or single query)

**Available Tools:**
1. `list_data_files` - List available data files
2. `summarize_data` - Run Stage 1 (Summarizer)
3. `suggest_tasks` - Run Stage 2 (Task Suggester)
4. `plan_analysis` - Run Stage 3 (Planner)
5. `execute_analysis` - Run Stage 4 (Executor)
6. `view_summary` - View file summaries
7. `view_tasks` - View task suggestions
8. `view_results` - View analysis results

**Agent Workflow:**
```
User Query (natural language)
  ↓
LLM Agent analyzes request
  ↓
Agent decides which tool(s) to call
  ↓
Tools execute (interact with pipeline)
  ↓
Results returned to agent
  ↓
Agent formulates response
  ↓
User receives natural language response
```

**Example Conversation:**
```
You: What data files do I have?
Agent: [calls list_data_files]
      You have 3 files: crop_yield.csv, rainfall.csv, fertilizer_usage.csv

You: Analyze them and predict crop yields
Agent: [calls summarize_data → suggest_tasks → plan_analysis → execute_analysis]
      ✓ Analysis complete! The XGBoost model achieved R²=0.87...
```

### Integration with Agentic Pipeline

The LLM agent wraps the existing 4-stage pipeline:
- **No modification** to pipeline code required
- Tools import and call pipeline stages directly
- Data flows through normal pipeline directories
- Can be used alongside or instead of manual pipeline execution

**Data Flow:**
```
llm-agent/src/tools.py
  ↓ (imports)
agentic-pipeline/src/stage{1,2,3,4}/
  ↓ (reads/writes)
agentic-pipeline/data/
```

### Key Features

- **Natural Language Interface**: Chat instead of running commands
- **Autonomous Tool Calling**: Agent selects appropriate tools
- **Multi-Step Planning**: Agent can execute complex workflows
- **Error Handling**: Graceful recovery with user guidance
- **Streaming**: See agent thinking in real-time
- **Model Flexibility**: Works with Claude or GPT models
- **Stateful**: Maintains conversation context

### Important Notes

- Requires agentic pipeline to be set up first
- Needs at least one LLM API key (Anthropic or OpenAI)
- Data files must be in `../agentic-pipeline/data/raw/`
- Can run in separate virtual environment from pipeline
- See `llm-agent/README.md` for detailed documentation

## Local Deep Research Agent

### Purpose
A LangGraph-based conversational research agent that uses GraphRAG for knowledge graph search.

### Development Commands

```bash
# Navigate to the research agent directory
cd "Capstone-Project/local-deep-research"

# Install uv (if not already installed)
# Follow: https://docs.astral.sh/uv/getting-started/installation/

# Setup environment variables
touch .env
# Add to .env:
# OPENAI_API_KEY=<your_key>
# LANGSMITH_API_KEY=<your_key>
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT="deep-agent"
# GRAPHRAG_API_KEY=<your_openai_key>
# GRAPHRAG_LLM_MODEL=gpt-4o-mini
# GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small

# Run the agent with uv
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

### Architecture

Built using LangGraph with a supervisor-worker pattern:

**Main Components**
- `src/agent.py`: Main agent logic with state machine (clarification → research question generation → supervisor → final report)
- `src/supervisor.py`: Supervisor subgraph that coordinates research workers
- `src/factory_research_agent.py`: Factory for creating specialized research agents
- `src/kg_search_manager.py`: Knowledge graph search manager using GraphRAG
- `src/tools.py`: Tool definitions for web search, document retrieval
- `src/states.py`: State definitions for the agent workflow
- `src/prompts.py`: Prompt templates for different agent stages
- `src/configs.py`: Configuration management

**Data Directory**
All unstructured data (CSVs, documents) must reside in `data/` directory.

**Key Dependencies**
- LangGraph for agent orchestration
- LangChain for LLM integration (OpenAI, Anthropic)
- GraphRAG for knowledge graph construction and retrieval
- Tavily for web search
- Docker for GraphRAG backend

### Agent Workflow

1. **Clarify with user**: Optionally ask clarifying questions (if `allow_clarification` enabled)
2. **Generate research question**: Create structured research questions
3. **Supervisor**: Coordinate research workers to gather information
4. **Final report**: Generate comprehensive report from gathered research

## Project Structure Notes

- The repository root is `Capstone Project/` (with space in name)
- Main project code is in `Capstone-Project/` subdirectory
- `.git` repository exists at `Capstone-Project/` level
- Test files are located in `Capstone-Project/agentic-pipeline/tests/` (not created yet for local-deep-research)

## Common Gotchas

- **Paths with spaces**: Always quote paths when using bash commands (e.g., `cd "Capstone Project"`)
- **Virtual environment**: Must activate venv before running pipeline commands
- **Data directory**: Agentic pipeline expects data in `data/raw/` before running
- **API keys**: Local deep research requires API keys; agentic pipeline works without them (uses rule-based heuristics)
- **Python version**: Local deep research requires Python 3.11+; agentic pipeline works with 3.8+
