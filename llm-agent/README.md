# LLM Agent for Agricultural Data Analysis

An intelligent conversational agent that uses Large Language Models (LLMs) to orchestrate a 4-stage agricultural data analysis pipeline. Built with **LangGraph** and **LangChain**, this agent allows you to interact with complex data analysis workflows through natural language.

## 🌟 Overview

Instead of manually running each stage of the pipeline, you can simply chat with an AI agent that:
- Understands your analysis goals through conversation
- Automatically calls the appropriate tools (pipeline stages)
- Guides you through the analysis workflow
- Interprets and explains results in natural language

**Example conversation:**
```
You: What data files do I have?
Agent: Let me check... [calls list_data_files]
      You have 3 files: crop_yield.csv, rainfall.csv, fertilizer_usage.csv

You: Can you analyze these and predict crop yields?
Agent: I'll help you with that. First, let me summarize the data... [calls summarize_data]
      ✓ Successfully summarized 3 files!
      Now let me suggest some tasks... [calls suggest_tasks]
      I found a high-feasibility task (T1) for predicting crop yields using rainfall and fertilizer data.
      Should I proceed with planning this analysis?

You: Yes, go ahead
Agent: [calls plan_analysis with task_id='T1']
      ✓ Plan created! Now executing the analysis... [calls execute_analysis]
      ✓ Analysis complete! The XGBoost model achieved R²=0.87 on the test set...
```

## 🏗️ Architecture

```
User Input (Natural Language)
        ↓
    LLM Agent (Claude/GPT)
        ↓
   Tool Selection & Execution
        ↓
┌───────────────────────────────┐
│  Pipeline Tools (via LangChain) │
├───────────────────────────────┤
│ 1. list_data_files            │
│ 2. summarize_data (Stage 1)   │
│ 3. suggest_tasks (Stage 2)    │
│ 4. plan_analysis (Stage 3)    │
│ 5. execute_analysis (Stage 4) │
│ 6. view_summary                │
│ 7. view_tasks                  │
│ 8. view_results                │
└───────────────────────────────┘
        ↓
  LLM Response (Natural Language)
        ↓
      User
```

### Key Components

**1. Agent (`src/agent.py`)**
- LangGraph-based agent using ReAct pattern
- Manages conversation state and tool calling
- Routes between agent thinking and tool execution

**2. Tools (`src/tools.py`)**
- Wraps each pipeline stage as a LangChain tool
- 8 tools available for different operations
- Provides detailed descriptions for LLM tool selection

**3. State Management (`src/states.py`)**
- Tracks conversation history
- Maintains pipeline execution state (what stages completed)
- Stores intermediate results and errors

**4. Configuration (`src/configs.py`)**
- LLM model selection (Claude or GPT)
- Temperature, max tokens, etc.
- Tool execution settings

## 📋 Prerequisites

1. **Python 3.8+** (Python 3.11+ recommended)

2. **Agentic Pipeline** - Must be set up first:
   ```bash
   cd ../agentic-pipeline
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **API Keys** - At least one of:
   - Anthropic API key (for Claude) - **Recommended**
   - OpenAI API key (for GPT)

## 🚀 Setup

### 1. Create Virtual Environment

```bash
cd llm-agent
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key(s)
# ANTHROPIC_API_KEY=sk-ant-...
# or
# OPENAI_API_KEY=sk-...
```

### 4. Ensure Data Files Exist

The agent works with data in the agentic-pipeline directory:

```bash
# Make sure you have data files in:
ls ../agentic-pipeline/data/raw/

# You should see CSV or JSON files like:
# crop_yield.csv, rainfall.csv, etc.
```

## 💻 Usage

### Interactive Mode (Recommended)

Start a conversation with the agent:

```bash
python main.py
```

**Example interactions:**

```
🧑 You: What data do I have available?
🤖 Agent: [calls list_data_files] You have 3 files: crop_yield.csv (2.5 MB), rainfall.csv (1.2 MB), fertilizer_usage.csv (0.8 MB)

🧑 You: Summarize the crop yield data
🤖 Agent: [calls summarize_data] ✓ Successfully summarized! crop_yield.csv has 5 columns, 10,000 rows...

🧑 You: What analyses can I run?
🤖 Agent: [calls suggest_tasks] I found 5 possible tasks...

🧑 You: Run task T1
🤖 Agent: [calls plan_analysis, then execute_analysis] ✓ Analysis complete! The model achieved...
```

### Single Query Mode

Run a single query non-interactively:

```bash
python main.py --query "Summarize all my data files and suggest analysis tasks"
```

### Custom Model

Use a specific LLM model:

```bash
# Use GPT-4
python main.py --model gpt-4o

# Use Claude Opus
python main.py --model claude-3-opus-20240229
```

## 🛠️ Available Tools

The agent has access to 8 tools:

### Data Exploration Tools

**1. `list_data_files`**
- Lists available data files in `data/raw/`
- Supports glob patterns (e.g., `*.csv`, `crop*`)

**2. `view_summary`**
- View detailed summary for a specific file
- Shows schema, statistics, candidate keys

**3. `view_tasks`**
- View suggested analysis tasks
- Can filter by task_id

**4. `view_results`**
- View results from completed analyses
- Options: summary, metrics, predictions, model_card

### Pipeline Stage Tools

**5. `summarize_data` (Stage 1)**
- Analyzes CSV/JSON files
- Generates schema, statistics, data quality metrics
- Identifies candidate keys

**6. `suggest_tasks` (Stage 2)**
- Proposes feasible analysis tasks
- Types: prediction, descriptive, clustering
- Provides feasibility scores

**7. `plan_analysis` (Stage 3)**
- Creates reproducible data plan
- Handles file joining and feature engineering
- Requires task_id from suggest_tasks

**8. `execute_analysis` (Stage 4)**
- Trains ML models (Ridge, XGBoost, Random Forest)
- Generates predictions and visualizations
- Creates comprehensive reports

## 📊 Workflow Example

Here's a typical analysis workflow:

```python
# 1. Explore available data
You: "What data files do I have?"
Agent: [calls list_data_files]

# 2. Summarize the data
You: "Analyze all the data files"
Agent: [calls summarize_data]

# 3. Get task suggestions
You: "What analyses can I run with this data?"
Agent: [calls suggest_tasks]

# 4. Select and plan a task
You: "Let's do task T1 - predict crop yields"
Agent: [calls plan_analysis with task_id='T1']

# 5. Execute the analysis
You: "Run the analysis"
Agent: [calls execute_analysis with task_id='T1']

# 6. View and interpret results
You: "Show me the model performance"
Agent: [calls view_results] The model achieved R²=0.87...
```

## 🎯 Example Queries

**Data Exploration:**
- "What data files are available?"
- "Show me the summary for crop_yield.csv"
- "What are the columns in rainfall.csv?"

**Analysis Requests:**
- "Predict crop yields using all available data"
- "Find patterns in rainfall data"
- "Suggest some analyses I can run"
- "What's the best task to start with?"

**Pipeline Execution:**
- "Run the full analysis pipeline"
- "Execute task T1"
- "Create a plan for predicting yields"

**Results Interpretation:**
- "Show me the results for task T1"
- "What was the model performance?"
- "Explain the predictions to me"
- "What are the key insights?"

## ⚙️ Configuration

### Changing the LLM Model

Edit `src/configs.py` or pass configuration at runtime:

```python
from src.agent import agent_graph

config = {
    "configurable": {
        "agent_llm": {
            "model": "claude-3-5-sonnet-20241022",  # or "gpt-4o"
            "temperature": 0.7,  # Lower = more focused, Higher = more creative
            "max_tokens": 4000,
        }
    }
}

# Run with custom config
result = agent_graph.invoke(state, config)
```

### Model Options

**Claude (Anthropic):**
- `claude-3-5-sonnet-20241022` (Recommended - best balance)
- `claude-3-opus-20240229` (Most capable)
- `claude-3-haiku-20240307` (Fastest, cheapest)

**GPT (OpenAI):**
- `gpt-4o` (Recommended)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (Cheapest)

## 🔍 How It Works

### LangGraph Agent Flow

```
1. User sends message
   ↓
2. Agent (LLM) receives message + tool descriptions
   ↓
3. Agent decides: [Call tool] or [Respond directly]
   ↓
4. If tool call:
   - Execute tool (e.g., summarize_data)
   - Get result
   - Return to step 2 (agent sees tool result)
   ↓
5. Agent formulates final response
   ↓
6. User receives response
```

### ReAct Pattern

The agent uses the **ReAct (Reasoning + Acting)** pattern:
- **Reason**: Analyzes the user's request and available tools
- **Act**: Calls appropriate tools to gather information
- **Reason**: Interprets tool results
- **Respond**: Provides answer to user

This allows the agent to:
- Make multi-step plans
- Gather information incrementally
- Adapt based on intermediate results
- Provide informed, contextualized responses

## 🧪 Testing

### Quick Test

```bash
# Test basic functionality
python main.py --query "List available data files"

# Test full pipeline
python main.py --query "Analyze my data and suggest tasks"
```

### Interactive Testing

```bash
python main.py

# Try these commands:
You: list my data files
You: summarize the data
You: suggest some tasks
You: show me task T1
```

## 📁 Project Structure

```
llm-agent/
├── src/
│   ├── __init__.py        # Package initialization
│   ├── agent.py           # Main LangGraph agent
│   ├── states.py          # State definitions
│   ├── tools.py           # Pipeline tool wrappers
│   ├── prompts.py         # System prompts
│   └── configs.py         # Configuration
├── main.py                # CLI interface
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔗 Integration with Agentic Pipeline

This agent wraps the existing 4-stage pipeline:

```
llm-agent/
└── src/tools.py  ──────→  Imports from:  ──────→  agentic-pipeline/
                                                    ├── src/stage1/summarizer.py
                                                    ├── src/stage2/task_suggester.py
                                                    ├── src/stage3/planner.py
                                                    └── src/stage4/executor.py
```

**Requirements:**
- Agentic pipeline must be set up in `../agentic-pipeline/`
- Data files must be in `../agentic-pipeline/data/raw/`
- Both virtual environments can be separate

## 🐛 Troubleshooting

### "No API keys found"
- Make sure you've created a `.env` file (copy from `.env.example`)
- Add your `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- Ensure the `.env` file is in the `llm-agent/` directory

### "No module named 'src.stage1'"
- The agentic pipeline must be installed
- Check that `../agentic-pipeline/` exists and has `src/` directory
- The agent adds `agentic-pipeline` to Python path automatically

### "No data files found"
- Place your CSV/JSON files in `../agentic-pipeline/data/raw/`
- Run `list_data_files` tool to verify

### Agent not calling tools
- Try a more explicit request: "Use the list_data_files tool"
- Check that your API key is valid and has credit
- Try lowering temperature in config (makes agent more focused)

### Tool execution errors
- Check logs for specific error messages
- Verify the agentic pipeline works standalone first
- Ensure all dependencies are installed

## 📚 Learn More

**LangGraph Documentation:**
- https://langchain-ai.github.io/langgraph/

**LangChain Tool Use:**
- https://python.langchain.com/docs/how_to/custom_tools/

**Anthropic Claude:**
- https://docs.anthropic.com/

**OpenAI API:**
- https://platform.openai.com/docs/

## 🤝 Contributing

This is a capstone project. Improvements welcome!

**Ideas for enhancement:**
- Add support for more LLM providers (Gemini, Mistral, etc.)
- Implement conversation memory persistence
- Add visualization preview in terminal
- Create web UI with Streamlit or Gradio
- Add support for custom data sources beyond CSV/JSON
- Implement multi-turn task refinement

## 📄 License

MIT License - See main project for details

## ✨ Key Features

✅ **Natural Language Interface** - Chat with your data pipeline
✅ **Tool Calling** - Agent autonomously selects and uses appropriate tools
✅ **Multi-Step Planning** - Agent can execute complex workflows
✅ **Model Flexibility** - Works with Claude or GPT models
✅ **Stateful Conversations** - Maintains context across turns
✅ **Error Handling** - Graceful error recovery and user guidance
✅ **Streaming Responses** - See agent thinking in real-time
✅ **Production Ready** - Built on LangGraph for scalability

---

**Made with ❤️ using LangGraph, LangChain, and Claude/GPT**
