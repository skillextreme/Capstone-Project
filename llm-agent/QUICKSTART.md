# 🚀 Quick Start Guide

Get started with the LLM Agent in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Agentic pipeline set up (see `../agentic-pipeline/`)
- Anthropic or OpenAI API key

## Step 1: Setup Virtual Environment

```bash
cd llm-agent
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Configure API Key

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
nano .env  # or use any text editor

# Add one of these:
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# OPENAI_API_KEY=sk-your-key-here
```

## Step 4: Verify Data Files

```bash
# Check that you have data files
ls ../agentic-pipeline/data/raw/

# You should see CSV or JSON files
# If not, copy your data files there first
```

## Step 5: Run the Agent!

```bash
python main.py
```

## Example First Conversation

```
🧑 You: What data files do I have?
🤖 Agent: [calls list_data_files]
         You have 3 files: crop_yield.csv (2.5 MB), rainfall.csv (1.2 MB), ...

🧑 You: Analyze all the data
🤖 Agent: [calls summarize_data]
         ✓ Successfully summarized 3 files! ...

🧑 You: What analyses can I run?
🤖 Agent: [calls suggest_tasks]
         I found 5 possible tasks. The highest feasibility task is T1:
         "Predict crop yield using rainfall and fertilizer data" (0.95)...

🧑 You: Run task T1
🤖 Agent: [calls plan_analysis, then execute_analysis]
         ✓ Plan created with 2 file joins and 8 engineered features...
         ✓ Analysis complete! The XGBoost model achieved R²=0.87 on test set...

🧑 You: quit
👋 Goodbye!
```

## Common Commands

**Data exploration:**
- "What data do I have?"
- "Summarize the data"
- "Show me crop_yield.csv"

**Run analysis:**
- "Analyze my data and suggest tasks"
- "Run task T1"
- "Predict crop yields"

**View results:**
- "Show me the results for T1"
- "What was the model performance?"
- "Show me the predictions"

## Troubleshooting

**"No API keys found"**
- Check that `.env` file exists in `llm-agent/` directory
- Verify API key is correctly formatted
- No quotes needed around the key in `.env`

**"No data files found"**
- Data must be in `../agentic-pipeline/data/raw/`
- Use `ls ../agentic-pipeline/data/raw/` to verify

**"Import error"**
- Make sure agentic-pipeline is set up
- Activate the correct virtual environment
- Run `pip install -r requirements.txt` again

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Try different queries and explore the tools
- Experiment with different LLM models
- Check out the [agentic-pipeline documentation](../agentic-pipeline/README.md)

---

**Need help?** Check the [README.md](README.md) or the troubleshooting section!
