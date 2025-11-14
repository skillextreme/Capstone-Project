"""Tool wrappers for the 4-stage pipeline."""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import glob

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add parent directory to path to import pipeline modules
PIPELINE_DIR = Path(__file__).parent.parent.parent / "agentic-pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from src.stage1.summarizer import Summarizer
from src.stage2.task_suggester import TaskSuggester
from src.stage3.planner import Planner
from src.stage4.executor import Executor
from src.config import Config
from src.utils.file_utils import load_json


# Initialize configuration
config = Config()


@tool("list_data_files")
def list_data_files(pattern: str = "*") -> str:
    """List available data files in the data/raw/ directory.

    Use this tool to see what data files are available for analysis.

    Args:
        pattern: Glob pattern to filter files (e.g., '*.csv', 'crop*'). Default is '*' for all files.

    Returns:
        A formatted string listing all matching files with their sizes.
    """
    try:
        raw_data_path = PIPELINE_DIR / config.paths.raw_data
        if not raw_data_path.exists():
            return f"Error: Data directory {raw_data_path} does not exist."

        pattern_path = raw_data_path / pattern
        files = glob.glob(str(pattern_path))

        if not files:
            return f"No files found matching pattern '{pattern}' in {raw_data_path}"

        result = f"Found {len(files)} file(s) in {raw_data_path}:\n\n"
        for file in sorted(files):
            file_path = Path(file)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            result += f"  - {file_path.name} ({size_mb:.2f} MB)\n"

        return result

    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool("summarize_data")
def summarize_data(file_names: Optional[List[str]] = None, force_rerun: bool = False) -> str:
    """Run Stage 1: Analyze and summarize data files.

    This tool analyzes CSV/JSON files and generates comprehensive summaries including:
    - Schema information (column names, types)
    - Statistical summaries (mean, std, min, max)
    - Data quality metrics (null rates, cardinality)
    - Candidate keys (primary/foreign key detection)

    Use this as the FIRST step in the analysis pipeline.

    Args:
        file_names: Specific files to summarize (e.g., ['crop_yield.csv']). If None, summarizes all files.
        force_rerun: Force re-summarization even if summaries already exist.

    Returns:
        Summary of the summarization process and what was generated.
    """
    try:
        summarizer = Summarizer(config)
        raw_data_path = PIPELINE_DIR / config.paths.raw_data
        summaries_path = PIPELINE_DIR / config.paths.summaries

        # Check if summaries already exist
        if not force_rerun and summaries_path.exists():
            existing_summaries = list(summaries_path.glob("*.json"))
            if existing_summaries:
                return (
                    f"Summaries already exist for {len(existing_summaries)} file(s). "
                    f"Use force_rerun=True to regenerate. "
                    f"Use 'view_summary' tool to view existing summaries."
                )

        # Get files to process
        if file_names:
            files_to_process = [raw_data_path / f for f in file_names]
            # Verify files exist
            missing = [f for f in files_to_process if not f.exists()]
            if missing:
                return f"Error: Files not found: {[f.name for f in missing]}"
        else:
            files_to_process = None  # Will process all files

        # Run summarization
        results = summarizer.run_all()

        # Format response
        response = f"✓ Successfully summarized data files!\n\n"
        response += f"Total files processed: {len(results)}\n"
        response += f"Summaries saved to: {summaries_path}\n\n"
        response += "Summary statistics:\n"

        for file_name, summary in results.items():
            schema = summary.get("schema", {})
            stats = summary.get("statistics", {})
            response += f"\n  {file_name}:\n"
            response += f"    - Columns: {len(schema.get('columns', []))}\n"
            response += f"    - Rows: {stats.get('total_rows', 'N/A')}\n"
            response += f"    - Candidate keys found: {len(summary.get('candidate_keys', {}).get('individual', []))}\n"

        response += f"\n💡 Next step: Use 'suggest_tasks' to get analysis task suggestions."

        return response

    except Exception as e:
        return f"Error during summarization: {str(e)}\n\nPlease ensure data files are in the data/raw/ directory."


@tool("suggest_tasks")
def suggest_tasks(force_rerun: bool = False) -> str:
    """Run Stage 2: Generate analysis task suggestions.

    This tool analyzes the data summaries and proposes feasible analysis tasks including:
    - Prediction tasks (regression/classification)
    - Descriptive tasks (aggregations, rankings)
    - Clustering tasks (unsupervised learning)

    Use this AFTER running 'summarize_data'.

    Args:
        force_rerun: Force re-generation of task suggestions.

    Returns:
        List of suggested analysis tasks with feasibility scores.
    """
    try:
        summaries_path = PIPELINE_DIR / config.paths.summaries
        tasks_file = PIPELINE_DIR / config.paths.tasks

        # Check if summaries exist
        if not summaries_path.exists() or not list(summaries_path.glob("*.json")):
            return (
                "Error: No data summaries found. "
                "Please run 'summarize_data' first to analyze the data files."
            )

        # Check if tasks already exist
        if not force_rerun and tasks_file.exists():
            return (
                "Task suggestions already exist. "
                "Use force_rerun=True to regenerate, or use 'view_tasks' to see existing tasks."
            )

        # Run task suggestion
        task_suggester = TaskSuggester(config)
        tasks = task_suggester.suggest_tasks()

        # Format response
        response = f"✓ Successfully generated task suggestions!\n\n"
        response += f"Total tasks suggested: {len(tasks)}\n"
        response += f"Tasks saved to: {tasks_file}\n\n"

        # Group by type
        by_type = {}
        for task in tasks:
            task_type = task.get("type", "unknown")
            by_type.setdefault(task_type, []).append(task)

        response += "Tasks by type:\n"
        for task_type, type_tasks in by_type.items():
            response += f"\n  {task_type.upper()} ({len(type_tasks)} tasks):\n"
            for task in type_tasks[:3]:  # Show first 3 of each type
                task_id = task.get("task_id", "N/A")
                description = task.get("description", "N/A")
                feasibility = task.get("feasibility_score", 0)
                response += f"    - [{task_id}] {description[:80]}... (feasibility: {feasibility:.2f})\n"

            if len(type_tasks) > 3:
                response += f"    ... and {len(type_tasks) - 3} more\n"

        response += f"\n💡 Next step: Use 'view_tasks' to see full details, then 'plan_analysis' with a task_id."

        return response

    except Exception as e:
        return f"Error suggesting tasks: {str(e)}"


@tool("plan_analysis")
def plan_analysis(task_id: str, force_rerun: bool = False) -> str:
    """Run Stage 3: Create an analysis plan for a specific task.

    This tool creates a reproducible data plan including:
    - Data loading and normalization
    - File joining (with cardinality tracking)
    - Feature engineering (lags, rolling windows, interactions)
    - Missing value handling

    Use this AFTER selecting a task from 'suggest_tasks'.

    Args:
        task_id: The task ID to plan for (e.g., 'T1', 'T2'). Get this from 'view_tasks'.
        force_rerun: Force re-planning even if plan already exists.

    Returns:
        Summary of the created plan including join strategy and features.
    """
    try:
        tasks_file = PIPELINE_DIR / config.paths.tasks
        intermediate_path = PIPELINE_DIR / config.paths.intermediate_data

        # Check if tasks exist
        if not tasks_file.exists():
            return (
                "Error: No task suggestions found. "
                "Please run 'suggest_tasks' first."
            )

        # Load tasks
        tasks = load_json(tasks_file)
        task = next((t for t in tasks if t.get("task_id") == task_id), None)

        if not task:
            available_ids = [t.get("task_id") for t in tasks]
            return (
                f"Error: Task '{task_id}' not found. "
                f"Available task IDs: {', '.join(available_ids)}\n"
                f"Use 'view_tasks' to see all available tasks."
            )

        # Check if plan already exists
        plan_file = intermediate_path / f"{task_id}_merged.parquet"
        if not force_rerun and plan_file.exists():
            return (
                f"Plan for task '{task_id}' already exists. "
                f"Use force_rerun=True to regenerate, or proceed to 'execute_analysis'."
            )

        # Run planner
        planner = Planner(config)
        result = planner.execute_plan(task)

        # Format response
        response = f"✓ Successfully created plan for task '{task_id}'!\n\n"
        response += f"Task: {task.get('description', 'N/A')}\n\n"

        provenance = result.get("provenance", {})
        response += "Plan details:\n"
        response += f"  - Files merged: {len(provenance.get('source_files', []))}\n"

        joins = provenance.get("joins_performed", [])
        if joins:
            response += f"  - Joins performed: {len(joins)}\n"
            for join in joins[:2]:
                response += f"    • {join.get('left_file', 'N/A')} ⟷ {join.get('right_file', 'N/A')} "
                response += f"(on: {', '.join(join.get('on', []))})\n"

        features = provenance.get("feature_engineering", {})
        response += f"  - Lag features: {len(features.get('lag_features_created', []))}\n"
        response += f"  - Rolling features: {len(features.get('rolling_features_created', []))}\n"
        response += f"  - Interaction features: {len(features.get('interaction_features_created', []))}\n"

        final_shape = provenance.get("final_shape", {})
        response += f"\n  Final dataset: {final_shape.get('rows', 'N/A')} rows × {final_shape.get('columns', 'N/A')} columns\n"

        response += f"\n💡 Next step: Use 'execute_analysis' with task_id='{task_id}' to train models."

        return response

    except Exception as e:
        return f"Error creating plan: {str(e)}"


@tool("execute_analysis")
def execute_analysis(task_id: str, force_rerun: bool = False) -> str:
    """Run Stage 4: Execute the analysis and generate results.

    This tool:
    - Trains machine learning models (Ridge, XGBoost, Random Forest)
    - Evaluates on holdout set
    - Generates predictions and visualizations
    - Creates model cards and metrics reports

    Use this AFTER creating a plan with 'plan_analysis'.

    Args:
        task_id: The task ID to execute (e.g., 'T1', 'T2').
        force_rerun: Force re-execution even if results already exist.

    Returns:
        Summary of the execution results including model performance.
    """
    try:
        tasks_file = PIPELINE_DIR / config.paths.tasks
        intermediate_path = PIPELINE_DIR / config.paths.intermediate_data
        outputs_path = PIPELINE_DIR / config.paths.outputs

        # Check if plan exists
        plan_file = intermediate_path / f"{task_id}_merged.parquet"
        if not plan_file.exists():
            return (
                f"Error: No plan found for task '{task_id}'. "
                f"Please run 'plan_analysis' with task_id='{task_id}' first."
            )

        # Load task
        tasks = load_json(tasks_file)
        task = next((t for t in tasks if t.get("task_id") == task_id), None)
        if not task:
            return f"Error: Task '{task_id}' not found in tasks file."

        # Check if results already exist
        results_file = outputs_path / f"{task_id}_metrics.json"
        if not force_rerun and results_file.exists():
            return (
                f"Results for task '{task_id}' already exist. "
                f"Use force_rerun=True to regenerate, or use 'view_results' to see existing results."
            )

        # Run executor
        executor = Executor(config)
        result = executor.run(task)

        # Format response
        response = f"✓ Successfully executed analysis for task '{task_id}'!\n\n"

        if task.get("type") == "prediction":
            metrics = result.get("metrics", {})
            best_model = result.get("best_model", "N/A")

            response += f"Best model: {best_model}\n\n"
            response += "Performance metrics:\n"

            if best_model in metrics:
                model_metrics = metrics[best_model]
                test_metrics = model_metrics.get("test", {})
                response += f"  Test Set:\n"
                response += f"    - R²: {test_metrics.get('r2', 'N/A'):.4f}\n"
                response += f"    - MAE: {test_metrics.get('mae', 'N/A'):.4f}\n"
                response += f"    - RMSE: {test_metrics.get('rmse', 'N/A'):.4f}\n"

        response += f"\n📁 Results saved to: {outputs_path}\n"
        response += f"  - Metrics: {task_id}_metrics.json\n"
        response += f"  - Predictions: {task_id}_predictions.csv\n"
        response += f"  - Model card: {task_id}_model_card.json\n"
        response += f"  - Visualizations: plots/\n"

        response += f"\n💡 Use 'view_results' with task_id='{task_id}' to see detailed results."

        return response

    except Exception as e:
        return f"Error executing analysis: {str(e)}"


@tool("view_summary")
def view_summary(file_name: str) -> str:
    """View the summary for a specific data file.

    Use this to see detailed information about a file after running 'summarize_data'.

    Args:
        file_name: Name of the file to view summary for (e.g., 'crop_yield.csv').

    Returns:
        Detailed summary information for the file.
    """
    try:
        summaries_path = PIPELINE_DIR / config.paths.summaries
        summary_file = summaries_path / f"{Path(file_name).stem}_summary.json"

        if not summary_file.exists():
            available_summaries = [f.stem.replace("_summary", "") for f in summaries_path.glob("*_summary.json")]
            return (
                f"Error: Summary for '{file_name}' not found. "
                f"Available summaries: {', '.join(available_summaries)}\n"
                f"Use 'summarize_data' to generate summaries."
            )

        summary = load_json(summary_file)

        # Format response
        response = f"📊 Summary for {file_name}\n\n"

        schema = summary.get("schema", {})
        response += f"Schema:\n"
        response += f"  - Total columns: {len(schema.get('columns', []))}\n"

        stats = summary.get("statistics", {})
        response += f"\nStatistics:\n"
        response += f"  - Total rows: {stats.get('total_rows', 'N/A')}\n"
        response += f"  - Memory size: {stats.get('memory_usage', 'N/A')}\n"

        response += f"\nColumns:\n"
        for col in schema.get("columns", [])[:10]:  # Show first 10 columns
            col_name = col.get("name", "N/A")
            col_type = col.get("type", "N/A")
            null_rate = col.get("null_rate", 0)
            response += f"  - {col_name} ({col_type}) - null rate: {null_rate:.1%}\n"

        if len(schema.get("columns", [])) > 10:
            response += f"  ... and {len(schema.get('columns', [])) - 10} more columns\n"

        keys = summary.get("candidate_keys", {})
        individual_keys = keys.get("individual", [])
        if individual_keys:
            response += f"\nCandidate keys: {', '.join(individual_keys)}\n"

        return response

    except Exception as e:
        return f"Error viewing summary: {str(e)}"


@tool("view_tasks")
def view_tasks(task_id: Optional[str] = None) -> str:
    """View suggested analysis tasks.

    Use this to see what analysis tasks have been suggested after running 'suggest_tasks'.

    Args:
        task_id: Specific task ID to view details for. If None, shows all tasks.

    Returns:
        Information about the suggested tasks.
    """
    try:
        tasks_file = PIPELINE_DIR / config.paths.tasks

        if not tasks_file.exists():
            return (
                "Error: No task suggestions found. "
                "Please run 'suggest_tasks' first."
            )

        tasks = load_json(tasks_file)

        if task_id:
            # Show specific task
            task = next((t for t in tasks if t.get("task_id") == task_id), None)
            if not task:
                available_ids = [t.get("task_id") for t in tasks]
                return f"Error: Task '{task_id}' not found. Available: {', '.join(available_ids)}"

            response = f"📋 Task {task_id}\n\n"
            response += f"Type: {task.get('type', 'N/A')}\n"
            response += f"Description: {task.get('description', 'N/A')}\n"
            response += f"Feasibility: {task.get('feasibility_score', 0):.2f}\n\n"

            if task.get("type") == "prediction":
                response += f"Target variable: {task.get('target', 'N/A')}\n"
                response += f"Files: {', '.join(task.get('files', []))}\n"

            rationale = task.get("rationale", {})
            if rationale:
                response += f"\nRationale:\n"
                for key, value in rationale.items():
                    response += f"  - {key}: {value}\n"

        else:
            # Show all tasks
            response = f"📋 All Suggested Tasks ({len(tasks)} total)\n\n"

            by_type = {}
            for task in tasks:
                task_type = task.get("type", "unknown")
                by_type.setdefault(task_type, []).append(task)

            for task_type, type_tasks in by_type.items():
                response += f"{task_type.upper()} ({len(type_tasks)} tasks):\n"
                for task in type_tasks:
                    task_id = task.get("task_id", "N/A")
                    description = task.get("description", "N/A")
                    feasibility = task.get("feasibility_score", 0)
                    response += f"  [{task_id}] {description}\n"
                    response += f"      Feasibility: {feasibility:.2f}\n"
                response += "\n"

            response += "💡 Use 'view_tasks' with a specific task_id to see full details."

        return response

    except Exception as e:
        return f"Error viewing tasks: {str(e)}"


@tool("view_results")
def view_results(task_id: str, result_type: str = "summary") -> str:
    """View analysis results for a completed task.

    Use this after running 'execute_analysis' to see the results.

    Args:
        task_id: The task ID to view results for (e.g., 'T1', 'T2').
        result_type: Type of result to view - 'summary', 'metrics', 'predictions', or 'model_card'.

    Returns:
        The requested results information.
    """
    try:
        outputs_path = PIPELINE_DIR / config.paths.outputs

        metrics_file = outputs_path / f"{task_id}_metrics.json"
        if not metrics_file.exists():
            return (
                f"Error: No results found for task '{task_id}'. "
                f"Please run 'execute_analysis' with task_id='{task_id}' first."
            )

        if result_type == "metrics":
            metrics = load_json(metrics_file)
            response = f"📊 Metrics for Task {task_id}\n\n"

            for model_name, model_metrics in metrics.items():
                response += f"{model_name}:\n"

                for split in ["train", "validation", "test"]:
                    if split in model_metrics:
                        split_metrics = model_metrics[split]
                        response += f"  {split.capitalize()}:\n"
                        for metric_name, value in split_metrics.items():
                            response += f"    - {metric_name}: {value:.4f}\n"
                response += "\n"

            return response

        elif result_type == "model_card":
            model_card_file = outputs_path / f"{task_id}_model_card.json"
            if not model_card_file.exists():
                return f"Error: Model card not found for task '{task_id}'."

            model_card = load_json(model_card_file)
            response = f"📄 Model Card for Task {task_id}\n\n"

            for section, content in model_card.items():
                response += f"{section.replace('_', ' ').title()}:\n"
                if isinstance(content, dict):
                    for key, value in content.items():
                        response += f"  - {key}: {value}\n"
                else:
                    response += f"  {content}\n"
                response += "\n"

            return response

        elif result_type == "predictions":
            predictions_file = outputs_path / f"{task_id}_predictions.csv"
            if not predictions_file.exists():
                return f"Error: Predictions file not found for task '{task_id}'."

            # Read first few lines
            import pandas as pd
            df = pd.read_csv(predictions_file)

            response = f"🔮 Predictions for Task {task_id}\n\n"
            response += f"Total predictions: {len(df)}\n"
            response += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            response += "First 10 rows:\n"
            response += df.head(10).to_string(index=False)

            return response

        else:  # summary
            metrics = load_json(metrics_file)
            response = f"📊 Results Summary for Task {task_id}\n\n"

            # Find best model
            best_model = None
            best_mae = float('inf')
            for model_name, model_metrics in metrics.items():
                test_mae = model_metrics.get("test", {}).get("mae", float('inf'))
                if test_mae < best_mae:
                    best_mae = test_mae
                    best_model = model_name

            if best_model:
                response += f"Best model: {best_model}\n\n"
                model_metrics = metrics[best_model]

                for split in ["train", "test"]:
                    if split in model_metrics:
                        split_metrics = model_metrics[split]
                        response += f"{split.capitalize()} metrics:\n"
                        for metric_name, value in split_metrics.items():
                            response += f"  - {metric_name}: {value:.4f}\n"
                        response += "\n"

            response += f"📁 Full results available in: {outputs_path}\n"
            response += f"  - Metrics: {task_id}_metrics.json\n"
            response += f"  - Predictions: {task_id}_predictions.csv\n"
            response += f"  - Model card: {task_id}_model_card.json\n"

            return response

    except Exception as e:
        return f"Error viewing results: {str(e)}"


def get_all_tools():
    """Get all available tools for the agent."""
    return [
        list_data_files,
        summarize_data,
        suggest_tasks,
        plan_analysis,
        execute_analysis,
        view_summary,
        view_tasks,
        view_results,
    ]
