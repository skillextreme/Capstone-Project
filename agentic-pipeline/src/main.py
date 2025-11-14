"""
Main Pipeline Orchestrator

Coordinates all 4 stages of the agentic pipeline with verification checkpoints.

Usage:
    # Run full pipeline
    python src/main.py --mode full

    # Run individual stages
    python src/main.py --mode stage1
    python src/main.py --mode stage2
    python src/main.py --mode stage3 --task-id T1
    python src/main.py --mode stage4 --task-id T1
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

from config import get_config
from utils.logging_utils import setup_logger, get_logger
from utils.file_utils import load_json, save_json

# Stage imports
from stage1.summarizer import Summarizer
from stage2.task_suggester import TaskSuggester
from stage3.planner import Planner
from stage4.executor import Executor

# Verification imports
from verifiers.schema_check import SchemaChecker
from verifiers.join_check import JoinChecker
from verifiers.metrics_check import MetricsChecker

logger = get_logger(__name__)


class Pipeline:
    """
    Main pipeline orchestrator.

    Coordinates all stages and verifications.

    Example:
        >>> pipeline = Pipeline()
        >>> pipeline.run_full()
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            config_file: Path to config file (optional)
        """
        # Load configuration
        self.config = get_config(config_file)

        # Setup logging
        log_config = self.config.get_stage_config('logging')
        setup_logger(
            __name__,
            log_file=log_config.get('file', {}).get('path') if log_config.get('file', {}).get('enabled') else None,
            level=log_config.get('level', 'INFO')
        )

        logger.info("=" * 80)
        logger.info("Agentic Pipeline Initialized")
        logger.info("=" * 80)

    def run_full(self):
        """
        Run the complete pipeline from Stage 1 to Stage 4.

        This executes:
        1. Stage 1: Summarizer + V1 verification
        2. Stage 2: Task Suggester + V2 (human selection)
        3. Stage 3: Planner + V3 verification (for each task)
        4. Stage 4: Executor + V4 verification (for each task)
        """
        logger.info("Running FULL PIPELINE")

        # Stage 1: Summarize
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Summarizer")
        logger.info("=" * 80)

        summaries = self.run_stage1()

        if not summaries:
            logger.error("Stage 1 failed - no summaries generated")
            return

        # Verification V1
        logger.info("\n" + "-" * 80)
        logger.info("VERIFICATION V1: Schema Check")
        logger.info("-" * 80)

        self.run_verification_v1()

        # Stage 2: Task Suggestion
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Task Suggester")
        logger.info("=" * 80)

        tasks = self.run_stage2()

        if not tasks:
            logger.error("Stage 2 failed - no tasks generated")
            return

        # Verification V2: Human adjudication
        logger.info("\n" + "-" * 80)
        logger.info("VERIFICATION V2: Human Adjudication")
        logger.info("-" * 80)

        selected_tasks = self.run_verification_v2(tasks)

        # Process each selected task
        for task in selected_tasks:
            task_id = task['task_id']

            logger.info("\n" + "=" * 80)
            logger.info(f"Processing Task {task_id}: {task['description']}")
            logger.info("=" * 80)

            # Stage 3: Planner
            logger.info("\n" + "=" * 80)
            logger.info(f"STAGE 3: Planner (Task {task_id})")
            logger.info("=" * 80)

            merged_df = self.run_stage3(task)

            if merged_df is None or len(merged_df) == 0:
                logger.error(f"Stage 3 failed for task {task_id}")
                continue

            # Verification V3
            logger.info("\n" + "-" * 80)
            logger.info(f"VERIFICATION V3: Join Check (Task {task_id})")
            logger.info("-" * 80)

            self.run_verification_v3(task_id)

            # Stage 4: Executor
            logger.info("\n" + "=" * 80)
            logger.info(f"STAGE 4: Executor (Task {task_id})")
            logger.info("=" * 80)

            results = self.run_stage4(task)

            if not results:
                logger.error(f"Stage 4 failed for task {task_id}")
                continue

            # Verification V4
            logger.info("\n" + "-" * 80)
            logger.info(f"VERIFICATION V4: Metrics Check (Task {task_id})")
            logger.info("-" * 80)

            self.run_verification_v4(task_id)

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

    def run_stage1(self) -> List[Dict[str, Any]]:
        """Run Stage 1: Summarizer."""
        data_config = self.config.get_stage_config('data')
        summarizer_config = self.config.get_stage_config('summarizer')

        summarizer = Summarizer(
            data_dir=data_config.get('raw_dir', 'data/raw'),
            output_dir=data_config.get('summaries_dir', 'data/summaries'),
            config=summarizer_config
        )

        summaries = summarizer.run_all()

        logger.info(f"✓ Stage 1 complete: {len(summaries)} files summarized")

        return summaries

    def run_verification_v1(self) -> List[Dict[str, Any]]:
        """Run Verification V1: Schema Check."""
        data_config = self.config.get_stage_config('data')
        v1_config = self.config.get_verification_config('v1')

        checker = SchemaChecker(config=v1_config)

        reports = checker.verify_all(
            summaries_dir=data_config.get('summaries_dir', 'data/summaries'),
            data_dir=data_config.get('raw_dir', 'data/raw')
        )

        logger.info(f"✓ Verification V1 complete: {len(reports)} files verified")

        return reports

    def run_stage2(self) -> List[Dict[str, Any]]:
        """Run Stage 2: Task Suggester."""
        data_config = self.config.get_stage_config('data')
        suggester_config = self.config.get_stage_config('task_suggester')

        suggester = TaskSuggester(
            summaries_dir=data_config.get('summaries_dir', 'data/summaries'),
            config=suggester_config
        )

        tasks = suggester.suggest_tasks()

        # Save tasks
        tasks_file = Path('data/tasks.json')
        save_json(tasks, tasks_file)

        logger.info(f"✓ Stage 2 complete: {len(tasks)} tasks suggested")
        logger.info(f"✓ Tasks saved to: {tasks_file}")

        return tasks

    def run_verification_v2(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run Verification V2: Human Adjudication.

        Displays tasks to user and allows selection.

        Args:
            tasks: List of suggested tasks

        Returns:
            List of selected tasks
        """
        print("\n" + "=" * 80)
        print("TASK SELECTION")
        print("=" * 80)

        print(f"\nFound {len(tasks)} suggested tasks:\n")

        for i, task in enumerate(tasks, 1):
            print(f"[{i}] {task['task_id']}: {task['type'].upper()}")
            print(f"    {task['description']}")
            print(f"    Files: {', '.join(task.get('required_files', []))}")
            print(f"    Feasibility: {task.get('feasibility', 'unknown')}")
            print()

        # For automation, select all high-feasibility tasks
        # In interactive mode, user would select manually
        selected = [t for t in tasks if t.get('feasibility') == 'high']

        if not selected:
            # If no high-feasibility tasks, select first task
            selected = tasks[:1]

        print(f"Auto-selected {len(selected)} high-feasibility tasks:")
        for task in selected:
            print(f"  - {task['task_id']}: {task['description']}")

        logger.info(f"✓ Verification V2 complete: {len(selected)} tasks selected")

        return selected

    def run_stage3(self, task: Dict[str, Any]):
        """Run Stage 3: Planner for a specific task."""
        data_config = self.config.get_stage_config('data')
        planner_config = self.config.get_stage_config('planner')

        planner = Planner(
            task=task,
            data_dir=data_config.get('raw_dir', 'data/raw'),
            output_dir=data_config.get('intermediate_dir', 'data/intermediate'),
            config=planner_config
        )

        merged_df = planner.execute_plan()

        logger.info(f"✓ Stage 3 complete: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        return merged_df

    def run_verification_v3(self, task_id: str) -> Dict[str, Any]:
        """Run Verification V3: Join Check for a specific task."""
        data_config = self.config.get_stage_config('data')
        v3_config = self.config.get_verification_config('v3')

        checker = JoinChecker(config=v3_config)

        intermediate_dir = Path(data_config.get('intermediate_dir', 'data/intermediate'))
        data_path = intermediate_dir / f"{task_id}_merged.parquet"
        plan_path = intermediate_dir / f"{task_id}_plan.json"

        report = checker.verify_plan(data_path, plan_path)

        logger.info(f"✓ Verification V3 complete: Status = {report['status']}")

        return report

    def run_stage4(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 4: Executor for a specific task."""
        data_config = self.config.get_stage_config('data')
        executor_config = self.config.get_stage_config('executor')

        task_id = task['task_id']
        data_path = Path(data_config.get('intermediate_dir', 'data/intermediate')) / f"{task_id}_merged.parquet"

        executor = Executor(
            task=task,
            data_path=str(data_path),
            output_dir=data_config.get('outputs_dir', 'data/outputs'),
            config=executor_config
        )

        results = executor.run()

        logger.info(f"✓ Stage 4 complete")

        return results

    def run_verification_v4(self, task_id: str) -> Dict[str, Any]:
        """Run Verification V4: Metrics Check for a specific task."""
        data_config = self.config.get_stage_config('data')
        v4_config = self.config.get_verification_config('v4')

        checker = MetricsChecker(config=v4_config)

        outputs_dir = Path(data_config.get('outputs_dir', 'data/outputs'))
        metrics_path = outputs_dir / f"{task_id}_metrics.json"
        model_card_path = outputs_dir / f"{task_id}_model_card.json"
        predictions_path = outputs_dir / f"{task_id}_predictions.csv"

        report = checker.verify_results(
            metrics_path,
            model_card_path,
            predictions_path if predictions_path.exists() else None
        )

        logger.info(f"✓ Verification V4 complete: Status = {report['status']}")

        return report


def main():
    """
    CLI entry point for the pipeline.

    Usage:
        python src/main.py --mode full
        python src/main.py --mode stage1
        python src/main.py --mode stage2
        python src/main.py --mode stage3 --task-id T1
        python src/main.py --mode stage4 --task-id T1
    """
    parser = argparse.ArgumentParser(
        description="Agentic Pipeline for Agricultural Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'stage1', 'stage2', 'stage3', 'stage4'],
        default='full',
        help='Pipeline mode (default: full)'
    )

    parser.add_argument(
        '--task-id',
        help='Task ID for stage3/stage4 (e.g., T1)'
    )

    parser.add_argument(
        '--config',
        help='Path to config file (default: config/pipeline_config.yaml)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = Pipeline(config_file=args.config)

    # Set log level
    if args.verbose:
        logger.setLevel('DEBUG')

    # Run requested mode
    if args.mode == 'full':
        pipeline.run_full()

    elif args.mode == 'stage1':
        pipeline.run_stage1()
        pipeline.run_verification_v1()

    elif args.mode == 'stage2':
        pipeline.run_stage2()

    elif args.mode == 'stage3':
        if not args.task_id:
            print("Error: --task-id required for stage3")
            sys.exit(1)

        tasks = load_json('data/tasks.json')
        task = next((t for t in tasks if t['task_id'] == args.task_id), None)

        if not task:
            print(f"Error: Task {args.task_id} not found")
            sys.exit(1)

        pipeline.run_stage3(task)
        pipeline.run_verification_v3(args.task_id)

    elif args.mode == 'stage4':
        if not args.task_id:
            print("Error: --task-id required for stage4")
            sys.exit(1)

        tasks = load_json('data/tasks.json')
        task = next((t for t in tasks if t['task_id'] == args.task_id), None)

        if not task:
            print(f"Error: Task {args.task_id} not found")
            sys.exit(1)

        pipeline.run_stage4(task)
        pipeline.run_verification_v4(args.task_id)


if __name__ == '__main__':
    main()
