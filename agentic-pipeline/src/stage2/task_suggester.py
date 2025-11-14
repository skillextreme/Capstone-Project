"""
Task Suggestion Agent - Stage 2

Analyzes summaries from Stage 1 and proposes feasible analysis tasks.
Uses rule-based heuristics or LLM-powered suggestions.

Output: List of task proposals with required files, keys, and features.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from itertools import combinations

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_json, save_json

logger = get_logger(__name__)


class TaskSuggester:
    """
    Stage 2: Task Suggestion Agent

    Proposes feasible analysis tasks based on available data summaries.

    Example:
        >>> suggester = TaskSuggester(summaries_dir="data/summaries")
        >>> tasks = suggester.suggest_tasks()
        >>> for task in tasks:
        ...     print(task['description'])
    """

    def __init__(
        self,
        summaries_dir: str = "data/summaries",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Task Suggester.

        Args:
            summaries_dir: Directory containing summary JSON files
            config: Configuration dictionary
        """
        self.summaries_dir = Path(summaries_dir)

        self.config = {
            'min_files_for_join': 2,
            'min_key_overlap': 0.5,
            'max_suggestions': 5,
            'llm_provider': 'none',  # 'none', 'openai', 'anthropic'
            'use_llm': False
        }

        if config:
            self.config.update(config)

        # Load summaries
        self.summaries = self._load_summaries()

        logger.info(f"Initialized Task Suggester with {len(self.summaries)} file summaries")

    def _load_summaries(self) -> List[Dict[str, Any]]:
        """Load all summary JSON files."""
        summary_files = sorted(self.summaries_dir.glob("*.summary.json"))

        summaries = []
        for file_path in summary_files:
            try:
                summary = load_json(file_path)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

        return summaries

    def suggest_tasks(self) -> List[Dict[str, Any]]:
        """
        Generate task suggestions based on available data.

        Returns:
            List of task dictionaries

        Example:
            >>> tasks = suggester.suggest_tasks()
            >>> print(f"Generated {len(tasks)} task suggestions")
        """
        if not self.summaries:
            logger.warning("No summaries available for task suggestion")
            return []

        tasks = []

        # Rule-based suggestions
        tasks.extend(self._suggest_prediction_tasks())
        tasks.extend(self._suggest_descriptive_tasks())
        tasks.extend(self._suggest_clustering_tasks())

        # Limit number of suggestions
        tasks = tasks[:self.config['max_suggestions']]

        # Add task IDs
        for i, task in enumerate(tasks, 1):
            task['task_id'] = f"T{i}"

        logger.info(f"Generated {len(tasks)} task suggestions")

        return tasks

    def _suggest_prediction_tasks(self) -> List[Dict[str, Any]]:
        """
        Suggest prediction tasks (regression/classification).

        Looks for:
        - Numeric target variables (yield, production, price)
        - Time-series structure (year column)
        - Candidate features from other files
        """
        tasks = []

        # Common target variables for agricultural data
        target_keywords = [
            'yield', 'production', 'area', 'price', 'income',
            'revenue', 'cost', 'profit', 'temperature', 'rainfall'
        ]

        # Find summaries with potential targets
        for summary in self.summaries:
            file_name = summary['file_name']
            columns = {col['name']: col for col in summary['columns']}

            # Look for numeric columns that could be targets
            for col_name, col_info in columns.items():
                if col_info['type'] == 'numeric':
                    col_lower = col_name.lower()

                    # Check if it matches target keywords
                    is_target = any(kw in col_lower for kw in target_keywords)

                    if is_target:
                        # Check if we have time structure
                        has_year = any('year' in c.lower() for c in columns.keys())

                        # Find potential feature files
                        feature_files = self._find_joinable_files(summary)

                        task = {
                            'type': 'prediction',
                            'subtype': 'regression',
                            'description': f"Predict {col_name} using historical data",
                            'target_variable': col_name,
                            'target_file': file_name,
                            'required_files': [file_name] + feature_files,
                            'required_keys': self._extract_common_keys(
                                [summary] + [s for s in self.summaries if s['file_name'] in feature_files]
                            ),
                            'time_series': has_year,
                            'features': self._suggest_features(summary, feature_files),
                            'feasibility': 'high' if has_year and feature_files else 'medium'
                        }

                        tasks.append(task)

        return tasks

    def _suggest_descriptive_tasks(self) -> List[Dict[str, Any]]:
        """
        Suggest descriptive analysis tasks.

        Looks for:
        - Aggregatable numeric columns
        - Grouping keys (state, crop, year)
        - Ranking/comparison opportunities
        """
        tasks = []

        for summary in self.summaries:
            file_name = summary['file_name']
            columns = {col['name']: col for col in summary['columns']}

            # Find numeric columns
            numeric_cols = [
                col['name'] for col in summary['columns']
                if col['type'] == 'numeric'
            ]

            # Find categorical grouping columns
            grouping_cols = [
                col['name'] for col in summary['columns']
                if col['type'] == 'categorical' and col.get('cardinality', 0) < 100
            ]

            if numeric_cols and grouping_cols:
                # Suggest aggregation task
                for num_col in numeric_cols[:2]:  # Limit to first 2
                    for group_col in grouping_cols[:2]:
                        task = {
                            'type': 'descriptive',
                            'subtype': 'aggregation',
                            'description': f"Analyze {num_col} by {group_col}",
                            'aggregation': 'mean',
                            'target_variable': num_col,
                            'group_by': [group_col],
                            'required_files': [file_name],
                            'required_keys': [group_col],
                            'feasibility': 'high'
                        }

                        tasks.append(task)

            # Suggest ranking task if we have year and numeric column
            has_year = any('year' in c.lower() for c in columns.keys())
            if has_year and numeric_cols and grouping_cols:
                task = {
                    'type': 'descriptive',
                    'subtype': 'ranking',
                    'description': f"Find top {grouping_cols[0]} by {numeric_cols[0]} over time",
                    'aggregation': 'max',
                    'target_variable': numeric_cols[0],
                    'group_by': grouping_cols[:2],
                    'required_files': [file_name],
                    'required_keys': grouping_cols[:2],
                    'feasibility': 'high'
                }

                tasks.append(task)

        return tasks

    def _suggest_clustering_tasks(self) -> List[Dict[str, Any]]:
        """
        Suggest unsupervised clustering tasks.

        Looks for:
        - Multiple numeric features
        - Meaningful grouping dimensions (states, crops)
        """
        tasks = []

        for summary in self.summaries:
            file_name = summary['file_name']

            # Count numeric columns
            numeric_cols = [
                col['name'] for col in summary['columns']
                if col['type'] == 'numeric'
            ]

            # Find entity columns (state, crop, district)
            entity_cols = [
                col['name'] for col in summary['columns']
                if col['type'] == 'categorical' and
                any(kw in col['name'].lower() for kw in ['state', 'crop', 'district', 'region'])
            ]

            # Need at least 3 numeric features for clustering
            if len(numeric_cols) >= 3 and entity_cols:
                task = {
                    'type': 'unsupervised',
                    'subtype': 'clustering',
                    'description': f"Cluster {entity_cols[0]} by {len(numeric_cols)} features",
                    'features': numeric_cols,
                    'entity_column': entity_cols[0],
                    'required_files': [file_name],
                    'required_keys': entity_cols,
                    'n_clusters': 5,
                    'feasibility': 'medium'
                }

                tasks.append(task)

        return tasks

    def _find_joinable_files(self, target_summary: Dict[str, Any]) -> List[str]:
        """
        Find other files that can be joined with the target file.

        Args:
            target_summary: Summary of target file

        Returns:
            List of joinable file names
        """
        target_keys = set(self._extract_keys(target_summary))
        joinable_files = []

        for summary in self.summaries:
            if summary['file_name'] == target_summary['file_name']:
                continue

            other_keys = set(self._extract_keys(summary))

            # Calculate Jaccard similarity
            if target_keys and other_keys:
                overlap = len(target_keys & other_keys)
                union = len(target_keys | other_keys)
                similarity = overlap / union

                if similarity >= self.config['min_key_overlap']:
                    joinable_files.append(summary['file_name'])

        return joinable_files

    def _extract_keys(self, summary: Dict[str, Any]) -> List[str]:
        """Extract all potential key columns from a summary."""
        keys = []

        # From candidate keys
        if 'candidate_keys' in summary:
            keys.extend(summary['candidate_keys'].get('primary', []))
            keys.extend(summary['candidate_keys'].get('foreign', []))

        # Common key column names
        common_keys = ['state', 'year', 'crop', 'district', 'season', 'month']

        for col in summary['columns']:
            col_lower = col['name'].lower()
            if any(key in col_lower for key in common_keys):
                if col['name'] not in keys:
                    keys.append(col['name'])

        # Flatten composite keys
        flat_keys = []
        for key in keys:
            if isinstance(key, list):
                flat_keys.extend(key)
            else:
                flat_keys.append(key)

        return list(set(flat_keys))

    def _extract_common_keys(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Find common keys across multiple summaries."""
        if not summaries:
            return []

        key_sets = [set(self._extract_keys(s)) for s in summaries]
        common_keys = set.intersection(*key_sets) if key_sets else set()

        return sorted(list(common_keys))

    def _suggest_features(
        self,
        target_summary: Dict[str, Any],
        feature_files: List[str]
    ) -> List[str]:
        """Suggest candidate features from joinable files."""
        features = []

        # Get numeric columns from feature files
        for file_name in feature_files:
            summary = next((s for s in self.summaries if s['file_name'] == file_name), None)

            if summary:
                for col in summary['columns']:
                    if col['type'] == 'numeric':
                        features.append(f"{file_name}:{col['name']}")

        return features[:10]  # Limit to 10 features


def main():
    """
    CLI entry point for Stage 2.

    Usage:
        python -m src.stage2.task_suggester
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Task Suggester")
    parser.add_argument(
        '--summaries-dir',
        default='data/summaries',
        help='Directory with summary JSON files'
    )
    parser.add_argument(
        '--output',
        default='data/tasks.json',
        help='Output path for task suggestions'
    )
    parser.add_argument(
        '--max-suggestions',
        type=int,
        default=5,
        help='Maximum number of suggestions'
    )

    args = parser.parse_args()

    # Create suggester
    config = {
        'max_suggestions': args.max_suggestions
    }

    suggester = TaskSuggester(
        summaries_dir=args.summaries_dir,
        config=config
    )

    # Generate suggestions
    tasks = suggester.suggest_tasks()

    # Save to file
    save_json(tasks, args.output)

    print(f"\n✓ Generated {len(tasks)} task suggestions")
    print(f"✓ Saved to: {args.output}\n")

    # Display suggestions
    for task in tasks:
        print(f"[{task['task_id']}] {task['type'].upper()}: {task['description']}")
        print(f"     Files: {', '.join(task['required_files'])}")
        print(f"     Feasibility: {task['feasibility']}")
        print()


if __name__ == '__main__':
    main()
