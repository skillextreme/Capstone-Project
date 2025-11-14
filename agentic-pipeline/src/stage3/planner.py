"""
Planner - Stage 3

Creates reproducible data plans for approved tasks:
1. Normalizes key names and values
2. Builds join graph across multiple files
3. Engineers features (lags, rolling stats, interactions)
4. Saves intermediate tables with provenance

Output: Cleaned, joined dataset ready for modeling (Stage 4)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_csv, save_parquet, save_json
from .normalizer import KeyNormalizer

logger = get_logger(__name__)


class Planner:
    """
    Stage 3: Planner / Join Builder

    Creates a reproducible data plan for an analysis task.

    Example:
        >>> planner = Planner(task=task_dict, data_dir="data/raw")
        >>> merged_df = planner.execute_plan()
    """

    def __init__(
        self,
        task: Dict[str, Any],
        data_dir: str = "data/raw",
        output_dir: str = "data/intermediate",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Planner.

        Args:
            task: Task dictionary from Stage 2
            data_dir: Directory with raw CSV files
            output_dir: Directory for intermediate outputs
            config: Configuration dictionary
        """
        self.task = task
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.config = {
            'join_type': 'left',
            'drop_na_threshold': 0.5,
            'create_lags': True,
            'lag_periods': [1, 2],
            'create_rolling': True,
            'rolling_windows': [3, 5],
            'create_interactions': False,
            'outlier_method': 'iqr',
            'key_normalizations': {}
        }

        if config:
            self.config.update(config)

        # Initialize normalizer
        self.normalizer = KeyNormalizer(config=self.config.get('key_normalizations', {}))

        logger.info(f"Initialized Planner for task: {task.get('task_id', 'unknown')}")
        logger.info(f"  Type: {task.get('type')}")
        logger.info(f"  Files required: {len(task.get('required_files', []))}")

    def execute_plan(self) -> pd.DataFrame:
        """
        Execute the complete data plan.

        Returns:
            Merged and feature-engineered DataFrame

        Example:
            >>> df = planner.execute_plan()
            >>> print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        """
        logger.info("Executing data plan...")

        # Step 1: Load and normalize files
        dataframes = self._load_and_normalize_files()

        # Step 2: Build join graph and merge
        merged_df = self._merge_files(dataframes)

        # Step 3: Engineer features
        merged_df = self._engineer_features(merged_df)

        # Step 4: Handle missing values
        merged_df = self._handle_missing_values(merged_df)

        # Step 5: Handle outliers
        merged_df = self._handle_outliers(merged_df)

        # Step 6: Save intermediate table and provenance
        self._save_outputs(merged_df)

        logger.info(f"Plan execution complete: {len(merged_df)} rows, {len(merged_df.columns)} columns")

        return merged_df

    def _load_and_normalize_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required files and normalize key columns.

        Returns:
            Dictionary of {file_name: DataFrame}
        """
        required_files = self.task.get('required_files', [])
        dataframes = {}

        logger.info(f"Loading {len(required_files)} files...")

        for file_name in required_files:
            file_path = self.data_dir / file_name

            if not file_path.exists():
                logger.error(f"Required file not found: {file_name}")
                continue

            # Load file
            df = load_csv(file_path)

            # Normalize key columns
            key_columns = self.task.get('required_keys', [])
            df = self.normalizer.normalize_all(df, columns=key_columns)

            # Store with standardized name
            dataframes[file_name] = df

            logger.info(f"  Loaded {file_name}: {len(df)} rows, {len(df.columns)} columns")

        return dataframes

    def _merge_files(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple dataframes based on common keys.

        Args:
            dataframes: Dict of {file_name: DataFrame}

        Returns:
            Merged DataFrame
        """
        if len(dataframes) == 0:
            raise ValueError("No dataframes to merge")

        if len(dataframes) == 1:
            logger.info("Single file - no joins required")
            return list(dataframes.values())[0].copy()

        logger.info(f"Merging {len(dataframes)} files...")

        # Determine join keys
        join_keys = self.task.get('required_keys', [])

        if not join_keys:
            logger.warning("No join keys specified - attempting to infer...")
            join_keys = self._infer_join_keys(dataframes)

        logger.info(f"  Join keys: {join_keys}")

        # Start with first dataframe
        file_names = list(dataframes.keys())
        merged_df = dataframes[file_names[0]].copy()

        # Progressively merge others
        for file_name in file_names[1:]:
            df_to_merge = dataframes[file_name]

            # Find common columns between merged_df and df_to_merge
            common_keys = [k for k in join_keys if k in merged_df.columns and k in df_to_merge.columns]

            if not common_keys:
                logger.warning(f"No common keys with {file_name} - skipping")
                continue

            # Perform join
            before_rows = len(merged_df)

            merged_df = pd.merge(
                merged_df,
                df_to_merge,
                on=common_keys,
                how=self.config['join_type'],
                suffixes=('', f'_{file_name.split(".")[0]}')
            )

            after_rows = len(merged_df)

            logger.info(f"  Joined with {file_name}: {before_rows} -> {after_rows} rows")

            # Check for join explosion
            if after_rows > before_rows * 1.5:
                logger.warning(f"  ⚠ Join explosion detected! Rows increased by {after_rows/before_rows:.1f}x")

        return merged_df

    def _infer_join_keys(self, dataframes: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Infer common join keys across dataframes.

        Args:
            dataframes: Dict of DataFrames

        Returns:
            List of common column names
        """
        if not dataframes:
            return []

        # Find columns common to all dataframes
        column_sets = [set(df.columns) for df in dataframes.values()]
        common_columns = set.intersection(*column_sets)

        # Prioritize typical key columns
        priority_keys = ['state', 'year', 'crop', 'district', 'season']

        inferred_keys = []
        for key in priority_keys:
            if key in common_columns or any(k.lower() == key for k in common_columns):
                # Find actual column name (case-insensitive match)
                actual_key = next((c for c in common_columns if c.lower() == key), None)
                if actual_key:
                    inferred_keys.append(actual_key)

        return inferred_keys

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on task type and configuration.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")

        # Create lagged features (for time series)
        if self.config['create_lags'] and self.task.get('time_series', False):
            df = self._create_lag_features(df)

        # Create rolling statistics
        if self.config['create_rolling'] and self.task.get('time_series', False):
            df = self._create_rolling_features(df)

        # Create interaction features
        if self.config['create_interactions']:
            df = self._create_interaction_features(df)

        logger.info(f"  Final feature count: {len(df.columns)}")

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of numeric columns."""
        if 'year' not in df.columns:
            logger.warning("No 'year' column - skipping lag features")
            return df

        # Get target variable
        target_col = self.task.get('target_variable')

        if not target_col or target_col not in df.columns:
            return df

        # Get grouping columns (state, crop)
        group_cols = [c for c in ['state', 'crop', 'district'] if c in df.columns]

        if not group_cols:
            logger.warning("No grouping columns - skipping lag features")
            return df

        logger.info(f"  Creating lag features for '{target_col}'...")

        # Sort by group and year
        df = df.sort_values(group_cols + ['year'])

        # Create lags
        for lag in self.config['lag_periods']:
            lag_col_name = f"{target_col}_lag{lag}"

            df[lag_col_name] = df.groupby(group_cols)[target_col].shift(lag)

            logger.info(f"    Created {lag_col_name}")

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics."""
        if 'year' not in df.columns:
            return df

        target_col = self.task.get('target_variable')

        if not target_col or target_col not in df.columns:
            return df

        group_cols = [c for c in ['state', 'crop', 'district'] if c in df.columns]

        if not group_cols:
            return df

        logger.info(f"  Creating rolling features for '{target_col}'...")

        # Sort by group and year
        df = df.sort_values(group_cols + ['year'])

        # Create rolling windows
        for window in self.config['rolling_windows']:
            # Rolling mean
            roll_mean_col = f"{target_col}_roll{window}_mean"
            df[roll_mean_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            roll_std_col = f"{target_col}_roll{window}_std"
            df[roll_std_col] = df.groupby(group_cols)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

            logger.info(f"    Created {roll_mean_col}, {roll_std_col}")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to avoid explosion
        numeric_cols = numeric_cols[:5]

        if len(numeric_cols) < 2:
            return df

        logger.info(f"  Creating interaction features...")

        # Create pairwise products
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interaction_col = f"{col1}_x_{col2}"
                df[interaction_col] = df[col1] * df[col2]

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on threshold."""
        logger.info("Handling missing values...")

        before_rows = len(df)

        # Drop rows with too many missing values
        missing_ratio = df.isna().sum(axis=1) / len(df.columns)
        df = df[missing_ratio <= self.config['drop_na_threshold']]

        after_rows = len(df)
        dropped = before_rows - after_rows

        if dropped > 0:
            logger.info(f"  Dropped {dropped} rows with >{self.config['drop_na_threshold']:.0%} missing values")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        # For now, just log outlier counts
        # More sophisticated handling can be added based on use case

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        logger.info(f"Checking for outliers in {len(numeric_cols)} numeric columns...")

        return df

    def _save_outputs(self, df: pd.DataFrame):
        """Save intermediate table and provenance information."""
        task_id = self.task.get('task_id', 'unknown')

        # Save merged data
        data_file = self.output_dir / f"{task_id}_merged.parquet"
        save_parquet(df, data_file)

        # Save join plan
        plan = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'task': self.task,
            'files_used': self.task.get('required_files', []),
            'join_keys': self.task.get('required_keys', []),
            'join_type': self.config['join_type'],
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist()
        }

        plan_file = self.output_dir / f"{task_id}_plan.json"
        save_json(plan, plan_file)

        logger.info(f"Saved outputs:")
        logger.info(f"  Data: {data_file}")
        logger.info(f"  Plan: {plan_file}")


def main():
    """
    CLI entry point for Stage 3.

    Usage:
        python -m src.stage3.planner --task-file data/tasks.json --task-id T1
    """
    import argparse
    from utils.file_utils import load_json

    parser = argparse.ArgumentParser(description="Stage 3: Planner / Join Builder")
    parser.add_argument(
        '--task-file',
        default='data/tasks.json',
        help='Path to tasks JSON file'
    )
    parser.add_argument(
        '--task-id',
        required=True,
        help='Task ID to execute (e.g., T1)'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory with raw CSV files'
    )
    parser.add_argument(
        '--output-dir',
        default='data/intermediate',
        help='Directory for intermediate outputs'
    )

    args = parser.parse_args()

    # Load tasks
    tasks = load_json(args.task_file)

    # Find requested task
    task = next((t for t in tasks if t['task_id'] == args.task_id), None)

    if not task:
        print(f"✗ Task {args.task_id} not found")
        return

    print(f"\nExecuting task: {task['description']}")

    # Create planner
    planner = Planner(
        task=task,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    # Execute
    df = planner.execute_plan()

    print(f"\n✓ Plan executed successfully")
    print(f"✓ Final dataset: {len(df)} rows, {len(df.columns)} columns")


if __name__ == '__main__':
    main()
