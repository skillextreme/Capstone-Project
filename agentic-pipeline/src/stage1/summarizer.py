"""
Summarizer Agent - Stage 1

Analyzes raw CSV/JSON files and generates comprehensive summaries including:
- Column names, types, and statistics
- Candidate primary and foreign keys
- Data quality metrics (null rates, value distributions)
- Sample values and ranges

Output: JSON summary files saved to data/summaries/
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_csv, load_json, save_json, get_file_list
from utils.stats_utils import (
    infer_column_type,
    calculate_basic_stats,
    check_key_candidate
)

logger = get_logger(__name__)


class Summarizer:
    """
    Stage 1: Summarizer Agent

    Analyzes raw data files and generates factual summaries.

    Attributes:
        data_dir: Directory containing raw CSV/JSON files
        output_dir: Directory to save summary JSON files
        config: Configuration dictionary with settings

    Example:
        >>> summarizer = Summarizer(
        ...     data_dir="data/raw",
        ...     output_dir="data/summaries",
        ...     config={"sample_size": 1000}
        ... )
        >>> summaries = summarizer.run_all()
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/summaries",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Summarizer.

        Args:
            data_dir: Path to directory with raw data files
            output_dir: Path to directory for output summaries
            config: Configuration dict (from pipeline_config.yaml)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        self.config = {
            'sample_size': None,  # None = use all rows
            'numeric_threshold': 0.95,
            'min_categorical_cardinality': 2,
            'max_categorical_cardinality': 100,
            'top_k_values': 10,
            'key_uniqueness_threshold': 0.95,
            'common_keys': ['state', 'year', 'crop', 'district', 'season', 'month']
        }

        # Override with user config
        if config:
            self.config.update(config)

        logger.info(f"Initialized Summarizer")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Output dir: {self.output_dir}")

    def summarize_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Generate a comprehensive summary for a single file.

        Args:
            file_path: Path to CSV or JSON file

        Returns:
            Summary dictionary with schema, stats, and key candidates

        Example:
            >>> summary = summarizer.summarize_file(Path("data/raw/yields.csv"))
            >>> print(summary['row_count'])
            15000
        """
        logger.info(f"Summarizing file: {file_path.name}")

        # Load data
        if file_path.suffix == '.csv':
            df = load_csv(file_path, sample_size=self.config['sample_size'])
        elif file_path.suffix == '.json':
            data = load_json(file_path)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Basic file info
        summary = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': [],
            'candidate_keys': {
                'primary': [],
                'foreign': []
            },
            'data_quality': {
                'total_cells': len(df) * len(df.columns),
                'null_cells': int(df.isna().sum().sum()),
                'null_rate': float(df.isna().sum().sum() / (len(df) * len(df.columns)))
            }
        }

        # Analyze each column
        for col_name in df.columns:
            col_summary = self._summarize_column(df[col_name], col_name)
            summary['columns'].append(col_summary)

        # Detect candidate keys
        summary['candidate_keys'] = self._detect_keys(df)

        logger.info(f"  Analyzed {summary['column_count']} columns, {summary['row_count']} rows")

        return summary

    def _summarize_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Summarize a single column.

        Args:
            series: Pandas Series to analyze
            col_name: Column name

        Returns:
            Column summary dictionary
        """
        # Infer type
        col_type = infer_column_type(
            series,
            numeric_threshold=self.config['numeric_threshold'],
            max_categorical_cardinality=self.config['max_categorical_cardinality']
        )

        # Calculate basic stats
        stats = calculate_basic_stats(series, col_type)

        # Check if key candidate
        is_key, uniqueness = check_key_candidate(
            series,
            uniqueness_threshold=self.config['key_uniqueness_threshold']
        )

        # Build column summary
        col_summary = {
            'name': col_name,
            'type': col_type,
            'is_key_candidate': is_key,
            'uniqueness_ratio': uniqueness,
            **stats
        }

        # Add sample values for non-numeric columns
        if col_type != 'numeric':
            non_null = series.dropna()
            if len(non_null) > 0:
                col_summary['sample_values'] = non_null.head(5).tolist()

        return col_summary

    def _detect_keys(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect candidate primary and foreign keys.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with 'primary' and 'foreign' key lists
        """
        primary_keys = []
        foreign_keys = []

        # Check individual columns
        for col in df.columns:
            is_key, uniqueness = check_key_candidate(
                df[col],
                uniqueness_threshold=self.config['key_uniqueness_threshold']
            )

            if is_key:
                primary_keys.append(col)
            elif col.lower() in [k.lower() for k in self.config['common_keys']]:
                foreign_keys.append(col)

        # Check composite keys (common combinations)
        common_composites = [
            ['state', 'year'],
            ['state', 'year', 'crop'],
            ['state', 'district', 'year'],
            ['year', 'crop']
        ]

        for key_combo in common_composites:
            # Check if all columns exist (case-insensitive)
            col_map = {c.lower(): c for c in df.columns}
            actual_combo = []

            for key in key_combo:
                if key.lower() in col_map:
                    actual_combo.append(col_map[key.lower()])

            if len(actual_combo) == len(key_combo):
                # Check uniqueness of composite key
                is_unique = df[actual_combo].duplicated().sum() == 0

                if is_unique and actual_combo not in primary_keys:
                    primary_keys.append(actual_combo)

        return {
            'primary': primary_keys,
            'foreign': foreign_keys
        }

    def run_all(self) -> List[Dict[str, Any]]:
        """
        Process all CSV and JSON files in the data directory.

        Returns:
            List of summary dictionaries

        Example:
            >>> summaries = summarizer.run_all()
            >>> print(f"Processed {len(summaries)} files")
        """
        # Get all CSV and JSON files
        csv_files = get_file_list(self.data_dir, "*.csv")
        json_files = get_file_list(self.data_dir, "*.json")
        all_files = csv_files + json_files

        if not all_files:
            logger.warning(f"No CSV or JSON files found in {self.data_dir}")
            return []

        logger.info(f"Found {len(all_files)} files to process")

        summaries = []

        # Process each file
        for file_path in tqdm(all_files, desc="Summarizing files"):
            try:
                summary = self.summarize_file(file_path)
                summaries.append(summary)

                # Save summary to JSON
                output_file = self.output_dir / f"{file_path.stem}.summary.json"
                save_json(summary, output_file)

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        logger.info(f"Successfully processed {len(summaries)} files")

        # Save combined index
        index_file = self.output_dir / "summaries_index.json"
        index = {
            'total_files': len(summaries),
            'files': [s['file_name'] for s in summaries],
            'summaries': summaries
        }
        save_json(index, index_file)

        return summaries


def main():
    """
    CLI entry point for Stage 1.

    Usage:
        python -m src.stage1.summarizer
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Summarizer Agent")
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory with raw data files'
    )
    parser.add_argument(
        '--output-dir',
        default='data/summaries',
        help='Directory for output summaries'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of rows to sample (None = all)'
    )

    args = parser.parse_args()

    # Create summarizer
    config = {
        'sample_size': args.sample_size
    }

    summarizer = Summarizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config
    )

    # Run
    summaries = summarizer.run_all()

    print(f"\n✓ Summarized {len(summaries)} files")
    print(f"✓ Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
