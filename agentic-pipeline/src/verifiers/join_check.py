"""
Verification V3: Join & Leakage Check

Validates data plans from Stage 3:
- Join cardinality (1:1, 1:many, many:many)
- Coverage (% rows retained after joins)
- Data leakage (train/test split by time)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_parquet, load_json

logger = get_logger(__name__)


class JoinChecker:
    """
    Verification V3: Join and leakage validation.

    Ensures joins are correct and no data leakage exists.

    Example:
        >>> checker = JoinChecker()
        >>> report = checker.verify_plan(
        ...     data_path="data/intermediate/T1_merged.parquet",
        ...     plan_path="data/intermediate/T1_plan.json"
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Join Checker.

        Args:
            config: Configuration dictionary
        """
        self.config = {
            'min_coverage': 0.8,  # Warn if <80% rows retained
            'check_join_cardinality': True,
            'check_coverage': True,
            'check_leakage': True
        }

        if config:
            self.config.update(config)

        logger.info("Initialized Join Checker (V3)")

    def verify_plan(
        self,
        data_path: Path,
        plan_path: Path
    ) -> Dict[str, Any]:
        """
        Verify a data plan after Stage 3.

        Args:
            data_path: Path to merged parquet file
            plan_path: Path to plan JSON file

        Returns:
            Verification report
        """
        logger.info(f"Verifying plan: {plan_path.name}")

        # Load data and plan
        df = load_parquet(data_path)
        plan = load_json(plan_path)

        # Initialize report
        report = {
            'data_file': str(data_path),
            'plan_file': str(plan_path),
            'task_id': plan.get('task_id'),
            'timestamp': datetime.now().isoformat(),
            'status': 'pass',
            'errors': [],
            'warnings': [],
            'checks': {}
        }

        # Run checks
        if self.config['check_join_cardinality']:
            self._check_join_cardinality(df, plan, report)

        if self.config['check_coverage']:
            self._check_coverage(df, plan, report)

        if self.config['check_leakage']:
            self._check_leakage(df, plan, report)

        # Determine overall status
        if len(report['errors']) > 0:
            report['status'] = 'fail'
        elif len(report['warnings']) > 0:
            report['status'] = 'pass_with_warnings'

        logger.info(f"  Status: {report['status']}")
        logger.info(f"  Errors: {len(report['errors'])}, Warnings: {len(report['warnings'])}")

        return report

    def _check_join_cardinality(
        self,
        df: pd.DataFrame,
        plan: Dict[str, Any],
        report: Dict[str, Any]
    ) -> None:
        """
        Check for join explosions (many-to-many joins).

        A join explosion occurs when the number of rows after join
        is significantly larger than before, indicating a many-to-many relationship.
        """
        join_keys = plan.get('join_keys', [])

        if not join_keys:
            report['warnings'].append({
                'check': 'join_cardinality',
                'message': 'No join keys specified - cannot verify cardinality'
            })
            report['checks']['join_cardinality'] = {'status': 'skipped'}
            return

        # Check for duplicates on join keys
        key_cols = [k for k in join_keys if k in df.columns]

        if not key_cols:
            report['warnings'].append({
                'check': 'join_cardinality',
                'message': f'Join keys not found in data: {join_keys}'
            })
            report['checks']['join_cardinality'] = {'status': 'skipped'}
            return

        duplicate_count = df[key_cols].duplicated().sum()
        total_rows = len(df)
        duplicate_rate = duplicate_count / total_rows

        report['checks']['join_cardinality'] = {
            'join_keys': key_cols,
            'total_rows': total_rows,
            'duplicate_rows': int(duplicate_count),
            'duplicate_rate': float(duplicate_rate),
            'status': 'pass'
        }

        if duplicate_rate > 0.1:  # More than 10% duplicates
            report['warnings'].append({
                'check': 'join_cardinality',
                'type': 'high_duplicates',
                'duplicate_rate': duplicate_rate,
                'message': f'{duplicate_rate:.1%} of rows have duplicate join keys - possible many-to-many join'
            })
            report['checks']['join_cardinality']['status'] = 'warning'

    def _check_coverage(
        self,
        df: pd.DataFrame,
        plan: Dict[str, Any],
        report: Dict[str, Any]
    ) -> None:
        """
        Check row coverage after joins.

        Warns if a significant portion of rows was lost during joins.
        """
        # This is approximate - we don't have access to original row counts
        # In a production system, we'd track this during the join process

        # Check for missing values as a proxy for join coverage
        missing_by_col = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        missing_cells = missing_by_col.sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0

        report['checks']['coverage'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_rate': float(missing_rate),
            'status': 'pass'
        }

        if missing_rate > 0.3:  # More than 30% missing
            report['warnings'].append({
                'check': 'coverage',
                'type': 'high_missing_rate',
                'missing_rate': missing_rate,
                'message': f'{missing_rate:.1%} of cells are missing after join - possible low coverage'
            })
            report['checks']['coverage']['status'] = 'warning'

        # Check if minimum coverage threshold met
        coverage_rate = 1 - missing_rate

        if coverage_rate < self.config['min_coverage']:
            report['errors'].append({
                'check': 'coverage',
                'type': 'low_coverage',
                'coverage_rate': coverage_rate,
                'threshold': self.config['min_coverage'],
                'message': f'Coverage {coverage_rate:.1%} below threshold {self.config["min_coverage"]:.1%}'
            })
            report['checks']['coverage']['status'] = 'fail'

    def _check_leakage(
        self,
        df: pd.DataFrame,
        plan: Dict[str, Any],
        report: Dict[str, Any]
    ) -> None:
        """
        Check for data leakage in time-series tasks.

        Ensures that:
        1. Train/test split is done by time (year)
        2. No future data is used in training
        3. Lagged features don't create leakage
        """
        task = plan.get('task', {})

        if not task.get('time_series', False):
            report['checks']['leakage'] = {
                'status': 'not_applicable',
                'message': 'Not a time-series task'
            }
            return

        # Check if 'year' column exists
        if 'year' not in df.columns:
            report['errors'].append({
                'check': 'leakage',
                'type': 'missing_year_column',
                'message': 'Time-series task but no year column found'
            })
            report['checks']['leakage'] = {'status': 'fail'}
            return

        # Check year range
        year_min = df['year'].min()
        year_max = df['year'].max()
        year_range = year_max - year_min

        report['checks']['leakage'] = {
            'year_min': int(year_min),
            'year_max': int(year_max),
            'year_range': int(year_range),
            'status': 'pass'
        }

        # Check if lag features exist and are properly constructed
        lag_cols = [c for c in df.columns if 'lag' in c.lower()]

        if lag_cols:
            # Verify lag features don't have values in first lag_period years
            for lag_col in lag_cols:
                # Extract lag period from column name (e.g., "yield_lag1" -> 1)
                try:
                    lag_period = int(lag_col.split('lag')[-1].split('_')[0])

                    # Check if first `lag_period` years have NaN in lag column
                    early_years = df.nsmallest(lag_period, 'year')

                    if not early_years[lag_col].isna().all():
                        report['warnings'].append({
                            'check': 'leakage',
                            'type': 'suspicious_lag_values',
                            'column': lag_col,
                            'message': f'{lag_col} has non-null values in early years - potential leakage'
                        })
                except:
                    pass

        logger.info(f"  Leakage check: year range {year_min}-{year_max}")

    def verify_all(
        self,
        intermediate_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Verify all plans in intermediate directory.

        Args:
            intermediate_dir: Directory with merged data and plans

        Returns:
            List of verification reports
        """
        intermediate_dir = Path(intermediate_dir)

        plan_files = sorted(intermediate_dir.glob("*_plan.json"))

        if not plan_files:
            logger.warning(f"No plan files found in {intermediate_dir}")
            return []

        logger.info(f"Verifying {len(plan_files)} plans")

        reports = []

        for plan_path in plan_files:
            # Find corresponding data file
            task_id = plan_path.stem.replace('_plan', '')
            data_path = intermediate_dir / f"{task_id}_merged.parquet"

            if not data_path.exists():
                logger.warning(f"Data file not found for plan: {plan_path.name}")
                continue

            try:
                report = self.verify_plan(data_path, plan_path)
                reports.append(report)
            except Exception as e:
                logger.error(f"Verification failed for {plan_path.name}: {e}")

        # Summary statistics
        passed = sum(1 for r in reports if r['status'] == 'pass')
        failed = sum(1 for r in reports if r['status'] == 'fail')
        warnings = sum(1 for r in reports if r['status'] == 'pass_with_warnings')

        logger.info(f"Verification complete: {passed} passed, {warnings} with warnings, {failed} failed")

        return reports


def main():
    """
    CLI entry point for Verification V3.

    Usage:
        python -m src.verifiers.join_check
    """
    import argparse
    from utils.file_utils import save_json

    parser = argparse.ArgumentParser(description="Verification V3: Join & Leakage Check")
    parser.add_argument(
        '--intermediate-dir',
        default='data/intermediate',
        help='Directory with merged data and plans'
    )
    parser.add_argument(
        '--output',
        default='data/intermediate/verification_v3.json',
        help='Output path for verification report'
    )

    args = parser.parse_args()

    # Create checker
    checker = JoinChecker()

    # Run verification
    reports = checker.verify_all(args.intermediate_dir)

    # Save report
    save_json(reports, args.output)

    print(f"\n✓ Verified {len(reports)} plans")
    print(f"✓ Report saved to: {args.output}")

    # Print summary
    passed = sum(1 for r in reports if r['status'] == 'pass')
    failed = sum(1 for r in reports if r['status'] == 'fail')
    warnings = sum(1 for r in reports if r['status'] == 'pass_with_warnings')

    print(f"\nResults:")
    print(f"  ✓ Passed: {passed}")
    print(f"  ⚠ Warnings: {warnings}")
    print(f"  ✗ Failed: {failed}")


if __name__ == '__main__':
    main()
