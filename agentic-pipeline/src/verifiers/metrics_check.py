"""
Verification V4: Metrics Check

Validates outputs from Stage 4:
- Model metrics (MAE, RMSE, R², MAPE)
- Residual diagnostics
- Model transparency (features, config saved)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_json

logger = get_logger(__name__)


class MetricsChecker:
    """
    Verification V4: Metrics and transparency validation.

    Ensures model outputs meet quality standards and are well-documented.

    Example:
        >>> checker = MetricsChecker()
        >>> report = checker.verify_results(
        ...     metrics_path="data/outputs/T1_metrics.json",
        ...     model_card_path="data/outputs/T1_model_card.json"
        ... )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Metrics Checker.

        Args:
            config: Configuration dictionary
        """
        self.config = {
            'min_r2': 0.0,  # Minimum acceptable R²
            'max_mape': 100.0,  # Maximum acceptable MAPE
            'residual_mean_threshold': 0.1,  # Residual mean should be near 0
            'check_metrics': True,
            'check_residuals': True,
            'check_transparency': True
        }

        if config:
            self.config.update(config)

        logger.info("Initialized Metrics Checker (V4)")

    def verify_results(
        self,
        metrics_path: Path,
        model_card_path: Path,
        predictions_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Verify results from Stage 4.

        Args:
            metrics_path: Path to metrics JSON file
            model_card_path: Path to model card JSON file
            predictions_path: Path to predictions CSV (optional)

        Returns:
            Verification report
        """
        logger.info(f"Verifying results: {metrics_path.name}")

        # Load files
        metrics = load_json(metrics_path)
        model_card = load_json(model_card_path)

        predictions = None
        if predictions_path and predictions_path.exists():
            predictions = pd.read_csv(predictions_path)

        # Initialize report
        report = {
            'metrics_file': str(metrics_path),
            'model_card_file': str(model_card_path),
            'task_id': metrics.get('task_id'),
            'timestamp': datetime.now().isoformat(),
            'status': 'pass',
            'errors': [],
            'warnings': [],
            'checks': {}
        }

        # Run checks
        if self.config['check_metrics']:
            self._check_metrics(metrics, report)

        if self.config['check_residuals'] and predictions is not None:
            self._check_residuals(predictions, report)

        if self.config['check_transparency']:
            self._check_transparency(model_card, report)

        # Determine overall status
        if len(report['errors']) > 0:
            report['status'] = 'fail'
        elif len(report['warnings']) > 0:
            report['status'] = 'pass_with_warnings'

        logger.info(f"  Status: {report['status']}")
        logger.info(f"  Errors: {len(report['errors'])}, Warnings: {len(report['warnings'])}")

        return report

    def _check_metrics(
        self,
        metrics: Dict[str, Any],
        report: Dict[str, Any]
    ) -> None:
        """
        Check model metrics.

        Validates that:
        - Metrics are present and reasonable
        - Test metrics are not suspiciously good (possible overfitting indicator)
        - R² is above minimum threshold
        """
        best_model = metrics.get('best_model')

        if not best_model or 'models' not in metrics:
            report['errors'].append({
                'check': 'metrics',
                'type': 'missing_metrics',
                'message': 'No model metrics found'
            })
            report['checks']['metrics'] = {'status': 'fail'}
            return

        model_metrics = metrics['models'].get(best_model, {})
        test_metrics = model_metrics.get('test', {})

        if not test_metrics:
            report['errors'].append({
                'check': 'metrics',
                'type': 'missing_test_metrics',
                'message': 'No test metrics found'
            })
            report['checks']['metrics'] = {'status': 'fail'}
            return

        # Extract metrics
        mae = test_metrics.get('mae')
        rmse = test_metrics.get('rmse')
        r2 = test_metrics.get('r2')
        mape = test_metrics.get('mape')

        report['checks']['metrics'] = {
            'best_model': best_model,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'test_mape': mape,
            'status': 'pass'
        }

        # Check R² threshold
        if r2 is not None and r2 < self.config['min_r2']:
            report['warnings'].append({
                'check': 'metrics',
                'type': 'low_r2',
                'r2': r2,
                'threshold': self.config['min_r2'],
                'message': f'R² ({r2:.3f}) below threshold ({self.config["min_r2"]})'
            })

        # Check MAPE threshold
        if mape is not None and not np.isnan(mape) and mape > self.config['max_mape']:
            report['warnings'].append({
                'check': 'metrics',
                'type': 'high_mape',
                'mape': mape,
                'threshold': self.config['max_mape'],
                'message': f'MAPE ({mape:.1f}%) exceeds threshold ({self.config["max_mape"]}%)'
            })

        # Check for suspiciously good metrics (R² > 0.99 might indicate leakage)
        if r2 is not None and r2 > 0.99:
            report['warnings'].append({
                'check': 'metrics',
                'type': 'suspiciously_high_r2',
                'r2': r2,
                'message': f'R² ({r2:.3f}) is very high - check for data leakage'
            })

        # Check train vs test metrics for overfitting
        train_metrics = model_metrics.get('train', {})
        train_mae = train_metrics.get('mae')
        test_mae = test_metrics.get('mae')

        if train_mae and test_mae:
            mae_ratio = test_mae / train_mae

            if mae_ratio > 2.0:  # Test error > 2x train error
                report['warnings'].append({
                    'check': 'metrics',
                    'type': 'possible_overfitting',
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'ratio': mae_ratio,
                    'message': f'Test MAE is {mae_ratio:.1f}x train MAE - possible overfitting'
                })

    def _check_residuals(
        self,
        predictions: pd.DataFrame,
        report: Dict[str, Any]
    ) -> None:
        """
        Check residual diagnostics.

        Validates that:
        - Residual mean is close to 0
        - Residuals are roughly normally distributed
        - No systematic bias
        """
        if 'residual' not in predictions.columns:
            report['warnings'].append({
                'check': 'residuals',
                'message': 'No residual column found in predictions'
            })
            report['checks']['residuals'] = {'status': 'skipped'}
            return

        residuals = predictions['residual'].dropna()

        if len(residuals) == 0:
            report['warnings'].append({
                'check': 'residuals',
                'message': 'No valid residuals found'
            })
            report['checks']['residuals'] = {'status': 'skipped'}
            return

        # Calculate residual statistics
        residual_mean = float(residuals.mean())
        residual_std = float(residuals.std())
        residual_skew = float(residuals.skew())

        report['checks']['residuals'] = {
            'mean': residual_mean,
            'std': residual_std,
            'skewness': residual_skew,
            'status': 'pass'
        }

        # Check if mean is close to zero (normalized by std)
        if abs(residual_mean / residual_std) > self.config['residual_mean_threshold']:
            report['warnings'].append({
                'check': 'residuals',
                'type': 'non_zero_mean',
                'mean': residual_mean,
                'std': residual_std,
                'message': f'Residual mean ({residual_mean:.4f}) not close to zero - possible bias'
            })

        # Check for severe skewness
        if abs(residual_skew) > 2.0:
            report['warnings'].append({
                'check': 'residuals',
                'type': 'high_skewness',
                'skewness': residual_skew,
                'message': f'Residuals are highly skewed ({residual_skew:.2f}) - check for outliers'
            })

    def _check_transparency(
        self,
        model_card: Dict[str, Any],
        report: Dict[str, Any]
    ) -> None:
        """
        Check model transparency.

        Validates that:
        - Model card contains required fields
        - Features are documented
        - Configuration is saved
        """
        required_fields = [
            'task_id',
            'timestamp',
            'model_name',
            'features',
            'target_variable',
            'train_samples',
            'test_samples',
            'metrics'
        ]

        missing_fields = [f for f in required_fields if f not in model_card]

        if missing_fields:
            report['warnings'].append({
                'check': 'transparency',
                'type': 'missing_fields',
                'missing_fields': missing_fields,
                'message': f'Model card missing fields: {missing_fields}'
            })

        # Check if features are documented
        features = model_card.get('features', [])

        report['checks']['transparency'] = {
            'has_model_card': True,
            'feature_count': len(features),
            'missing_fields': missing_fields,
            'status': 'pass' if not missing_fields else 'warning'
        }

        if not features:
            report['warnings'].append({
                'check': 'transparency',
                'type': 'no_features',
                'message': 'No features documented in model card'
            })

    def verify_all(
        self,
        outputs_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Verify all results in outputs directory.

        Args:
            outputs_dir: Directory with output files

        Returns:
            List of verification reports
        """
        outputs_dir = Path(outputs_dir)

        metrics_files = sorted(outputs_dir.glob("*_metrics.json"))

        if not metrics_files:
            logger.warning(f"No metrics files found in {outputs_dir}")
            return []

        logger.info(f"Verifying {len(metrics_files)} result sets")

        reports = []

        for metrics_path in metrics_files:
            # Find corresponding files
            task_id = metrics_path.stem.replace('_metrics', '')
            model_card_path = outputs_dir / f"{task_id}_model_card.json"
            predictions_path = outputs_dir / f"{task_id}_predictions.csv"

            if not model_card_path.exists():
                logger.warning(f"Model card not found for: {metrics_path.name}")
                continue

            try:
                report = self.verify_results(
                    metrics_path,
                    model_card_path,
                    predictions_path if predictions_path.exists() else None
                )
                reports.append(report)
            except Exception as e:
                logger.error(f"Verification failed for {metrics_path.name}: {e}")

        # Summary statistics
        passed = sum(1 for r in reports if r['status'] == 'pass')
        failed = sum(1 for r in reports if r['status'] == 'fail')
        warnings = sum(1 for r in reports if r['status'] == 'pass_with_warnings')

        logger.info(f"Verification complete: {passed} passed, {warnings} with warnings, {failed} failed")

        return reports


def main():
    """
    CLI entry point for Verification V4.

    Usage:
        python -m src.verifiers.metrics_check
    """
    import argparse
    from utils.file_utils import save_json

    parser = argparse.ArgumentParser(description="Verification V4: Metrics Check")
    parser.add_argument(
        '--outputs-dir',
        default='data/outputs',
        help='Directory with output files'
    )
    parser.add_argument(
        '--output',
        default='data/outputs/verification_v4.json',
        help='Output path for verification report'
    )

    args = parser.parse_args()

    # Create checker
    checker = MetricsChecker()

    # Run verification
    reports = checker.verify_all(args.outputs_dir)

    # Save report
    save_json(reports, args.output)

    print(f"\n✓ Verified {len(reports)} result sets")
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
