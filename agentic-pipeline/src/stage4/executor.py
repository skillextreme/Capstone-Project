"""
Executor - Stage 4

Runs approved analyses and produces final outputs:
- Trains models and generates predictions
- Creates visualizations
- Produces model cards and reports
- Saves all artifacts with provenance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.file_utils import load_parquet, load_json, save_json
from .models import ModelTrainer
from .visualizer import Visualizer

logger = get_logger(__name__)


class Executor:
    """
    Stage 4: Executor

    Runs analyses and produces outputs.

    Example:
        >>> executor = Executor(
        ...     task=task_dict,
        ...     data_path="data/intermediate/T1_merged.parquet"
        ... )
        >>> results = executor.run()
    """

    def __init__(
        self,
        task: Dict[str, Any],
        data_path: str,
        output_dir: str = "data/outputs",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Executor.

        Args:
            task: Task dictionary from Stage 2
            data_path: Path to merged data from Stage 3
            output_dir: Directory for outputs
            config: Configuration dictionary
        """
        self.task = task
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.config = {
            'split': {
                'method': 'time',
                'test_split_year': 2020,
                'test_size': 0.2,
                'random_seed': 42
            },
            'models': {},
            'visualization': {
                'plots': {
                    'actual_vs_predicted': True,
                    'residuals_by_segment': True,
                    'time_series': True,
                    'feature_importance': True,
                    'error_distribution': True
                }
            }
        }

        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value

        # Load data
        self.df = load_parquet(self.data_path)

        logger.info(f"Initialized Executor for task: {task.get('task_id', 'unknown')}")
        logger.info(f"  Loaded data: {len(self.df)} rows, {len(self.df.columns)} columns")

    def run(self) -> Dict[str, Any]:
        """
        Execute the analysis.

        Returns:
            Results dictionary with metrics, predictions, and file paths
        """
        logger.info("Running analysis...")

        task_type = self.task.get('type', 'prediction')

        if task_type == 'prediction':
            results = self._run_prediction()
        elif task_type == 'descriptive':
            results = self._run_descriptive()
        elif task_type == 'unsupervised':
            results = self._run_clustering()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        logger.info("Analysis complete!")

        return results

    def _run_prediction(self) -> Dict[str, Any]:
        """Run prediction (regression) task."""
        logger.info("Running prediction task...")

        # Get target variable
        target_col = self.task.get('target_variable')

        if not target_col or target_col not in self.df.columns:
            raise ValueError(f"Target variable '{target_col}' not found in data")

        # Prepare features and target
        X, y = self._prepare_features_target(target_col)

        # Split data
        X_train, X_test, y_train, y_test, test_df = self._split_data(X, y)

        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")

        # Train models
        trainer = ModelTrainer(config=self.config)
        model_results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        # Get best model
        best_model_name, best_model_result = trainer.get_best_model()

        logger.info(f"Best model: {best_model_name}")

        # Create visualizations
        plots_dir = self.output_dir / "plots"
        visualizer = Visualizer(output_dir=plots_dir)

        plot_files = []

        if self.config['visualization']['plots']['actual_vs_predicted']:
            plot_files.append(visualizer.plot_actual_vs_predicted(
                y_test, best_model_result['predictions']['test'], best_model_name
            ))

        if self.config['visualization']['plots']['residuals_by_segment']:
            residuals = y_test.values - best_model_result['predictions']['test']

            for seg_col in ['state', 'crop']:
                if seg_col in test_df.columns:
                    plot_files.append(visualizer.plot_residuals_by_segment(
                        test_df, seg_col, residuals, best_model_name
                    ))

        if self.config['visualization']['plots']['time_series'] and 'year' in test_df.columns:
            plot_files.append(visualizer.plot_time_series(
                test_df, y_test, best_model_result['predictions']['test'],
                best_model_name, segment_col='state' if 'state' in test_df.columns else None
            ))

        if self.config['visualization']['plots']['feature_importance']:
            plot_files.append(visualizer.plot_feature_importance(
                best_model_result['feature_importance'], best_model_name
            ))

        if self.config['visualization']['plots']['error_distribution']:
            residuals = y_test.values - best_model_result['predictions']['test']
            plot_files.append(visualizer.plot_error_distribution(
                residuals, best_model_name
            ))

        # Save outputs
        task_id = self.task.get('task_id', 'unknown')

        # Save metrics
        metrics_file = self.output_dir / f"{task_id}_metrics.json"
        metrics_output = {
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'models': {
                name: result['metrics']
                for name, result in model_results.items()
            }
        }
        save_json(metrics_output, metrics_file)

        # Save predictions
        predictions_df = test_df.copy()
        predictions_df['actual'] = y_test.values
        predictions_df['predicted'] = best_model_result['predictions']['test']
        predictions_df['residual'] = predictions_df['actual'] - predictions_df['predicted']

        predictions_file = self.output_dir / f"{task_id}_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)

        # Create model card
        model_card = {
            'task_id': task_id,
            'task_description': self.task.get('description'),
            'timestamp': datetime.now().isoformat(),
            'model_name': best_model_name,
            'features': X_train.columns.tolist(),
            'feature_count': len(X_train.columns),
            'target_variable': target_col,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'metrics': best_model_result['metrics']['test'],
            'config': self.config
        }

        model_card_file = self.output_dir / f"{task_id}_model_card.json"
        save_json(model_card, model_card_file)

        logger.info("Saved outputs:")
        logger.info(f"  Metrics: {metrics_file}")
        logger.info(f"  Predictions: {predictions_file}")
        logger.info(f"  Model card: {model_card_file}")
        logger.info(f"  Plots: {len([p for p in plot_files if p])} files")

        return {
            'task_id': task_id,
            'best_model': best_model_name,
            'metrics': best_model_result['metrics'],
            'files': {
                'metrics': str(metrics_file),
                'predictions': str(predictions_file),
                'model_card': str(model_card_file),
                'plots': [str(p) for p in plot_files if p]
            }
        }

    def _run_descriptive(self) -> Dict[str, Any]:
        """Run descriptive analysis task."""
        logger.info("Running descriptive task...")

        task_id = self.task.get('task_id', 'unknown')
        target_col = self.task.get('target_variable')
        group_by = self.task.get('group_by', [])
        aggregation = self.task.get('aggregation', 'mean')

        # Perform aggregation
        if group_by:
            result_df = self.df.groupby(group_by)[target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).reset_index()

            result_df = result_df.sort_values('mean', ascending=False)
        else:
            result_df = self.df[[target_col]].describe()

        # Save results
        output_file = self.output_dir / f"{task_id}_descriptive.csv"
        result_df.to_csv(output_file, index=False)

        logger.info(f"Saved descriptive results to: {output_file}")

        return {
            'task_id': task_id,
            'result': result_df.to_dict('records'),
            'files': {
                'results': str(output_file)
            }
        }

    def _run_clustering(self) -> Dict[str, Any]:
        """Run clustering task."""
        logger.info("Running clustering task...")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        task_id = self.task.get('task_id', 'unknown')
        features = self.task.get('features', [])
        n_clusters = self.task.get('n_clusters', 5)
        entity_col = self.task.get('entity_column')

        # Select features
        feature_cols = [f for f in features if f in self.df.columns]
        X = self.df[feature_cols].fillna(0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Add clusters to dataframe
        result_df = self.df.copy()
        result_df['cluster'] = clusters

        # Save results
        output_file = self.output_dir / f"{task_id}_clusters.csv"
        result_df.to_csv(output_file, index=False)

        # Cluster summary
        summary = result_df.groupby('cluster')[feature_cols].mean()
        summary_file = self.output_dir / f"{task_id}_cluster_summary.csv"
        summary.to_csv(summary_file)

        logger.info(f"Saved clustering results to: {output_file}")

        return {
            'task_id': task_id,
            'n_clusters': n_clusters,
            'files': {
                'results': str(output_file),
                'summary': str(summary_file)
            }
        }

    def _prepare_features_target(self, target_col: str):
        """Prepare feature matrix and target vector."""
        # Drop non-numeric and identifier columns
        drop_cols = [target_col]

        # Add common identifier columns
        id_cols = ['state', 'crop', 'district', 'season']
        drop_cols.extend([c for c in id_cols if c in self.df.columns])

        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])

        # Remove target from features
        X = numeric_df.drop(columns=[c for c in drop_cols if c in numeric_df.columns], errors='ignore')

        # Target
        y = self.df[target_col]

        # Drop rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Fill missing features with median
        X = X.fillna(X.median())

        logger.info(f"Prepared features: {len(X.columns)} features, {len(X)} samples")

        return X, y

    def _split_data(self, X, y):
        """Split data into train and test sets."""
        method = self.config['split']['method']

        if method == 'time' and 'year' in self.df.columns:
            # Time-based split
            split_year = self.config['split']['test_split_year']

            valid_idx = y.notna()
            years = self.df['year'][valid_idx]

            train_mask = years <= split_year
            test_mask = years > split_year

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[valid_idx][train_mask]
            y_test = y[valid_idx][test_mask]

            # Get test DataFrame for visualizations
            test_df = self.df[valid_idx][test_mask]

            logger.info(f"Time split at year {split_year}")

        else:
            # Random split
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['split']['test_size'],
                random_state=self.config['split']['random_seed']
            )

            test_df = self.df.loc[y_test.index]

            logger.info("Random split")

        return X_train, X_test, y_train, y_test, test_df


def main():
    """
    CLI entry point for Stage 4.

    Usage:
        python -m src.stage4.executor --task-file data/tasks.json --task-id T1 --data-path data/intermediate/T1_merged.parquet
    """
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Executor")
    parser.add_argument(
        '--task-file',
        default='data/tasks.json',
        help='Path to tasks JSON file'
    )
    parser.add_argument(
        '--task-id',
        required=True,
        help='Task ID to execute'
    )
    parser.add_argument(
        '--data-path',
        help='Path to merged data (if not following convention)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/outputs',
        help='Directory for outputs'
    )

    args = parser.parse_args()

    # Load tasks
    tasks = load_json(args.task_file)
    task = next((t for t in tasks if t['task_id'] == args.task_id), None)

    if not task:
        print(f"✗ Task {args.task_id} not found")
        return

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = f"data/intermediate/{args.task_id}_merged.parquet"

    print(f"\nExecuting analysis: {task['description']}")

    # Create executor
    executor = Executor(
        task=task,
        data_path=data_path,
        output_dir=args.output_dir
    )

    # Run
    results = executor.run()

    print(f"\n✓ Analysis complete")
    print(f"✓ Best model: {results.get('best_model', 'N/A')}")

    if 'metrics' in results:
        test_metrics = results['metrics'].get('test', {})
        print(f"✓ Test MAE: {test_metrics.get('mae', 0):.4f}")
        print(f"✓ Test R²: {test_metrics.get('r2', 0):.4f}")


if __name__ == '__main__':
    main()
