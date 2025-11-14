"""
Visualization Module

Creates plots and charts for model results:
- Actual vs Predicted
- Residuals by segment (state, crop)
- Time series plots
- Feature importance
- Error distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Creates visualizations for model results.

    Example:
        >>> viz = Visualizer(output_dir="data/outputs/plots")
        >>> viz.plot_actual_vs_predicted(y_test, y_pred, "model_name")
    """

    def __init__(
        self,
        output_dir: str = "data/outputs/plots",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Visualizer.

        Args:
            output_dir: Directory to save plots
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            'dpi': 300,
            'figsize': (12, 8),
            'style': 'whitegrid'
        }

        if config:
            self.config.update(config)

        logger.info(f"Initialized Visualizer (output: {self.output_dir})")

    def plot_actual_vs_predicted(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> Optional[Path]:
        """
        Plot actual vs predicted values.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot

        Returns:
            Path to saved plot (if save=True)
        """
        fig, ax = plt.subplots(figsize=self.config['figsize'])

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            file_path = self.output_dir / f"actual_vs_pred_{model_name}.png"
            plt.savefig(file_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot: {file_path.name}")
            plt.close()
            return file_path

        return None

    def plot_residuals_by_segment(
        self,
        df: pd.DataFrame,
        segment_col: str,
        residuals: np.ndarray,
        model_name: str,
        top_n: int = 20,
        save: bool = True
    ) -> Optional[Path]:
        """
        Plot residuals by segment (e.g., by state or crop).

        Args:
            df: DataFrame with segment column
            segment_col: Column name for segmentation
            residuals: Residual values
            model_name: Name of the model
            top_n: Number of segments to show
            save: Whether to save the plot

        Returns:
            Path to saved plot (if save=True)
        """
        if segment_col not in df.columns:
            logger.warning(f"Column '{segment_col}' not found - skipping segment plot")
            return None

        # Create dataframe with residuals
        plot_df = pd.DataFrame({
            segment_col: df[segment_col],
            'residual': residuals
        })

        # Calculate mean absolute residual by segment
        segment_errors = plot_df.groupby(segment_col)['residual'].apply(
            lambda x: np.abs(x).mean()
        ).sort_values(ascending=False).head(top_n)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        segment_errors.plot(kind='barh', ax=ax)

        ax.set_xlabel('Mean Absolute Residual', fontsize=12)
        ax.set_ylabel(segment_col.capitalize(), fontsize=12)
        ax.set_title(f'Residuals by {segment_col.capitalize()} - {model_name}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            file_path = self.output_dir / f"residuals_by_{segment_col}_{model_name}.png"
            plt.savefig(file_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot: {file_path.name}")
            plt.close()
            return file_path

        return None

    def plot_time_series(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        time_col: str = 'year',
        segment_col: Optional[str] = None,
        save: bool = True
    ) -> Optional[Path]:
        """
        Plot time series of actual vs predicted values.

        Args:
            df: DataFrame with time and segment columns
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            time_col: Time column name
            segment_col: Optional segment column (e.g., 'state')
            save: Whether to save the plot

        Returns:
            Path to saved plot (if save=True)
        """
        if time_col not in df.columns:
            logger.warning(f"Column '{time_col}' not found - skipping time series plot")
            return None

        plot_df = pd.DataFrame({
            time_col: df[time_col].values,
            'actual': y_true.values,
            'predicted': y_pred
        })

        if segment_col and segment_col in df.columns:
            plot_df[segment_col] = df[segment_col].values

            # Plot top 5 segments
            top_segments = plot_df.groupby(segment_col)['actual'].mean().nlargest(5).index

            fig, axes = plt.subplots(5, 1, figsize=(14, 16))

            for i, segment in enumerate(top_segments):
                segment_df = plot_df[plot_df[segment_col] == segment]

                segment_df = segment_df.groupby(time_col)[['actual', 'predicted']].mean()

                ax = axes[i]
                ax.plot(segment_df.index, segment_df['actual'], 'o-', label='Actual', linewidth=2)
                ax.plot(segment_df.index, segment_df['predicted'], 's--', label='Predicted', linewidth=2)

                ax.set_title(f'{segment}', fontsize=12, fontweight='bold')
                ax.set_xlabel(time_col.capitalize())
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)

            fig.suptitle(f'Time Series by {segment_col.capitalize()} - {model_name}',
                         fontsize=14, fontweight='bold', y=0.995)
        else:
            # Aggregate by time
            time_agg = plot_df.groupby(time_col)[['actual', 'predicted']].mean()

            fig, ax = plt.subplots(figsize=self.config['figsize'])

            ax.plot(time_agg.index, time_agg['actual'], 'o-', label='Actual', linewidth=2)
            ax.plot(time_agg.index, time_agg['predicted'], 's--', label='Predicted', linewidth=2)

            ax.set_xlabel(time_col.capitalize(), fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Time Series - {model_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            file_path = self.output_dir / f"time_series_{model_name}.png"
            plt.savefig(file_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot: {file_path.name}")
            plt.close()
            return file_path

        return None

    def plot_feature_importance(
        self,
        feature_importance: List[Dict[str, Any]],
        model_name: str,
        top_n: int = 20,
        save: bool = True
    ) -> Optional[Path]:
        """
        Plot feature importance.

        Args:
            feature_importance: List of dicts with 'feature' and importance score
            model_name: Name of the model
            top_n: Number of top features to show
            save: Whether to save the plot

        Returns:
            Path to saved plot (if save=True)
        """
        importance_df = pd.DataFrame(feature_importance).head(top_n)

        # Determine importance column name
        imp_col = 'importance' if 'importance' in importance_df.columns else 'coefficient'

        if imp_col not in importance_df.columns:
            logger.warning("No importance column found - skipping feature importance plot")
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        # For coefficients, take absolute value
        if imp_col == 'coefficient':
            importance_df['abs_importance'] = importance_df[imp_col].abs()
            plot_col = 'abs_importance'
            xlabel = 'Absolute Coefficient'
        else:
            plot_col = imp_col
            xlabel = 'Importance Score'

        importance_df = importance_df.sort_values(plot_col)

        ax.barh(range(len(importance_df)), importance_df[plot_col])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save:
            file_path = self.output_dir / f"feature_importance_{model_name}.png"
            plt.savefig(file_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot: {file_path.name}")
            plt.close()
            return file_path

        return None

    def plot_error_distribution(
        self,
        residuals: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> Optional[Path]:
        """
        Plot distribution of residuals.

        Args:
            residuals: Residual values
            model_name: Name of the model
            save: Whether to save the plot

        Returns:
            Path to saved plot (if save=True)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Residual', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Error Analysis - {model_name}', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            file_path = self.output_dir / f"error_distribution_{model_name}.png"
            plt.savefig(file_path, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Saved plot: {file_path.name}")
            plt.close()
            return file_path

        return None
