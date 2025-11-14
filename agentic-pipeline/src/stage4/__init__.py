"""
Stage 4: Executor

Runs approved analyses and produces outputs:
- Predictive models (regression, classification)
- Descriptive analytics (aggregations, rankings)
- Visualizations and reports
- Model cards and metrics
"""

from .executor import Executor
from .models import ModelTrainer
from .visualizer import Visualizer

__all__ = ['Executor', 'ModelTrainer', 'Visualizer']
