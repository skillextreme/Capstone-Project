"""
Stage 2: Task Suggestion Agent

Analyzes file summaries and proposes feasible analysis tasks including:
- Predictive tasks (regression, classification)
- Descriptive tasks (aggregations, lookups)
- Unsupervised tasks (clustering, dimensionality reduction)
"""

from .task_suggester import TaskSuggester

__all__ = ['TaskSuggester']
