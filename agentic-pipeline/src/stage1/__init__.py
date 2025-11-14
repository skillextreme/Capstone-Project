"""
Stage 1: Summarizer Agent

Generates factual, non-opinionated summaries of raw data files.
Extracts schema information, statistics, and candidate keys.
"""

from .summarizer import Summarizer

__all__ = ['Summarizer']
