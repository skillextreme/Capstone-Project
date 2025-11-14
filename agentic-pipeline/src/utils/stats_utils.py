"""
Statistical utilities for the agentic pipeline.
Provides functions for type inference, basic statistics, and outlier detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from .logging_utils import get_logger

logger = get_logger(__name__)


def infer_column_type(
    series: pd.Series,
    numeric_threshold: float = 0.95,
    max_categorical_cardinality: int = 100
) -> str:
    """
    Infer the type of a pandas Series.

    Args:
        series: Pandas Series to analyze
        numeric_threshold: Fraction of values that must parse as numeric
        max_categorical_cardinality: Max unique values for categorical

    Returns:
        One of: 'numeric', 'categorical', 'date', 'string', 'boolean'

    Example:
        >>> s = pd.Series(['2020', '2021', '2022', '2023'])
        >>> infer_column_type(s)
        'numeric'
    """
    # Handle empty series
    if len(series) == 0:
        return 'string'

    # Drop nulls for analysis
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'string'

    # Check if boolean
    unique_values = non_null.unique()
    if len(unique_values) == 2:
        if set(unique_values).issubset({True, False, 1, 0, 'true', 'false', 'True', 'False'}):
            return 'boolean'

    # Check if already numeric type
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'

    # Try to parse as numeric
    try:
        parsed = pd.to_numeric(non_null, errors='coerce')
        numeric_ratio = parsed.notna().sum() / len(non_null)

        if numeric_ratio >= numeric_threshold:
            return 'numeric'
    except:
        pass

    # Try to parse as datetime
    try:
        pd.to_datetime(non_null, errors='raise')
        return 'date'
    except:
        pass

    # Check cardinality for categorical vs string
    cardinality = len(unique_values)

    if cardinality <= max_categorical_cardinality:
        return 'categorical'
    else:
        return 'string'


def calculate_basic_stats(series: pd.Series, col_type: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for a Series based on its type.

    Args:
        series: Pandas Series to analyze
        col_type: Column type ('numeric', 'categorical', etc.)

    Returns:
        Dictionary of statistics

    Example:
        >>> s = pd.Series([10, 20, 30, 40, 50])
        >>> stats = calculate_basic_stats(s, 'numeric')
        >>> print(stats['mean'])
        30.0
    """
    stats = {
        'null_count': int(series.isna().sum()),
        'null_rate': float(series.isna().mean()),
        'total_count': len(series)
    }

    non_null = series.dropna()

    if col_type == 'numeric':
        # Numeric statistics
        try:
            numeric_series = pd.to_numeric(non_null, errors='coerce')
            stats.update({
                'mean': float(numeric_series.mean()),
                'std': float(numeric_series.std()),
                'min': float(numeric_series.min()),
                'max': float(numeric_series.max()),
                'median': float(numeric_series.median()),
                'q25': float(numeric_series.quantile(0.25)),
                'q75': float(numeric_series.quantile(0.75))
            })
        except Exception as e:
            logger.warning(f"Error calculating numeric stats: {e}")

    elif col_type == 'categorical':
        # Categorical statistics
        value_counts = non_null.value_counts()
        stats.update({
            'cardinality': len(value_counts),
            'top_values': value_counts.head(10).to_dict(),
            'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None
        })

    elif col_type == 'date':
        # Date statistics
        try:
            date_series = pd.to_datetime(non_null)
            stats.update({
                'min_date': str(date_series.min()),
                'max_date': str(date_series.max()),
                'date_range_days': (date_series.max() - date_series.min()).days
            })
        except Exception as e:
            logger.warning(f"Error calculating date stats: {e}")

    else:
        # String statistics
        stats.update({
            'unique_count': len(non_null.unique()),
            'sample_values': non_null.head(5).tolist()
        })

    return stats


def detect_outliers(
    series: pd.Series,
    method: str = 'iqr',
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detect outliers in a numeric Series.

    Args:
        series: Numeric Series to analyze
        method: 'iqr' or 'zscore'
        iqr_multiplier: Multiplier for IQR method (default: 1.5)
        zscore_threshold: Threshold for z-score method (default: 3.0)

    Returns:
        Tuple of (outlier_mask, outlier_info)
        - outlier_mask: Boolean Series (True = outlier)
        - outlier_info: Dict with outlier statistics

    Example:
        >>> s = pd.Series([1, 2, 3, 100, 4, 5])
        >>> mask, info = detect_outliers(s, method='iqr')
        >>> print(f"Found {info['outlier_count']} outliers")
    """
    # Ensure numeric
    numeric_series = pd.to_numeric(series, errors='coerce')

    if method == 'iqr':
        # IQR method
        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)

        info = {
            'method': 'iqr',
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'iqr': float(iqr),
            'outlier_count': int(outlier_mask.sum()),
            'outlier_rate': float(outlier_mask.mean())
        }

    elif method == 'zscore':
        # Z-score method
        mean = numeric_series.mean()
        std = numeric_series.std()

        z_scores = np.abs((numeric_series - mean) / std)
        outlier_mask = z_scores > zscore_threshold

        info = {
            'method': 'zscore',
            'mean': float(mean),
            'std': float(std),
            'threshold': zscore_threshold,
            'outlier_count': int(outlier_mask.sum()),
            'outlier_rate': float(outlier_mask.mean())
        }

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outlier_mask, info


def check_key_candidate(
    series: pd.Series,
    uniqueness_threshold: float = 0.95
) -> Tuple[bool, float]:
    """
    Check if a column could be a primary key.

    Args:
        series: Series to check
        uniqueness_threshold: Minimum uniqueness ratio

    Returns:
        Tuple of (is_key_candidate, uniqueness_ratio)

    Example:
        >>> s = pd.Series(['A', 'B', 'C', 'D', 'E'])
        >>> is_key, ratio = check_key_candidate(s)
        >>> print(f"Key candidate: {is_key}, Uniqueness: {ratio:.2f}")
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return False, 0.0

    unique_count = len(non_null.unique())
    total_count = len(non_null)
    uniqueness_ratio = unique_count / total_count

    is_candidate = uniqueness_ratio >= uniqueness_threshold

    return is_candidate, uniqueness_ratio


def normalize_string_column(
    series: pd.Series,
    normalization_map: Optional[Dict[str, str]] = None
) -> pd.Series:
    """
    Normalize string values in a Series.

    Args:
        series: Series to normalize
        normalization_map: Dict mapping variants to standard form

    Returns:
        Normalized Series

    Example:
        >>> s = pd.Series(['UP', 'U.P.', 'Uttar Pradesh'])
        >>> norm_map = {'UP': 'Uttar Pradesh', 'U.P.': 'Uttar Pradesh'}
        >>> normalized = normalize_string_column(s, norm_map)
        >>> print(normalized.unique())
        ['Uttar Pradesh']
    """
    # Strip whitespace and convert to consistent case
    normalized = series.astype(str).str.strip()

    # Apply normalization map if provided
    if normalization_map:
        normalized = normalized.replace(normalization_map)

    return normalized
