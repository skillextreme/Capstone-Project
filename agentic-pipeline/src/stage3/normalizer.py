"""
Key Normalizer

Handles normalization of key columns (state, crop, etc.) to ensure
consistent values across different data sources.

Example:
    "UP" → "Uttar Pradesh"
    "U.P." → "Uttar Pradesh"
    "Rice" → "rice"
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from utils.stats_utils import normalize_string_column

logger = get_logger(__name__)


class KeyNormalizer:
    """
    Normalizes key column values to standard forms.

    Example:
        >>> normalizer = KeyNormalizer(config={'state': {'UP': 'Uttar Pradesh'}})
        >>> df['state'] = normalizer.normalize(df, 'state')
    """

    def __init__(self, config: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize the Key Normalizer.

        Args:
            config: Dict of {column_name: {variant: standard_form}}
        """
        self.normalization_rules = config or {}

        # Add default rules
        self._add_default_rules()

        logger.info(f"Initialized KeyNormalizer with rules for {len(self.normalization_rules)} columns")

    def _add_default_rules(self):
        """Add commonly used normalization rules."""
        defaults = {
            'state': {
                'UP': 'Uttar Pradesh',
                'U.P.': 'Uttar Pradesh',
                'Delhi NCR': 'Delhi',
                'TN': 'Tamil Nadu',
                'T.N.': 'Tamil Nadu',
                'AP': 'Andhra Pradesh',
                'A.P.': 'Andhra Pradesh',
                'HP': 'Himachal Pradesh',
                'H.P.': 'Himachal Pradesh',
                'MP': 'Madhya Pradesh',
                'M.P.': 'Madhya Pradesh',
                'WB': 'West Bengal',
                'W.B.': 'West Bengal',
                'MH': 'Maharashtra',
                'KA': 'Karnataka',
                'KL': 'Kerala',
                'TN': 'Tamil Nadu',
                'RJ': 'Rajasthan',
                'GJ': 'Gujarat',
            },
            'crop': {
                'Rice': 'rice',
                'RICE': 'rice',
                'Wheat': 'wheat',
                'WHEAT': 'wheat',
                'Sugarcane': 'sugarcane',
                'Sugar Cane': 'sugarcane',
                'SUGARCANE': 'sugarcane',
                'Maize': 'maize',
                'MAIZE': 'maize',
                'Corn': 'maize',
                'Jowar': 'jowar',
                'JOWAR': 'jowar',
                'Sorghum': 'jowar',
            }
        }

        # Merge with user-provided rules (user rules take precedence)
        for col, rules in defaults.items():
            if col not in self.normalization_rules:
                self.normalization_rules[col] = {}
            # Add defaults that don't conflict with user rules
            for variant, standard in rules.items():
                if variant not in self.normalization_rules[col]:
                    self.normalization_rules[col][variant] = standard

    def normalize(
        self,
        df: pd.DataFrame,
        column: str,
        inplace: bool = False
    ) -> pd.Series:
        """
        Normalize a column in a DataFrame.

        Args:
            df: DataFrame containing the column
            column: Column name to normalize
            inplace: Whether to modify df in place

        Returns:
            Normalized Series

        Example:
            >>> df['state'] = normalizer.normalize(df, 'state')
        """
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return df[column] if column in df.columns else pd.Series()

        # Get normalization rules for this column
        rules = self.normalization_rules.get(column, {})

        if not rules:
            logger.debug(f"No normalization rules for column '{column}'")
            return df[column]

        # Normalize
        logger.info(f"Normalizing column '{column}' with {len(rules)} rules")

        normalized = normalize_string_column(df[column], rules)

        # Count changes
        changes = (df[column].astype(str) != normalized.astype(str)).sum()
        logger.info(f"  Normalized {changes} values in '{column}'")

        if inplace:
            df[column] = normalized

        return normalized

    def normalize_all(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Normalize multiple columns in a DataFrame.

        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (None = all with rules)
            inplace: Whether to modify df in place

        Returns:
            DataFrame with normalized columns

        Example:
            >>> df = normalizer.normalize_all(df, columns=['state', 'crop'])
        """
        if not inplace:
            df = df.copy()

        # If columns not specified, use all columns with rules
        if columns is None:
            columns = [col for col in self.normalization_rules.keys() if col in df.columns]

        logger.info(f"Normalizing {len(columns)} columns")

        for col in columns:
            if col in df.columns:
                df[col] = self.normalize(df, col, inplace=True)

        return df

    def add_rules(self, column: str, rules: Dict[str, str]):
        """
        Add normalization rules for a column.

        Args:
            column: Column name
            rules: Dict of {variant: standard_form}

        Example:
            >>> normalizer.add_rules('crop', {'RICE': 'rice', 'Rice': 'rice'})
        """
        if column not in self.normalization_rules:
            self.normalization_rules[column] = {}

        self.normalization_rules[column].update(rules)
        logger.info(f"Added {len(rules)} normalization rules for '{column}'")

    def get_unique_values(self, df: pd.DataFrame, column: str) -> List[str]:
        """
        Get unique values in a column (after normalization).

        Args:
            df: DataFrame
            column: Column name

        Returns:
            List of unique values

        Example:
            >>> unique_states = normalizer.get_unique_values(df, 'state')
        """
        if column not in df.columns:
            return []

        normalized = self.normalize(df, column)
        return sorted(normalized.dropna().unique().tolist())
