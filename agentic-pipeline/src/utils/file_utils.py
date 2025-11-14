"""
File I/O utilities for the agentic pipeline.
Handles loading and saving CSV, JSON, YAML, and Parquet files.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> config = load_config("config/pipeline_config.yaml")
        >>> print(config['data']['raw_dir'])
        data/raw
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_csv(
    file_path: Union[str, Path],
    sample_size: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.

    Args:
        file_path: Path to CSV file
        sample_size: Number of rows to load (None = all rows)
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame containing CSV data

    Example:
        >>> df = load_csv("data/raw/yields.csv", sample_size=1000)
        >>> print(f"Loaded {len(df)} rows")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info(f"Loading CSV: {file_path}")

    if sample_size:
        df = pd.read_csv(file_path, nrows=sample_size, **kwargs)
        logger.info(f"Loaded {len(df)} rows (sampled from {sample_size})")
    else:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded {len(df)} rows")

    return df


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data (dict or list)

    Example:
        >>> data = load_json("data/summaries/yields.summary.json")
        >>> print(data['row_count'])
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    logger.debug(f"Loading JSON: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def save_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (dict or list)
        file_path: Output file path
        indent: JSON indentation (default: 2)

    Example:
        >>> summary = {"file": "yields.csv", "rows": 1000}
        >>> save_json(summary, "data/summaries/yields.summary.json")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving JSON: {file_path}")

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    logger.info(f"Saved JSON to: {file_path}")


def load_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load Parquet file into pandas DataFrame.

    Args:
        file_path: Path to Parquet file
        **kwargs: Additional arguments passed to pd.read_parquet

    Returns:
        DataFrame containing Parquet data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    logger.info(f"Loading Parquet: {file_path}")
    df = pd.read_parquet(file_path, **kwargs)
    logger.info(f"Loaded {len(df)} rows")

    return df


def save_parquet(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    **kwargs
) -> None:
    """
    Save DataFrame to Parquet file.

    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments passed to df.to_parquet
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving Parquet: {file_path}")
    df.to_parquet(file_path, index=False, **kwargs)
    logger.info(f"Saved {len(df)} rows to: {file_path}")


def get_file_list(
    directory: Union[str, Path],
    pattern: str = "*.csv"
) -> List[Path]:
    """
    Get list of files matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern (default: "*.csv")

    Returns:
        List of matching file paths

    Example:
        >>> files = get_file_list("data/raw", "*.csv")
        >>> print(f"Found {len(files)} CSV files")
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")

    return files
