"""
Configuration Management

Loads and manages pipeline configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent))

from utils.file_utils import load_config as load_yaml_config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class Config:
    """
    Pipeline configuration manager.

    Loads configuration from:
    1. YAML file (config/pipeline_config.yaml)
    2. Environment variables (.env)
    3. Command-line overrides

    Example:
        >>> config = Config()
        >>> print(config.get('data.raw_dir'))
        data/raw
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML config file (optional)
        """
        # Load .env file if it exists
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path}")

        # Default config file
        if config_file is None:
            config_file = Path(__file__).parent.parent / 'config' / 'pipeline_config.yaml'

        # Load YAML config
        self.config = {}
        if Path(config_file).exists():
            self.config = load_yaml_config(config_file)
            logger.info(f"Loaded config from: {config_file}")
        else:
            logger.warning(f"Config file not found: {config_file}")

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to config."""
        # Check for common overrides
        if os.getenv('TEST_SPLIT_YEAR'):
            self.config.setdefault('executor', {})
            self.config['executor'].setdefault('split', {})
            self.config['executor']['split']['test_split_year'] = int(os.getenv('TEST_SPLIT_YEAR'))

        if os.getenv('RANDOM_SEED'):
            self.config['executor']['split']['random_seed'] = int(os.getenv('RANDOM_SEED'))

        if os.getenv('LOG_LEVEL'):
            self.config.setdefault('logging', {})
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Config key (e.g., 'data.raw_dir')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get('data.raw_dir')
            'data/raw'
            >>> config.get('nonexistent.key', 'default')
            'default'
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.

        Args:
            key: Config key (e.g., 'data.raw_dir')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        Get configuration for a specific stage.

        Args:
            stage: Stage name ('summarizer', 'task_suggester', 'planner', 'executor')

        Returns:
            Stage configuration dictionary
        """
        return self.config.get(stage, {})

    def get_verification_config(self, verification: str) -> Dict[str, Any]:
        """
        Get configuration for a specific verification.

        Args:
            verification: Verification name ('v1', 'v3', 'v4')

        Returns:
            Verification configuration dictionary
        """
        return self.config.get('verification', {}).get(verification, {})

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()


# Global config instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.

    Args:
        config_file: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_file)

    return _global_config
