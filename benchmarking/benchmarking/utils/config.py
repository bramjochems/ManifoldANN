"""Configuration loading and validation."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_algorithm_metadata(config_dir: str = None) -> Dict[str, Any]:
    """Load algorithm metadata from algorithms.yaml.

    Args:
        config_dir: Directory containing config files

    Returns:
        Dictionary mapping algorithm names to metadata
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    metadata_path = config_dir / "algorithms.yaml"

    if not metadata_path.exists():
        return {}

    with open(metadata_path) as f:
        data = yaml.safe_load(f)

    return data.get("algorithms", {})


def load_config(dataset_name: str, config_dir: str = None) -> Dict[str, Any]:
    """Load configuration for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'fashion-mnist', 'nytimes')
        config_dir: Directory containing config files (defaults to package configs/)

    Returns:
        Dictionary containing dataset configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if config_dir is None:
        # Default to package configs directory
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {list_available_configs(config_dir)}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["dataset", "metric", "n_train", "n_test", "algorithms"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Config {config_path} missing required fields: {missing}")

    # Validate metric
    if config["metric"] not in ["euclidean", "angular"]:
        raise ValueError(f"Invalid metric: {config['metric']}")

    # Validate algorithms dict
    if not isinstance(config["algorithms"], dict):
        raise ValueError("'algorithms' must be a dictionary")

    return config


def list_available_configs(config_dir: str = None) -> list:
    """List all available dataset configurations.

    Args:
        config_dir: Directory containing config files

    Returns:
        List of available dataset names (without .yaml extension)
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        return []

    return sorted([f.stem for f in config_dir.glob("*.yaml")])
