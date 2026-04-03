"""
utils/config.py
===============
Loads the central params.yaml configuration file.
"""
import os
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"


def load_config(config_path=None):
    """Load and return the central params.yaml configuration."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_hf_token():
    """Retrieve the Hugging Face API token from the environment."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n"
            "  Windows:    set HF_TOKEN=your_token_here\n"
            "  Linux/Mac:  export HF_TOKEN=your_token_here"
        )
    return token
