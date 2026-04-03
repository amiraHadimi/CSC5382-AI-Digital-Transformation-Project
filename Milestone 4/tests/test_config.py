"""Unit tests for src/utils/config.py"""
import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config


class TestLoadConfig:
    def test_loads_without_error(self):
        assert isinstance(load_config(), dict)

    def test_has_required_sections(self):
        cfg = load_config()
        for s in ["model", "inference", "data", "mlflow", "zenml", "codecarbon"]:
            assert s in cfg

    def test_model_section(self):
        cfg = load_config()
        assert "hf_author" in cfg["model"]
        assert cfg["model"]["num_labels"] == 1

    def test_inference_section(self):
        cfg = load_config()
        assert cfg["inference"]["max_len"] > 0

    def test_mlflow_section(self):
        cfg = load_config()
        assert "experiment_name" in cfg["mlflow"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/params.yaml")
