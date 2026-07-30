import json
from unittest.mock import patch

import pytest

from nanobot.utils.calibration import ModelCalibrationManager, _load_model_map


@pytest.fixture
def mock_workspace_path(tmp_path):
    """Mocks nanobot.utils.calibration.get_workspace_path to return a temp directory."""
    with patch("nanobot.utils.calibration.get_workspace_path", return_value=tmp_path):
        yield tmp_path


@pytest.fixture(autouse=True)
def reset_calibration_state():
    """Reset lazy-loaded module state before each test."""
    import nanobot.utils.calibration as cal_module

    cal_module._CALIBRATION_FILE_PATH = None
    cal_module._MODEL_SERIES = None
    cal_module._MODEL_TO_SERIES_MAP = None


def test_load_model_map_success(mock_workspace_path):
    # Setup valid model_map.json
    model_map_data = {
        "gpt-4": ["gpt-4", "gpt-4-turbo", "gpt-4o"],
        "claude-3": ["claude-3-opus", "claude-3-sonnet"],
    }
    model_map_file = mock_workspace_path / "model_map.json"
    model_map_file.write_text(json.dumps(model_map_data), encoding="utf-8")

    result = _load_model_map()

    assert result == {
        "gpt-4": ("gpt-4", "gpt-4-turbo", "gpt-4o"),
        "claude-3": ("claude-3-opus", "claude-3-sonnet"),
    }


def test_load_model_map_missing_file(mock_workspace_path):
    # Ensure file doesn't exist
    model_map_file = mock_workspace_path / "model_map.json"
    if model_map_file.exists():
        model_map_file.unlink()

    result = _load_model_map()
    assert result == {}


def test_load_model_map_malformed_json(mock_workspace_path):
    # Setup malformed JSON
    model_map_file = mock_workspace_path / "model_map.json"
    model_map_file.write_text("invalid json", encoding="utf-8")

    result = _load_model_map()
    assert result == {}


def test_get_ratio_resolution_order(mock_workspace_path):
    manager = ModelCalibrationManager()

    # 1. User-defined series: 'custom-model' in 'custom-series'.
    manager.series["custom-series"] = ["custom-model"]
    manager.ratios["custom-series"] = 1.5
    assert manager.get_ratio("custom-model") == 1.5

    # 2. Direct model ID ratio.
    manager.ratios["direct-model"] = 0.8
    assert manager.get_ratio("direct-model") == 0.8

    # 3. Default for unknown model
    assert manager.get_ratio("unknown-model") == 1.0


def test_update_ratio_ema(mock_workspace_path):
    manager = ModelCalibrationManager()

    # Initial ratio for 'model-a' should be the raw ratio on first update
    # raw = 120 / 100 = 1.2
    # current = 1.2 (since it doesn't exist in self.ratios)
    # new = 0.1 * 1.2 + 0.9 * 1.2 = 1.2
    ratio = manager.update_ratio("model-a", 120, 100)
    assert ratio == pytest.approx(1.2)
    assert manager.ratios["model-a"] == pytest.approx(1.2)

    # Second update
    # raw = 140 / 100 = 1.4
    # current = 1.2
    # new = 0.1 * 1.4 + 0.9 * 1.2 = 0.14 + 1.08 = 1.22
    ratio = manager.update_ratio("model-a", 140, 100)
    assert ratio == pytest.approx(1.22)
    assert manager.ratios["model-a"] == pytest.approx(1.22)


def test_update_ratio_invalid_estimate(mock_workspace_path):
    manager = ModelCalibrationManager()

    # estimated = 0
    assert manager.update_ratio("model-a", 100, 0) == 1.0
    # estimated < 0
    assert manager.update_ratio("model-a", 100, -10) == 1.0


def test_update_ratio_persistence(mock_workspace_path):
    manager = ModelCalibrationManager()

    manager.update_ratio("model-persist", 110, 100)

    # Check if file was written
    cal_file = mock_workspace_path / "model_calibration.json"
    assert cal_file.exists()
    data = json.loads(cal_file.read_text())
    assert "ratios" in data
    assert data["ratios"]["model-persist"] == pytest.approx(1.1)


def test_model_calibration_manager_init_with_data(mock_workspace_path):
    # Setup existing calibration data
    cal_file = mock_workspace_path / "model_calibration.json"
    cal_data = {
        "series": {"gpt-series": ["gpt-4", "gpt-4o"]},
        "ratios": {"gpt-series": 1.1, "claude-direct": 0.9},
    }
    cal_file.write_text(json.dumps(cal_data))

    manager = ModelCalibrationManager()
    assert manager.series["gpt-series"] == ["gpt-4", "gpt-4o"]
    assert manager.ratios["gpt-series"] == 1.1
    assert manager.ratios["claude-direct"] == 0.9


def test_model_calibration_manager_init_malformed_data(mock_workspace_path):
    cal_file = mock_workspace_path / "model_calibration.json"
    cal_file.write_text("invalid json")

    manager = ModelCalibrationManager()
    assert manager.series == {}
    assert manager.ratios == {}
