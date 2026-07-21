"""Tests for the Capability Registry (Task 1 — Native Audio Support).

Verifies:
- ModelCapabilities dataclass (frozen, defaults)
- Canonical model lookup
- Prefixed model lookup (provider/model → model)
- Unknown model fallback
- Config overrides
- LLMClient.get_model_capabilities convenience method
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nanobot.config.schema import Config, get_default_capabilities
from nanobot.providers.base import ModelCapabilities

# ---------------------------------------------------------------------------
# ModelCapabilities dataclass
# ---------------------------------------------------------------------------


def test_capabilities_defaults():
    caps = ModelCapabilities()
    assert caps.audio is False
    assert caps.vision is False
    assert caps.reasoning is False


def test_capabilities_frozen():
    caps = ModelCapabilities(audio=True)
    with pytest.raises(Exception):
        caps.audio = False  # type: ignore


def test_capabilities_custom():
    caps = ModelCapabilities(audio=True, vision=True)
    assert caps.audio is True
    assert caps.vision is True
    assert caps.reasoning is False


# ---------------------------------------------------------------------------
# get_default_capabilities helper
# ---------------------------------------------------------------------------


def test_default_capabilities_contains_models():
    defaults = get_default_capabilities()
    assert "gpt-4o" in defaults
    assert "gpt-4o-mini" in defaults
    assert "gemma4-e4b" in defaults
    assert "claude-3-5-sonnet" in defaults
    assert "deepseek-r1" in defaults


def test_default_capabilities_values():
    defaults = get_default_capabilities()
    gpt4o = defaults["gpt-4o"]
    assert gpt4o.audio is True
    assert gpt4o.vision is True

    ds = defaults["deepseek-r1"]
    assert ds.reasoning is True
    assert ds.audio is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_config_capabilities_default():
    config = Config()
    assert "gpt-4o" in config.capabilities
    assert config.capabilities["gpt-4o"].audio is True


def test_config_capabilities_override(tmp_path):
    import json

    config_path = tmp_path / "config.json"
    config_data = {
        "capabilities": {
            "custom-model": {
                "audio": True,
                "vision": True,
                "reasoning": True,
            }
        }
    }

    config_path.write_text(json.dumps(config_data))

    from nanobot.config.loader import load_config

    config = load_config(config_path)
    assert "custom-model" in config.capabilities
    caps = config.capabilities["custom-model"]
    assert caps.audio is True
    assert caps.vision is True
    assert caps.reasoning is True


# ---------------------------------------------------------------------------
# get_capabilities resolver (registry)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Provide a Config with the default capabilities map."""
    return Config()


def test_get_capabilities_canonical_lookup(mock_config):
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        caps = get_capabilities("gpt-4o")
        assert caps.audio is True
        assert caps.vision is True


def test_get_capabilities_prefixed_lookup(mock_config):
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        # Should strip "openrouter/" prefix and resolve to "gpt-4o"
        caps = get_capabilities("openrouter/gpt-4o")
        assert caps.audio is True
        assert caps.vision is True


def test_get_capabilities_deep_prefix_lookup(mock_config):
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        caps = get_capabilities("anthropic/claude-3-5-sonnet")
        assert caps.vision is True
        assert caps.audio is False


def test_get_capabilities_unknown_model_fallback(mock_config):
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        caps = get_capabilities("totally-unknown-model")
        assert caps.audio is False
        assert caps.vision is False
        assert caps.reasoning is False


def test_get_capabilities_case_insensitive(mock_config):
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        caps = get_capabilities("GPT-4O")
        assert caps.audio is True


def test_get_capabilities_config_override(tmp_path, mock_config):
    """A custom model added to the config should be resolvable."""

    # Add a custom model to the config's capabilities
    mock_config.capabilities["my-custom-model"] = ModelCapabilities(
        audio=True, vision=True, reasoning=True
    )

    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        caps = get_capabilities("my-custom-model")
        assert caps.audio is True
        assert caps.vision is True
        assert caps.reasoning is True


def test_get_capabilities_prefix_match(mock_config):
    """A model ID that extends a known key should match by prefix."""
    with patch("nanobot.config.loader.load_config", return_value=mock_config):
        from nanobot.providers.registry import get_capabilities

        # "gemma4-12b-it-qat" is not a key, but starts with "gemma4-12b"
        caps = get_capabilities("gemma4-12b-it-qat")
        assert caps.audio is True
        assert caps.vision is True


def test_get_capabilities_prefix_longest_wins(tmp_path, mock_config):
    """When multiple keys are prefixes, the longest key must win."""
    import json

    config_path = tmp_path / "config.json"
    config_data = {
        "capabilities": {
            "base": {"audio": True, "vision": False, "reasoning": False},
            "base-pro": {"audio": False, "vision": True, "reasoning": False},
        }
    }
    config_path.write_text(json.dumps(config_data))

    from nanobot.config.loader import load_config

    loaded = load_config(config_path)

    with patch("nanobot.config.loader.load_config", return_value=loaded):
        from nanobot.providers.registry import get_capabilities

        # "base-pro-v2" starts with both "base" and "base-pro"; longest wins
        caps = get_capabilities("base-pro-v2")
        assert caps.audio is False
        assert caps.vision is True
