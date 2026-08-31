"""Tests for the config overlayPolicy (whitelisted deep-merge overlay)."""

import json

import pytest

from nanobot.config import loader
from nanobot.config.schema import Config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("NANOBOT_SECRETS", raising=False)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setattr(loader, "_current_secrets_path", None)
    monkeypatch.setattr(loader, "_current_overlay_path", None)
    monkeypatch.delenv("NANOBOT_CONFIG_OVERLAY", raising=False)
    loader._injected_secret_paths.clear()
    yield
    monkeypatch.setattr(loader, "_current_overlay_path", None)


def _base(overlay_policy=None, **extra) -> dict:
    cfg = {
        "models": {
            "a": {"model": "a", "temperature": 0.5},
            "b": {"model": "b", "contextMax": 8192},
        },
        "agents": {"defaults": {"workspace": "/ws", "maxTokens": 4096}},
        "tools": {"restrictToWorkspace": True},
        "channels": {"telegram": {"enabled": True}},
    }
    if overlay_policy is not None:
        cfg["overlayPolicy"] = overlay_policy
    cfg.update(extra)
    return cfg


def _policy(**kw) -> dict:
    p = {
        "allowedSections": ["models", "agents"],
        "deniedPaths": ["agents.defaults.workspace"],
        "onViolation": "warn",
    }
    p.update(kw)
    return p


def test_no_policy_keeps_legacy_section_replace(tmp_path):
    base = _base()
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"models": {"a": {"model": "a"}}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    # Legacy behavior: section replaced wholesale.
    assert out["models"] == {"a": {"model": "a"}}
    assert "b" not in out["models"]


def test_whitelisted_section_deep_merges(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(
        json.dumps({"models": {"a": {"temperature": 0.9}, "c": {"model": "c"}}}),
        encoding="utf-8",
    )

    out = loader.apply_overlay(base, ov)

    # Deep merge: existing model's other keys survive; new model added.
    assert out["models"]["a"] == {"model": "a", "temperature": 0.9}
    assert out["models"]["b"]["contextMax"] == 8192
    assert out["models"]["c"]["model"] == "c"


def test_non_whitelisted_section_rejected_warn(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(
        json.dumps(
            {
                "tools": {"restrictToWorkspace": False},  # not whitelisted
                "models": {"a": {"temperature": 0.7}},  # allowed
            }
        ),
        encoding="utf-8",
    )

    out = loader.apply_overlay(base, ov)

    assert out["tools"]["restrictToWorkspace"] is True  # untouched
    assert out["models"]["a"]["temperature"] == 0.7  # allowed part applied


def test_overlay_cannot_set_overlay_policy(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"overlayPolicy": {"allowedSections": ["tools"]}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["overlayPolicy"]["allowedSections"] == ["models", "agents"]


def test_base_config_with_camel_case_overlay_policy_validates():
    """Regression: the deployed template writes ``overlayPolicy`` camelCase.

    The root ``Config`` is a BaseSettings that does NOT inherit the
    to_camel alias generator from Base, so without an explicit
    validation_alias the camelCase key is rejected as extra_forbidden and
    the gateway silently boots with the DEFAULT config.
    """
    data = _base(_policy())
    cfg = Config.model_validate(data)

    assert cfg.overlay_policy.allowed_sections == ["models", "agents"]
    assert cfg.overlay_policy.denied_paths == ["agents.defaults.workspace"]
    assert cfg.overlay_policy.on_violation == "warn"


def test_base_config_with_snake_case_overlay_policy_validates():
    data = _base()
    data["overlay_policy"] = _policy()
    cfg = Config.model_validate(data)
    assert cfg.overlay_policy.allowed_sections == ["models", "agents"]


def test_denied_path_modification_rolled_back(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(
        json.dumps({"agents": {"defaults": {"workspace": "/pwned", "maxTokens": 8192}}}),
        encoding="utf-8",
    )

    out = loader.apply_overlay(base, ov)

    assert out["agents"]["defaults"]["workspace"] == "/ws"  # frozen
    assert out["agents"]["defaults"]["maxTokens"] == 8192  # allowed sibling applied


def test_denied_path_addition_rolled_back(tmp_path):
    base = _base(_policy(deniedPaths=["agents.defaults.model"]))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"agents": {"defaults": {"model": "evil/model"}}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert "model" not in out["agents"]["defaults"]


def test_denied_path_addition_prunes_created_containers(tmp_path):
    base = _base(_policy(deniedPaths=["models.ghost"]))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"models": {"ghost": {"model": "g"}}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert "ghost" not in out["models"]
    assert out["models"]["a"]["model"] == "a"


def test_on_violation_fail_raises(tmp_path):
    base = _base(_policy(onViolation="fail"))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"tools": {"x": 1}}), encoding="utf-8")

    with pytest.raises(loader.ConfigOverlayPolicyError):
        loader.apply_overlay(base, ov)


def test_on_violation_fail_clean_boot_when_no_violation(tmp_path):
    base = _base(_policy(onViolation="fail"))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"models": {"a": {"temperature": 0.1}}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["models"]["a"]["temperature"] == 0.1


def test_non_object_section_rejected(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"models": ["not", "an", "object"]}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["models"]["a"]["model"] == "a"  # base untouched


def test_overlay_not_object_is_skipped(tmp_path):
    base = _base(_policy())
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps([1, 2]), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["models"]["a"]["model"] == "a"


def test_policy_snake_case_keys(tmp_path):
    base = _base()
    base["overlayPolicy"] = {
        "allowed_sections": ["models"],
        "denied_paths": ["agents.defaults.workspace"],
        "on_violation": "warn",
    }
    ov = tmp_path / "ov.json"
    ov.write_text(
        json.dumps(
            {"models": {"a": {"temperature": 0.3}}, "tools": {"restrictToWorkspace": False}}
        ),
        encoding="utf-8",
    )

    out = loader.apply_overlay(base, ov)

    assert out["models"]["a"]["temperature"] == 0.3
    assert out["tools"]["restrictToWorkspace"] is True


def test_list_values_replaced_not_merged(tmp_path):
    base = _base(_policy(allowedSections=["agents"]))
    base["agents"]["allowlist"] = ["a", "b"]
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"agents": {"allowlist": ["c"]}}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["agents"]["allowlist"] == ["c"]


def test_invalid_on_violation_falls_back_to_warn(tmp_path):
    base = _base(_policy(onViolation="scream"))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"tools": {"x": 1}}), encoding="utf-8")

    # Unknown mode must not raise; the (non-whitelisted) part is simply rejected.
    out = loader.apply_overlay(base, ov)

    assert out["tools"] == {"restrictToWorkspace": True}
    assert out["models"]["a"]["model"] == "a"


def test_empty_overlay_ok(tmp_path):
    base = _base(_policy(onViolation="fail"))
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({}), encoding="utf-8")

    out = loader.apply_overlay(base, ov)

    assert out["models"]["a"]["model"] == "a"
