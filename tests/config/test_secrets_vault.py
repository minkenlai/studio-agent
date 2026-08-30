"""Tests for in-memory secrets-vault injection (load_config / save_config)."""

import json

import pytest

from nanobot.config import loader


class _RecordingLogger:
    """Minimal logger stub so tests don't depend on global loguru state."""

    def __init__(self):
        self.messages: list[str] = []

    def _record(self, level: str, msg: str, *args, **kwargs):
        self.messages.append(f"{level} {msg}")

    def debug(self, msg, *a, **k):
        self._record("DEBUG", msg, *a, **k)

    def info(self, msg, *a, **k):
        self._record("INFO", msg, *a, **k)

    def warning(self, msg, *a, **k):
        self._record("WARNING", msg, *a, **k)

    def error(self, msg, *a, **k):
        self._record("ERROR", msg, *a, **k)

    def exception(self, msg, *a, **k):
        self._record("ERROR", msg, *a, **k)

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Isolate vault resolution + logging for every test."""
    monkeypatch.delenv("NANOBOT_SECRETS", raising=False)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setattr(loader, "_current_secrets_path", None)
    loader._injected_secret_paths.clear()

    rec = _RecordingLogger()
    monkeypatch.setattr(loader, "logger", rec)
    yield rec
    loader._injected_secret_paths.clear()


def _write_config(path, sensitive: dict | None = None) -> dict:
    """Write a minimal config with the given sensitive leaves set."""
    sensitive = sensitive or {}

    def val(key, default="***"):
        return sensitive.get(key, default)

    cfg = {
        "agents": {"defaults": {"workspace": str(path.parent)}},
        "providers": {
            "gemini": {"apiKey": val("providers.gemini.apiKey")},
            "ollama": {"apiKey": val("providers.ollama.apiKey")},
        },
        "channels": {"telegram": {"token": val("channels.telegram.token")}},
        "models": {"test-model": {"model": "test-model", "contextMax": 1000}},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _write_vault(path, secrets: dict) -> None:
    path.write_text(json.dumps(secrets), encoding="utf-8")


def test_inject_fills_masked_leaves_in_memory(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path)
    _write_vault(
        vault_path,
        {
            "providers.gemini.apiKey": "GEMINI-REAL-123",
            "channels.telegram.token": "TG-REAL-456",
        },
    )
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "GEMINI-REAL-123"
    assert config.channels.telegram["token"] == "TG-REAL-456"  # extra=allow -> raw dict
    # The file on disk must still be masked.
    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["providers"]["gemini"]["apiKey"] == "***"
    assert on_disk["channels"]["telegram"]["token"] == "***"
    assert "Injected 2/2 secrets" in _isolate.text
    # Vault omits the ollama leaf (test scenario) -> drift warning for it.
    assert "still masked" in _isolate.text
    assert "providers.ollama.apiKey" in _isolate.text


def test_inject_empty_vault_value_means_intentionally_empty(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path)
    _write_vault(
        vault_path,
        {
            "providers.ollama.apiKey": "",  # explicitly disabled
            "providers.gemini.apiKey": "GEMINI-REAL-123",
            "channels.telegram.token": "TG-REAL-456",
        },
    )
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)

    assert config.providers.ollama.api_key == ""
    # All three vault keys resolved; nothing left masked -> no warnings.
    assert "still masked" not in _isolate.text
    assert "Injected 3/3 secrets" in _isolate.text


def test_inject_warns_on_drift_key_not_in_config(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path)
    _write_vault(vault_path, {"providers.nonexistent.apiKey": "XYZ"})
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "***"  # untouched, no vault entry
    assert "drift?" in _isolate.text


def test_inject_does_not_override_explicit_config_value(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path, {"providers.gemini.apiKey": "LOCAL-EXPLICIT"})
    _write_vault(vault_path, {"providers.gemini.apiKey": "VAULT-VALUE"})
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "LOCAL-EXPLICIT"
    assert "not overridden" in _isolate.text


def test_no_vault_keeps_masks_silently(tmp_path, _isolate):
    cfg_path = tmp_path / "config.json"
    _write_config(cfg_path)

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "***"
    assert "still masked" not in _isolate.text
    assert loader._injected_secret_paths == set()


def test_vault_missing_file_is_skipped(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    _write_config(cfg_path)
    monkeypatch.setenv("NANOBOT_SECRETS", str(tmp_path / "nope.json"))

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "***"
    assert "skipping injection" in _isolate.text


def test_vault_non_dict_is_skipped(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path)
    vault_path.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)

    assert config.providers.gemini.api_key == "***"
    assert "not a JSON object" in _isolate.text


def test_get_secrets_path_resolution_order(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.json"
    envv = tmp_path / "env.json"
    cred = tmp_path / "cred" / "secrets.json"
    for p in (explicit, envv, cred):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    # 3. credentials tmpfs auto-detect
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path / "cred"))
    assert loader.get_secrets_path() == cred

    # 2. env var beats credentials dir
    monkeypatch.setenv("NANOBOT_SECRETS", str(envv))
    assert loader.get_secrets_path() == envv

    # 1. explicit global beats env var
    loader.set_secrets_path(explicit)
    assert loader.get_secrets_path() == explicit

    # nothing set -> None
    loader._current_secrets_path = None
    monkeypatch.delenv("NANOBOT_SECRETS")
    monkeypatch.delenv("CREDENTIALS_DIRECTORY")
    assert loader.get_secrets_path() is None


def test_save_config_re_masks_injected_secrets(tmp_path, monkeypatch, _isolate):
    cfg_path = tmp_path / "config.json"
    vault_path = tmp_path / "secrets.json"
    _write_config(cfg_path)
    _write_vault(vault_path, {"providers.gemini.apiKey": "GEMINI-REAL-123"})
    monkeypatch.setenv("NANOBOT_SECRETS", str(vault_path))

    config = loader.load_config(cfg_path)
    assert config.providers.gemini.api_key == "GEMINI-REAL-123"

    # Simulate an in-process re-save (e.g. onboard refresh) while vault loaded.
    config.agents.defaults.max_tokens = 777
    loader.save_config(config, cfg_path)

    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["providers"]["gemini"]["apiKey"] == "***"  # re-masked
    assert on_disk["agents"]["defaults"]["maxTokens"] == 777  # other edits kept


def test_save_config_without_vault_keeps_user_values(tmp_path, _isolate):
    cfg_path = tmp_path / "config.json"
    # No vault loaded; user types a real key directly (onboarding-style).
    _write_config(cfg_path)
    config = loader.load_config(cfg_path)
    config.providers.gemini.api_key = "USER-TYPED-KEY"
    loader.save_config(config, cfg_path)

    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["providers"]["gemini"]["apiKey"] == "USER-TYPED-KEY"
