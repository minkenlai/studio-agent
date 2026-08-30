import os
from pathlib import Path

from nanobot.config.paths import (
    get_bridge_install_dir,
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_workspace_path,
    is_default_workspace,
)


def test_runtime_dirs_follow_config_path(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "instance-a"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)

    assert get_data_dir() == config_dir
    assert get_cron_dir() == config_dir / "cron"
    assert get_logs_dir() == config_dir / "logs"


def test_get_data_dir_fallback_on_read_only(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "readonly-instance"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)

    # Simulate read-only directory
    original_access = os.access

    def mock_access(path, mode):
        if str(path) == str(config_dir) and mode == os.W_OK:
            return False
        return original_access(path, mode)

    monkeypatch.setattr(os, "access", mock_access)
    # Mock ensure_dir to avoid creating ~/.nanobot in restricted test env
    monkeypatch.setattr("nanobot.config.paths.ensure_dir", lambda p: p)

    assert get_data_dir() == Path.home() / ".nanobot"


def test_media_dir_supports_channel_namespace(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "instance-b"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: config_file)

    assert get_media_dir() == config_dir / "media"
    assert get_media_dir("telegram") == config_dir / "media" / "telegram"


def test_shared_and_legacy_paths_remain_global() -> None:
    assert get_cli_history_path() == Path.home() / ".nanobot" / "history" / "cli_history"
    assert get_bridge_install_dir() == Path.home() / ".nanobot" / "bridge"
    assert get_legacy_sessions_dir() == Path.home() / ".nanobot" / "sessions"


def test_get_data_dir_ignores_credentials_dir_config(monkeypatch, tmp_path: Path) -> None:
    # Even if the config path lives inside the credentials tmpfs (and the
    # directory were writable), runtime state must NOT be rooted there.
    cred = tmp_path / "cred"
    (cred / "run-credentials").mkdir(parents=True)
    tmpfs_cfg = cred / "run-credentials" / "nanobot.service" / "config.json"
    tmpfs_cfg.parent.mkdir(parents=True)
    tmpfs_cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("nanobot.config.paths.get_config_path", lambda: tmpfs_cfg)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(cred / "run-credentials" / "nanobot.service"))
    monkeypatch.setattr("nanobot.config.paths.ensure_dir", lambda p: p)

    assert get_data_dir() == Path.home() / ".nanobot"
    assert get_cron_dir() == Path.home() / ".nanobot" / "cron"


def test_config_path_resolves_from_credentials_dir(monkeypatch, tmp_path: Path) -> None:
    from nanobot.config import loader

    cred = tmp_path / "cred"
    cred.mkdir()
    (cred / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(cred))
    monkeypatch.setattr(loader, "_current_config_path", None)

    assert loader.get_config_path() == cred / "config.json"


def test_config_path_credentials_dir_without_file_falls_back(monkeypatch, tmp_path: Path) -> None:
    from nanobot.config import loader

    cred = tmp_path / "cred-empty"
    cred.mkdir()  # no config.json inside
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(cred))
    monkeypatch.setattr(loader, "_current_config_path", None)

    assert loader.get_config_path() == Path.home() / ".nanobot" / "config.json"


def test_config_path_explicit_flag_overrides_credentials_dir(monkeypatch, tmp_path: Path) -> None:
    from nanobot.config import loader

    cred = tmp_path / "cred"
    cred.mkdir()
    (cred / "config.json").write_text("{}", encoding="utf-8")
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(cred))

    loader.set_config_path(explicit)
    try:
        assert loader.get_config_path() == explicit
    finally:
        loader._current_config_path = None


def test_config_path_default_without_credentials(monkeypatch) -> None:
    from nanobot.config import loader

    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setattr(loader, "_current_config_path", None)

    assert loader.get_config_path() == Path.home() / ".nanobot" / "config.json"


def test_workspace_path_is_explicitly_resolved(monkeypatch) -> None:
    # Mock ensure_dir to avoid real mkdir in restricted env
    monkeypatch.setattr("nanobot.config.paths.ensure_dir", lambda p: p)

    assert get_workspace_path() == Path.home() / ".nanobot" / "workspace"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths() -> None:
    assert is_default_workspace(None) is True
    assert is_default_workspace(Path.home() / ".nanobot" / "workspace") is True
    assert is_default_workspace("~/custom-workspace") is False
