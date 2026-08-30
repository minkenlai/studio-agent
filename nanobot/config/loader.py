"""Configuration loading utilities."""

import json
import os
from pathlib import Path

import pydantic
from loguru import logger

from nanobot.config.schema import Config

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None

# Global variable to store overlay config path
_current_overlay_path: Path | None = None

# Global variable to store the secrets vault path (None = auto-detect).
_current_secrets_path: Path | None = None

# Mask values used for sensitive leaves in the config template.
MASKED_VALUES = ("***", "[REDACTED]", "<REDACTED>")

# Dotted paths filled by the most recent :func:`inject_secrets` call.
# Used by :func:`save_config` to re-mask in-memory secrets before writing.
_injected_secret_paths: set[str] = set()


def set_overlay_path(path: Path) -> None:
    """Set the current config overlay path."""
    global _current_overlay_path
    _current_overlay_path = path


def get_overlay_path() -> Path | None:
    """Get the config overlay path, checking global state then env var."""
    if _current_overlay_path:
        return _current_overlay_path
    env = os.environ.get("NANOBOT_CONFIG_OVERLAY")
    if env:
        return Path(env).expanduser()
    return None


class ConfigOverlayPolicyError(Exception):
    """Raised when a config overlay violates the base config's overlayPolicy
    and the policy's ``onViolation`` is ``fail``."""


def _get_dotted(node, dotted: str):
    """Return (found, value) for a dotted path in nested dicts."""
    cur = node
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _set_dotted_safe(node, dotted: str, value) -> None:
    """Set a dotted path, creating intermediate dicts as needed."""
    cur = node
    keys = dotted.split(".")
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge overlay into base (dicts merged, lists/scalars
    replaced). Mutates base in place."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _policy_value(policy_raw: dict, *names: str):
    """Read a policy field by any of the given camelCase/snake_case names."""
    for name in names:
        if name in policy_raw and policy_raw[name] is not None:
            return policy_raw[name]
    return None


def _enforce_overlay_policy(base: dict, overlay: dict, policy_raw: dict) -> dict:
    """Apply an overlay under the base config's overlayPolicy.

    - Only whitelisted top-level sections are applied (deep-merged).
    - ``overlayPolicy`` itself can never be set by an overlay.
    - ``deniedPaths`` are frozen: any change to them is rolled back.
    - ``onViolation``: ``warn`` (default) rejects the offending parts and
      continues; ``fail`` raises :class:`ConfigOverlayPolicyError`.
    """
    allowed = {
        str(s).lower()
        for s in (_policy_value(policy_raw, "allowedSections", "allowed_sections") or [])
    }
    denied = [
        str(p) for p in (_policy_value(policy_raw, "deniedPaths", "denied_paths") or []) if str(p)
    ]
    on_violation = str(_policy_value(policy_raw, "onViolation", "on_violation") or "warn").lower()
    if on_violation not in ("warn", "fail"):
        logger.warning(
            f"overlayPolicy.onViolation={on_violation!r} is not 'warn'|'fail' — using 'warn'."
        )
        on_violation = "warn"

    violations: list[str] = []

    # 1. Filter sections against the whitelist.
    clean: dict = {}
    for key, value in overlay.items():
        if key in ("overlayPolicy", "overlay_policy"):
            violations.append("'overlayPolicy' may not be set by an overlay")
        elif key.lower() in allowed:
            clean[key] = value
        else:
            violations.append(f"section {key!r} is not in allowedSections")

    # 2. Snapshot denied paths before merging.
    snapshots = {}
    for dotted in denied:
        found, value = _get_dotted(base, dotted)
        snapshots[dotted] = ("present", value) if found else ("absent", None)

    # 3. Deep-merge the clean sections.
    for key, value in clean.items():
        if not isinstance(value, dict):
            violations.append(f"section {key!r} must be a JSON object")
            continue
        node = base.get(key)
        if node is None:
            base[key] = {}
            node = base[key]
        if not isinstance(node, dict):
            violations.append(f"base section {key!r} is not an object; cannot merge")
            continue
        _deep_merge(node, value)

    # 4. Enforce frozen paths (roll back any change).
    for dotted, (state, value) in snapshots.items():
        found, now = _get_dotted(base, dotted)
        if state == "absent":
            if found:
                violations.append(f"deniedPath {dotted!r} was added by the overlay")
                _remove_dotted(base, dotted)
        elif found and now != value:
            violations.append(f"deniedPath {dotted!r} was modified by the overlay")
            _restore_dotted(base, dotted, value)

    if violations:
        message = "Config overlay policy violation: " + "; ".join(violations)
        if on_violation == "fail":
            logger.error(f"{message} — refusing to boot (onViolation=fail).")
            raise ConfigOverlayPolicyError(message)
        logger.warning(f"{message} — offending parts were not applied (onViolation=warn).")

    logger.info(
        f"Config overlay applied from {len(clean)}/{len(overlay)} sections "
        f"(policy: {len(allowed)} allowed, {len(denied)} frozen)."
    )
    return base


def _restore_dotted(node: dict, dotted: str, value) -> None:
    """Restore a frozen dotted path to its snapshot value (container-safe)."""
    keys = dotted.split(".")
    cur = node
    for key in keys[:-1]:
        if not isinstance(cur, dict) or not isinstance(cur.get(key), dict):
            return
        cur = cur[key]
    if isinstance(cur, dict):
        cur[keys[-1]] = value


def _remove_dotted(node: dict, dotted: str) -> None:
    """Remove a dotted path, pruning containers left empty by the removal."""
    keys = dotted.split(".")
    trail: list[dict] = [node]
    cur = node
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
        trail.append(cur)
        if not isinstance(cur, dict):
            return
    if keys[-1] not in cur:
        return
    del cur[keys[-1]]
    # Prune empty containers bottom-up (never prune the root).
    for i in range(len(trail) - 1, 0, -1):
        if trail[i]:
            break
        del trail[i - 1][keys[i - 1]]


def apply_overlay(base: dict, overlay_path: Path) -> dict:
    """
    Merge overlay JSON into the base config.

    If the base config carries an ``overlayPolicy``, the overlay is applied
    under that policy (whitelisted sections deep-merged, denied paths
    frozen). Without a policy, the legacy section-replace behavior applies.
    """
    if not overlay_path.exists():
        logger.warning(f"Config overlay not found: {overlay_path} — skipping.")
        return base
    try:
        with open(overlay_path, encoding="utf-8") as f:
            overlay = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load config overlay from {overlay_path}: {e} — skipping.")
        return base
    if not isinstance(overlay, dict):
        logger.warning(f"Config overlay {overlay_path} is not a JSON object — skipping.")
        return base

    policy_raw = base.get("overlayPolicy") or base.get("overlay_policy")
    if isinstance(policy_raw, dict):
        return _enforce_overlay_policy(base, overlay, policy_raw)

    logger.info(f"Applying config overlay from {overlay_path} (sections: {list(overlay.keys())})")
    base.update(overlay)
    return base


def set_secrets_path(path: Path) -> None:
    """Set the secrets vault path explicitly (overrides env/credentials dir)."""
    global _current_secrets_path
    _current_secrets_path = path


def get_secrets_path() -> Path | None:
    """Get the secrets vault path.

    Resolution order:
      1. Explicit global (``set_secrets_path``).
      2. ``NANOBOT_SECRETS`` environment variable.
      3. Systemd credentials tmpfs: ``$CREDENTIALS_DIRECTORY/secrets.json``
         (populated by ``LoadCredential=secrets.json:<vault>``).

    Returns ``None`` when no vault is available — injection is then skipped.
    """
    if _current_secrets_path is not None:
        return _current_secrets_path
    env = os.environ.get("NANOBOT_SECRETS")
    if env:
        return Path(env).expanduser()
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if cred_dir:
        path = Path(cred_dir) / "secrets.json"
        if path.exists():
            return path
    return None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path.

    Resolution order:
      1. Explicit global (``set_config_path``, e.g. ``--config``).
      2. Systemd credentials tmpfs: ``$CREDENTIALS_DIRECTORY/config.json``
         (populated by ``LoadCredential=config.json:<base>``) — same
         auto-detect pattern as the secrets vault.
      3. Default: ``~/.nanobot/config.json``.
    """
    if _current_config_path:
        return _current_config_path
    cred_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if cred_dir:
        path = Path(cred_dir) / "config.json"
        if path.exists():
            return path
    return Path.home() / ".nanobot" / "config.json"


def _set_dotted(data: dict, dotted: str, value: str) -> None:
    """Set a value in a nested dict by dotted path (creating dicts as needed)."""
    node = data
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _scrub_dotted(data: dict, dotted: str) -> bool:
    """Replace the value at a dotted path with the canonical mask.

    Returns True if a value was actually re-masked.
    """
    node = data
    keys = dotted.split(".")
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    if isinstance(node, dict) and keys[-1] in node:
        node[keys[-1]] = "***"
        return True
    return False


def _load_secrets_vault(path: Path | None) -> dict[str, str] | None:
    """Load the dotted-path secrets vault, or ``None`` if unavailable."""
    if path is None:
        return None
    if not path.exists():
        logger.warning(f"Secrets vault not found: {path} — skipping injection.")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load secrets vault {path}: {e} — skipping injection.")
        return None
    if not isinstance(raw, dict):
        logger.warning(f"Secrets vault {path} is not a JSON object — skipping injection.")
        return None
    # Keep only str -> str entries; the rest are logged and ignored.
    vault = {}
    for key, value in raw.items():
        if isinstance(value, str):
            vault[str(key)] = value
        else:
            logger.warning(f"Secrets vault entry {key!r} is not a string — ignored.")
    return vault or None


def inject_secrets(data: dict, vault: dict[str, str]) -> dict:
    """Inject vault values into the raw config dict, in memory only.

    Every masked sensitive leaf that has a vault entry is filled with the real
    value. Vault keys with no matching (masked) config leaf are warned about as
    drift. Returns the same dict, mutated, for convenience.
    """
    if not vault:
        return data
    injected = 0
    for dotted, value in vault.items():
        if not dotted:
            continue
        keys = dotted.split(".")
        node = data
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            logger.warning(f"Secrets vault key has no config leaf (drift?): {dotted}")
            continue
        current = node[keys[-1]]
        if current in MASKED_VALUES:
            # An empty vault value is an explicit "intentionally empty"
            # (disabled provider/channel), not a missing secret.
            node[keys[-1]] = value
            _injected_secret_paths.add(dotted)
            injected += 1
        elif current != value:
            logger.warning(
                f"Secrets vault key {dotted}: config value already set — not overridden."
            )
    # Report any masked sensitive leaves the vault could not fill.
    leftover = [p for p, _ in _masked_sensitive_leaves(data)]
    if leftover:
        for path in leftover:
            logger.warning(f"Sensitive config value still masked (no vault entry?): {path}")
    logger.info(f"Injected {injected}/{len(vault)} secrets from vault (in memory).")
    return data


def _masked_sensitive_leaves(node, prefix=""):
    """Yield (dotted_path, value) for sensitive leaves still holding a mask."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from _masked_sensitive_leaves(value, path)
            elif isinstance(value, str) and value in MASKED_VALUES:
                if path.lower().endswith(("apikey", "token", "secret", "password")):
                    yield path, value


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    If a secrets vault is available (see :func:`get_secrets_path`), masked
    sensitive values in the config are filled in **in memory only** — the
    config file on disk is never rewritten with real secrets.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    # Start from a clean slate: any save_config in this process must only
    # re-mask what *this* load injected.
    _injected_secret_paths.clear()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            overlay_path = get_overlay_path()
            if overlay_path:
                data = apply_overlay(data, overlay_path)
            vault = _load_secrets_vault(get_secrets_path())
            if vault:
                data = inject_secrets(data, vault)
            return Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Any secrets that were injected into memory from a vault (see
    :func:`inject_secrets`) are re-masked before writing, so real secret
    values never leak to disk. This only re-masks vault-injected paths —
    values a user typed directly (e.g. via the onboarding wizard, no vault)
    are saved normally.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)

    # Re-mask anything that only exists in memory via the vault, so real
    # secrets are never written to disk.
    remasked = 0
    for dotted in sorted(_injected_secret_paths):
        if _scrub_dotted(data, dotted):
            remasked += 1
    if remasked:
        logger.info(f"Re-masked {remasked} vault-injected secret(s) before save.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
