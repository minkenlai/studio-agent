"""Shell execution tool."""

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        containment_root: str | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        # Containment boundary for the path guard when restrict_to_workspace
        # is on. Defaults to the working dir (legacy behavior). Subagents with
        # a profile workspace inside the shared sandbox pass the sandbox root
        # here so their cwd can sit in a subdirectory without shrinking the
        # boundary.
        self.containment_root = containment_root
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        # Known slash commands that should not be mistaken for absolute paths
        self._slash_command_names = frozenset(
            {
                "/new",
                "/help",
                "/status",
                "/stop",
                "/repl",
                "/restart",
                "/halt",
            }
        )
        # Benign null devices: redirects to these are harmless and must not trip
        # the path-outside-workspace guard (e.g. "cmd 2>/dev/null").
        self._benign_null_devices = frozenset(
            {
                "/dev/null",
                "/dev/zero",
                "/dev/full",
                "/dev/tty",
                "/dev/stdin",
                "/dev/stdout",
                "/dev/stderr",
            }
        )

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        # Audit logging: record the command execution
        import datetime

        try:
            # Use a stable path for the audit log relative to the workspace logs dir
            audit_path = Path("logs") / "audit.log"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] CWD: {cwd} | CMD: {command}\n")
        except Exception as e:
            logger.warning("Failed to write to audit.log: {}", e)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError) as e:
                            logger.debug("Process already reaped or not found: {}", e)
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from nanobot.security.network import contains_internal_url

        if contains_internal_url(cmd):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()
            root_path = (
                Path(self.containment_root).expanduser().resolve()
                if self.containment_root
                else cwd_path
            )

            for raw in self._filter_benign_tokens(self._extract_absolute_paths(cmd)):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and root_path not in p.parents and p != root_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)  # Windows: C:\...
        posix_paths = re.findall(
            r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command
        )  # POSIX: /absolute only
        home_paths = re.findall(
            r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command
        )  # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths

    def _filter_benign_tokens(self, paths: list[str]) -> list[str]:
        """Remove known slash commands and benign null devices from extracted paths.

        Both are harmless tokens that the path extractor can mistake for absolute
        paths (e.g. the redirect in "cmd 2>/dev/null"), which would otherwise trip
        the path-outside-workspace guard.
        """
        benign = self._slash_command_names | self._benign_null_devices
        return [p for p in paths if p not in benign]
