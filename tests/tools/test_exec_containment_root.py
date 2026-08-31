"""Tests for ExecTool.containment_root (Option B profile workspace).

With a profile working dir, exec runs from a subdirectory of the sandbox
while the guard boundary stays at the sandbox root. ``containment_root`` is
additive: when unset the legacy cwd-anchored behavior is preserved exactly.
"""

import pytest

from nanobot.agent.tools.shell import ExecTool

ROOT = "/home/nanobot/.nanobot/workspace"
SUB = f"{ROOT}/ws/experimental"


class TestContainmentRoot:
    """containment_root widens the guard boundary beyond the cwd."""

    @pytest.fixture
    def containment(self):
        # cwd in a subdirectory, boundary at the sandbox root
        return ExecTool(
            restrict_to_workspace=True,
            working_dir=SUB,
            containment_root=ROOT,
        )

    def test_path_inside_subdir_allowed(self, containment):
        assert containment._guard_command("cat ws/experimental/notes.md", SUB) is None

    def test_path_in_sibling_dir_allowed(self, containment):
        # Sibling of the subdir, still inside the root: allowed under
        # containment_root (would have been blocked by the legacy guard).
        assert containment._guard_command(f"ls {ROOT}/kb", SUB) is None

    def test_path_outside_root_still_blocked(self, containment):
        assert "blocked by safety guard" in containment._guard_command("cat /etc/passwd", SUB)

    def test_cwd_itself_still_allowed(self, containment):
        assert containment._guard_command(f"ls {SUB}", SUB) is None

    def test_traversal_still_blocked(self, containment):
        assert "path traversal" in containment._guard_command("cat ../../etc/passwd", SUB)


class TestLegacyBehaviorUnchanged:
    """Without containment_root, the guard is anchored to the cwd (as before)."""

    @pytest.fixture
    def legacy(self):
        return ExecTool(restrict_to_workspace=True, working_dir=ROOT)

    def test_sibling_dir_blocked(self, legacy):
        assert "blocked by safety guard" in legacy._guard_command(
            f"ls {ROOT}/kb", f"{ROOT}/nanobot-dev"
        )

    def test_path_inside_cwd_allowed(self, legacy):
        assert (
            legacy._guard_command(f"cat {ROOT}/nanobot-dev/README.md", f"{ROOT}/nanobot-dev")
            is None
        )


class TestContainmentRootNotRestricting:
    """containment_root only matters when restrict_to_workspace is on."""

    def test_no_restriction_ignores_containment(self):
        tool = ExecTool(restrict_to_workspace=False, working_dir=SUB, containment_root=ROOT)
        # Without restriction, absolute paths are never blocked by the guard
        assert tool._guard_command("cat /etc/hostname", SUB) is None
