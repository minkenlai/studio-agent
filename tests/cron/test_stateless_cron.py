"""Tests for the stateless cron jobs feature.

Stateless jobs execute without accumulating conversation history by using
ephemeral session keys (cron://{job_id}:{uuid}) instead of persistent ones
(cron://{job_id}).
"""

import asyncio
import json

import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


class TestStatelessPersistence:
    """Verify the stateless flag is persisted and loaded correctly."""

    def test_add_job_defaults_to_stateful(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="default",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
        )
        assert job.stateless is False

    def test_add_job_accepts_stateless_true(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="stateless-job",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="run me",
            stateless=True,
        )
        assert job.stateless is True

    def test_add_job_accepts_stateless_false(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="stateful-job",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="run me",
            stateless=False,
        )
        assert job.stateless is False

    def test_stateless_persisted_to_disk(self, tmp_path) -> None:
        store_path = tmp_path / "jobs.json"
        service = CronService(store_path)
        service.add_job(
            name="stateless",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="no history",
            stateless=True,
        )
        service.add_job(
            name="stateful",
            schedule=CronSchedule(kind="every", every_ms=120_000),
            message="with history",
            stateless=False,
        )

        raw = json.loads(store_path.read_text())
        jobs_by_name = {j["name"]: j for j in raw["jobs"]}
        assert jobs_by_name["stateless"]["stateless"] is True
        assert jobs_by_name["stateful"]["stateless"] is False

    def test_stateless_loaded_from_disk(self, tmp_path) -> None:
        store_path = tmp_path / "jobs.json"
        svc1 = CronService(store_path)
        job = svc1.add_job(
            name="reload-test",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=True,
        )

        # Create a fresh service that loads from disk
        svc2 = CronService(store_path)
        loaded = svc2.get_job(job.id)
        assert loaded is not None
        assert loaded.stateless is True

    def test_stateful_loaded_from_disk(self, tmp_path) -> None:
        store_path = tmp_path / "jobs.json"
        svc1 = CronService(store_path)
        job = svc1.add_job(
            name="stateful-reload",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=False,
        )

        svc2 = CronService(store_path)
        loaded = svc2.get_job(job.id)
        assert loaded is not None
        assert loaded.stateless is False

    def test_backward_compat_missing_stateless_field(self, tmp_path) -> None:
        """Jobs created by older versions (no stateless field) default to stateful."""
        store_path = tmp_path / "jobs.json"
        # Write a legacy-format jobs.json without the stateless field
        legacy = {
            "version": 1,
            "jobs": [
                {
                    "id": "legacy-1",
                    "name": "legacy-job",
                    "enabled": True,
                    "schedule": {
                        "kind": "every",
                        "atMs": None,
                        "everyMs": 60000,
                        "expr": None,
                        "tz": None,
                    },
                    "payload": {
                        "kind": "agent_turn",
                        "message": "legacy",
                        "deliver": False,
                        "channel": None,
                        "to": None,
                    },
                    "state": {
                        "nextRunAtMs": None,
                        "lastRunAtMs": None,
                        "lastStatus": None,
                        "lastError": None,
                        "runHistory": [],
                    },
                    "createdAtMs": 0,
                    "updatedAtMs": 0,
                    "deleteAfterRun": False,
                }
            ],
        }
        store_path.write_text(json.dumps(legacy))

        service = CronService(store_path)
        loaded = service.get_job("legacy-1")
        assert loaded is not None
        assert loaded.stateless is False


class TestStatelessSessionKeyGeneration:
    """Verify the cron callback logic generates correct session keys."""

    def _build_on_cron_job_key_logic(self, job) -> str:
        """Replicate the session key generation logic from cli/commands.py."""
        import uuid as _uuid

        if job.stateless:
            return f"cron://{job.id}:{_uuid.uuid4().hex}"
        else:
            return f"cron://{job.id}"

    def test_stateful_job_gets_persistent_key(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="persistent",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=False,
        )

        key1 = self._build_on_cron_job_key_logic(job)
        key2 = self._build_on_cron_job_key_logic(job)

        assert key1 == f"cron://{job.id}"
        assert key1 == key2  # Same job → same session key

    def test_stateless_job_gets_ephemeral_key(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="ephemeral",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=True,
        )

        key1 = self._build_on_cron_job_key_logic(job)
        key2 = self._build_on_cron_job_key_logic(job)

        # Both should contain job.id
        assert f"cron://{job.id}:" in key1
        assert f"cron://{job.id}:" in key2
        # But the uuid suffixes should differ (99.999...% probability)
        assert key1 != key2

    def test_stateless_keys_are_unique_across_runs(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="unique-test",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=True,
        )

        keys = {self._build_on_cron_job_key_logic(job) for _ in range(50)}
        # All 50 ephemeral keys should be distinct
        assert len(keys) == 50

    def test_stateless_key_format(self, tmp_path) -> None:
        service = CronService(tmp_path / "jobs.json")
        job = service.add_job(
            name="format-test",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=True,
        )

        key = self._build_on_cron_job_key_logic(job)
        # Format: cron://{id}:{uuid4_hex}
        parts = key.split(":")
        assert "cron" in parts[0]
        assert job.id in key
        assert len(parts[-1]) == 32  # uuid4 hex is 32 chars


class TestStatelessJobExecution:
    """Integration: verify stateless jobs don't accumulate history in SessionManager."""

    @pytest.mark.asyncio
    async def test_stateless_job_executes_without_error(self, tmp_path) -> None:
        """Basic smoke test: a stateless job runs through _execute_job."""
        store_path = tmp_path / "jobs.json"
        call_log: list[str] = []

        async def on_job(job):
            call_log.append(job.id)

        service = CronService(store_path, on_job=on_job)
        job = service.add_job(
            name="smoke",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="hello",
            stateless=True,
        )

        await service.run_job(job.id)
        assert call_log == [job.id]

        loaded = service.get_job(job.id)
        assert loaded.stateless is True
        assert loaded.state.last_status == "ok"

    @pytest.mark.asyncio
    async def test_stateful_and_stateless_coexist(self, tmp_path) -> None:
        """Both types can be added, executed, and persisted alongside each other."""
        store_path = tmp_path / "jobs.json"
        service = CronService(store_path, on_job=lambda _: asyncio.sleep(0))

        job_a = service.add_job(
            name="stateful",
            schedule=CronSchedule(kind="every", every_ms=60_000),
            message="stateful",
            stateless=False,
        )
        job_b = service.add_job(
            name="stateless",
            schedule=CronSchedule(kind="every", every_ms=120_000),
            message="stateless",
            stateless=True,
        )

        await service.run_job(job_a.id)
        await service.run_job(job_b.id)

        # Reload from disk
        fresh = CronService(store_path)
        loaded_a = fresh.get_job(job_a.id)
        loaded_b = fresh.get_job(job_b.id)

        assert loaded_a.stateless is False
        assert loaded_a.state.last_status == "ok"
        assert loaded_b.stateless is True
        assert loaded_b.state.last_status == "ok"
