"""Tests for deterministic cron jobs (exec_command and skill_script)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.cron import CronTool
from nanobot.bus.events import OutboundMessage
from nanobot.cron.bound_runner import run_bound_deterministic_cron_job
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronPayload, CronSchedule


class _MockRecorder:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def write_run_record(self, run_id: str, record: dict[str, Any]) -> None:
        self.records[run_id] = record


def test_cron_payload_store_dict_roundtrip() -> None:
    payload = CronPayload(
        kind="exec_command",
        command="python test.py",
        skill_name="my_skill",
        script_name="sync.py",
        args=["--verbose", "--force"],
        quiet=True,
        session_key="telegram:12345",
        origin_channel="telegram",
        origin_chat_id="12345",
    )
    store_dict = {
        "kind": payload.kind,
        "command": payload.command,
        "skillName": payload.skill_name,
        "scriptName": payload.script_name,
        "args": payload.args,
        "quiet": payload.quiet,
        "sessionKey": payload.session_key,
        "originChannel": payload.origin_channel,
        "originChatId": payload.origin_chat_id,
    }
    restored = CronPayload.from_store_dict(store_dict)
    assert restored.kind == "exec_command"
    assert restored.command == "python test.py"
    assert restored.skill_name == "my_skill"
    assert restored.script_name == "sync.py"
    assert restored.args == ["--verbose", "--force"]
    assert restored.quiet is True
    assert restored.session_key == "telegram:12345"


def test_cron_tool_add_exec_command(tmp_path: Path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(service)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                command="echo 'hello world'",
                every_seconds=60,
            )
        )
    assert "Created job" in result
    jobs = service.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.kind == "exec_command"
    assert jobs[0].payload.command == "echo 'hello world'"

    list_output = tool._list_jobs()
    assert "Command: echo 'hello world'" in list_output

    # Verify persistence to disk and reloading from fresh service instance
    service._save_store()
    reloaded_service = CronService(tmp_path / "cron" / "jobs.json")
    reloaded_jobs = reloaded_service.list_jobs()
    assert len(reloaded_jobs) == 1
    assert reloaded_jobs[0].payload.kind == "exec_command"
    assert reloaded_jobs[0].payload.command == "echo 'hello world'"


def test_cron_tool_add_skill_script(tmp_path: Path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(service)

    with request_context(
        RequestContext(channel="discord", chat_id="456", session_key="discord:456")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                name="poll-leads",
                skill_name="poll-lead-sheets",
                script_name="poll.py",
                args=["--sheet", "inbound"],
                every_seconds=300,
            )
        )
    assert "Created job 'poll-leads'" in result
    jobs = service.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.kind == "skill_script"
    assert jobs[0].payload.skill_name == "poll-lead-sheets"
    assert jobs[0].payload.script_name == "poll.py"
    assert jobs[0].payload.args == ["--sheet", "inbound"]

    list_output = tool._list_jobs()
    assert "Skill Script: poll-lead-sheets/poll.py" in list_output

    # Verify persistence to disk and reloading from fresh service instance
    service._save_store()
    reloaded_service = CronService(tmp_path / "cron" / "jobs.json")
    reloaded_jobs = reloaded_service.list_jobs()
    assert len(reloaded_jobs) == 1
    assert reloaded_jobs[0].payload.kind == "skill_script"
    assert reloaded_jobs[0].payload.skill_name == "poll-lead-sheets"
    assert reloaded_jobs[0].payload.script_name == "poll.py"
    assert reloaded_jobs[0].payload.args == ["--sheet", "inbound"]


async def test_run_bound_deterministic_exec_command(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-cmd",
        name="test-cmd",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="echo 'deterministic task output'",
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert result == "deterministic task output"
    assert len(delivered_messages) == 1
    assert delivered_messages[0].content == "deterministic task output"
    assert delivered_messages[0].channel == "telegram"
    assert delivered_messages[0].chat_id == "123"

    run_record = list(recorder.records.values())[-1]
    assert run_record["status"] == "ok"
    assert run_record["response"] == "deterministic task output"


async def test_run_bound_deterministic_exec_command_failure(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-fail",
        name="test-fail",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="sh -c 'echo \"syntax error occurred\" >&2; exit 2'",
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    with pytest.raises(RuntimeError, match="syntax error occurred"):
        await run_bound_deterministic_cron_job(
            job,
            workspace=tmp_path,
            deliver_callback=deliver,
            cron=recorder,
        )

    assert len(delivered_messages) == 1
    assert "failed (code 2)" in delivered_messages[0].content
    assert "syntax error occurred" in delivered_messages[0].content

    run_record = list(recorder.records.values())[-1]
    assert run_record["status"] == "error"


async def test_run_bound_deterministic_skill_script(tmp_path: Path) -> None:
    # Create mock skill with script in workspace
    script_dir = tmp_path / "skills" / "demo-skill" / "scripts"
    script_dir.mkdir(parents=True)
    script_file = script_dir / "demo.py"
    script_file.write_text("import sys\nprint('demo output:', ' '.join(sys.argv[1:]))\n")

    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-skill",
        name="test-skill",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="skill_script",
            skill_name="demo-skill",
            script_name="demo.py",
            args=["param1", "param2"],
            session_key="discord:456",
            origin_channel="discord",
            origin_chat_id="456",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert result == "demo output: param1 param2"
    assert len(delivered_messages) == 1
    assert delivered_messages[0].content == "demo output: param1 param2"
    assert delivered_messages[0].channel == "discord"
    assert delivered_messages[0].chat_id == "456"


async def test_run_bound_deterministic_exec_command_quiet_empty_skips_delivery(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-cmd-quiet-empty",
        name="test-cmd-quiet-empty",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="true",
            quiet=True,
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert "completed with no output" in (result or "")
    assert len(delivered_messages) == 0  # Delivery was skipped!

    run_record = list(recorder.records.values())[-1]
    assert run_record["status"] == "ok"


async def test_run_bound_deterministic_exec_command_quiet_with_output_delivers(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-cmd-quiet-output",
        name="test-cmd-quiet-output",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="echo 'non-empty report'",
            quiet=True,
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert result == "non-empty report"
    assert len(delivered_messages) == 1
    assert delivered_messages[0].content == "non-empty report"


async def test_run_bound_deterministic_exec_command_quiet_with_stderr_delivers(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-cmd-quiet-stderr",
        name="test-cmd-quiet-stderr",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="echo 'warning from stderr' >&2",
            quiet=True,
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert "warning from stderr" in (result or "")
    assert len(delivered_messages) == 1
    assert "warning from stderr" in delivered_messages[0].content


async def test_run_bound_deterministic_skill_script_quiet_empty_skips_delivery(tmp_path: Path) -> None:
    script_dir = tmp_path / "skills" / "silent-skill" / "scripts"
    script_dir.mkdir(parents=True)
    script_file = script_dir / "silent.py"
    script_file.write_text("pass\n")

    recorder = _MockRecorder()
    delivered_messages: list[OutboundMessage] = []

    async def deliver(msg: OutboundMessage, **_kwargs: Any) -> None:
        delivered_messages.append(msg)

    job = CronJob(
        id="test-skill-quiet",
        name="test-skill-quiet",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="skill_script",
            skill_name="silent-skill",
            script_name="silent.py",
            quiet=True,
            session_key="discord:456",
            origin_channel="discord",
            origin_chat_id="456",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=deliver,
        cron=recorder,
    )

    assert "executed successfully with no output" in (result or "")
    assert len(delivered_messages) == 0  # Delivery was skipped!


def test_cron_tool_add_with_quiet_flag(tmp_path: Path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(service)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                command="echo 'check'",
                quiet=True,
                every_seconds=60,
            )
        )
    assert "Created job" in result
    jobs = service.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.quiet is True

    list_output = tool._list_jobs()
    assert "Quiet: True" in list_output


async def test_run_bound_deterministic_exec_command_with_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from unittest.mock import MagicMock

    from nanobot.agent.tools.shell import ExecToolConfig

    executed_commands: list[str] = []

    async def mock_create_subprocess_shell(cmd: str, **kwargs: Any) -> Any:
        executed_commands.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(return_value=asyncio.sleep(0, result=(b"SANDBOX EXEC OK\n", b"")))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    monkeypatch.setattr(sys, "platform", "linux")

    recorder = _MockRecorder()
    job = CronJob(
        id="test-sandbox-cmd",
        name="test-sandbox-cmd",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="echo 'sandboxed task'",
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    exec_cfg = ExecToolConfig(sandbox="bwrap", sandbox_ro_binds=["/extra/ro"])
    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        cron=recorder,
        exec_config=exec_cfg,
    )

    assert result == "SANDBOX EXEC OK"
    assert len(executed_commands) == 1
    assert executed_commands[0].startswith("bwrap ")
    assert "sandboxed task" in executed_commands[0]


async def test_run_bound_deterministic_skill_script_with_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from unittest.mock import MagicMock

    from nanobot.agent.tools.shell import ExecToolConfig

    skill_dir = tmp_path / "skills" / "sandboxed_skill" / "scripts"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "run.py").write_text("print('test')\n")

    executed_commands: list[str] = []

    async def mock_create_subprocess_shell(cmd: str, **kwargs: Any) -> Any:
        executed_commands.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(return_value=asyncio.sleep(0, result=(b"SANDBOX SKILL OK\n", b"")))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_create_subprocess_shell)
    monkeypatch.setattr(sys, "platform", "linux")

    recorder = _MockRecorder()
    job = CronJob(
        id="test-sandbox-skill",
        name="test-sandbox-skill",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="skill_script",
            skill_name="sandboxed_skill",
            script_name="run.py",
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
        ),
    )

    exec_cfg = ExecToolConfig(sandbox="bwrap", sandbox_ro_binds=["/extra/ro"])
    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        cron=recorder,
        exec_config=exec_cfg,
    )

    assert result == "SANDBOX SKILL OK"
    assert len(executed_commands) == 1
    assert executed_commands[0].startswith("bwrap ")
    assert "run.py" in executed_commands[0]


async def test_cron_tool_add_direct_message(tmp_path: Path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(service)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = await tool.execute(
            action="add",
            message="Daily standup in 5 minutes!",
            direct=True,
            every_seconds=3600,
        )
    assert "Created job" in result
    jobs = service.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.kind == "direct_message"
    assert jobs[0].payload.message == "Daily standup in 5 minutes!"

    list_output = tool._list_jobs()
    assert "Direct Message: Daily standup in 5 minutes!" in list_output


async def test_cron_tool_add_custom_destination_and_validation(tmp_path: Path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(service)

    # Validation: channel without chat_id
    errs = tool.validate_params({"action": "add", "message": "hi", "channel": "slack"})
    assert any("both 'channel' and 'chat_id' are required" in e for e in errs)

    # Validation: chat_id without channel
    errs = tool.validate_params({"action": "add", "message": "hi", "chat_id": "C123"})
    assert any("both 'channel' and 'chat_id' are required" in e for e in errs)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = await tool.execute(
            action="add",
            message="Weekly report",
            direct=True,
            channel="slack",
            chat_id="C012345678",
            thread_id="1712345678.000100",
            record_session=False,
            every_seconds=3600,
        )
    assert "Created job" in result
    jobs = service.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.target_channel == "slack"
    assert jobs[0].payload.target_chat_id == "C012345678"
    assert jobs[0].payload.target_thread_id == "1712345678.000100"
    assert jobs[0].payload.record_session is False

    list_output = tool._list_jobs()
    assert "Target: slack:C012345678 (thread: 1712345678.000100)" in list_output
    assert "Record Session: False" in list_output


async def test_run_bound_deterministic_direct_message(tmp_path: Path) -> None:
    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    job = CronJob(
        id="test-dm-job",
        name="test-dm-job",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="direct_message",
            message="Time for daily sync",
            session_key="telegram:123",
            origin_channel="telegram",
            origin_chat_id="123",
            target_channel="slack",
            target_chat_id="C999",
            target_thread_id="123.456",
            record_session=True,
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    assert result == "Time for daily sync"
    assert len(delivered_messages) == 1
    msg, record, session_key = delivered_messages[0]
    assert msg.channel == "slack"
    assert msg.chat_id == "C999"
    assert msg.content == "Time for daily sync"
    assert msg.metadata.get("thread_id") == "123.456"
    assert msg.metadata.get("thread_ts") == "123.456"  # slack specific mapping
    assert record is True
    assert session_key == "slack:C999"


async def test_run_bound_deterministic_destination_routing_success_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    # 1. Success with output -> delivered to TARGET
    async def mock_exec_ok(cmd: str, **kwargs: Any) -> Any:
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(return_value=asyncio.sleep(0, result=(b"METRICS: 100\n", b"")))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_exec_ok)

    job_ok = CronJob(
        id="test-dest-ok",
        name="test-dest-ok",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python get_metrics.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="discord",
            target_chat_id="metrics-channel",
            record_session=False,
        ),
    )

    res_ok = await run_bound_deterministic_cron_job(
        job_ok,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )
    assert res_ok == "METRICS: 100"
    assert len(delivered_messages) == 1
    msg, record, session_key = delivered_messages.pop()
    assert msg.channel == "discord"
    assert msg.chat_id == "metrics-channel"
    assert msg.content == "METRICS: 100"
    assert record is False
    assert session_key == "discord:metrics-channel"

    # 2. Failure with error -> delivered to ORIGIN (admin chat)
    async def mock_exec_err(cmd: str, **kwargs: Any) -> Any:
        proc = MagicMock()
        proc.returncode = 1
        proc.communicate = MagicMock(return_value=asyncio.sleep(0, result=(b"", b"SyntaxError in script\n")))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_exec_err)

    job_err = CronJob(
        id="test-dest-err",
        name="test-dest-err",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python broken_script.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="discord",
            target_chat_id="metrics-channel",
        ),
    )

    with pytest.raises(RuntimeError, match="SyntaxError in script"):
        await run_bound_deterministic_cron_job(
            job_err,
            workspace=tmp_path,
            deliver_callback=mock_deliver,
            cron=recorder,
        )

    # The error message should route to ORIGIN (telegram:admin), NOT discord
    assert len(delivered_messages) == 1
    msg, record, session_key = delivered_messages.pop()
    assert msg.channel == "telegram"
    assert msg.chat_id == "admin"
    assert "SyntaxError in script" in msg.content
    assert session_key == "telegram:admin"


async def test_run_bound_deterministic_destination_routing_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    async def mock_exec_empty(cmd: str, **kwargs: Any) -> Any:
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(return_value=asyncio.sleep(0, result=(b"", b"")))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_exec_empty)

    # 1. quiet=False with custom destination -> delivers notice to ORIGIN, not target
    job_no_output_not_quiet = CronJob(
        id="test-no-out-not-quiet",
        name="test-no-out-not-quiet",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python poll.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="discord",
            target_chat_id="announcements",
            quiet=False,
        ),
    )

    await run_bound_deterministic_cron_job(
        job_no_output_not_quiet,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    assert len(delivered_messages) == 1
    msg, record, session_key = delivered_messages.pop()
    assert msg.channel == "telegram"
    assert msg.chat_id == "admin"
    assert "completed with no output" in msg.content
    assert session_key == "telegram:admin"

    # 2. quiet=True with custom destination -> completely silent (0 deliveries)
    job_no_output_quiet = CronJob(
        id="test-no-out-quiet",
        name="test-no-out-quiet",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python poll.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="discord",
            target_chat_id="announcements",
            quiet=True,
        ),
    )

    await run_bound_deterministic_cron_job(
        job_no_output_quiet,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    assert len(delivered_messages) == 0


async def test_cron_payload_store_persistence(tmp_path: Path) -> None:
    store_file = tmp_path / "cron" / "jobs.json"
    service = CronService(store_file)
    service._running = True
    service.add_job(
        name="dest-job",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="Test Msg",
        kind="direct_message",
        session_key="telegram:123",
        origin_channel="telegram",
        origin_chat_id="123",
        target_channel="slack",
        target_chat_id="C555",
        target_thread_id="thread-888",
        record_session=False,
    )

    # Load in a fresh service
    reloaded_service = CronService(store_file)
    jobs = reloaded_service.list_jobs()
    assert len(jobs) == 1
    j = jobs[0]
    assert j.name == "dest-job"
    assert j.payload.kind == "direct_message"
    assert j.payload.message == "Test Msg"
    assert j.payload.session_key == "telegram:123"
    assert j.payload.origin_channel == "telegram"
    assert j.payload.origin_chat_id == "123"
    assert j.payload.target_channel == "slack"
    assert j.payload.target_chat_id == "C555"
    assert j.payload.target_thread_id == "thread-888"
    assert j.payload.record_session is False


async def test_run_bound_deterministic_destination_routing_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    async def mock_exec(cmd: str, **kwargs: Any) -> Any:
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(
            return_value=asyncio.sleep(0, result=(b"NEW LEADS: 5\n", b"API WARNING: token expiring\n"))
        )
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_exec)

    job = CronJob(
        id="poll-leads-allday",
        name="poll-leads-allday",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python poll_leads.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="telegram",
            target_chat_id="staff-group",
            record_session=False,
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    # The result in audit log / return contains both
    assert "NEW LEADS: 5" in (result or "")
    assert "API WARNING: token expiring" in (result or "")

    # Exactly 2 messages delivered: stdout to target, stderr to origin
    assert len(delivered_messages) == 2

    # Message 1: stdout to target (staff-group)
    msg_target, record_target, target_session = delivered_messages[0]
    assert msg_target.channel == "telegram"
    assert msg_target.chat_id == "staff-group"
    assert msg_target.content == "NEW LEADS: 5"
    assert "API WARNING" not in msg_target.content
    assert target_session == "telegram:staff-group"
    assert record_target is False

    # Message 2: stderr to origin (admin)
    msg_origin, record_origin, origin_session = delivered_messages[1]
    assert msg_origin.channel == "telegram"
    assert msg_origin.chat_id == "admin"
    assert "API WARNING: token expiring" in msg_origin.content
    assert "⚠️ Scheduled task 'poll-leads-allday' [stderr]:" in msg_origin.content
    assert origin_session == "telegram:admin"
    assert record_origin is True


async def test_run_bound_deterministic_destination_routing_stderr_only_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    async def mock_exec(cmd: str, **kwargs: Any) -> Any:
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = MagicMock(
            return_value=asyncio.sleep(0, result=(b"", b"POLL ERROR: connection timeout\n"))
        )
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", mock_exec)

    job = CronJob(
        id="poll-leads-timeout",
        name="poll-leads-timeout",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="exec_command",
            command="python poll_leads.py",
            session_key="telegram:admin",
            origin_channel="telegram",
            origin_chat_id="admin",
            target_channel="telegram",
            target_chat_id="staff-group",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    assert "POLL ERROR: connection timeout" in (result or "")

    # Target (staff group) receives NOTHING
    # Origin (admin) receives stderr notification
    assert len(delivered_messages) == 1
    msg, record, session_key = delivered_messages[0]
    assert msg.channel == "telegram"
    assert msg.chat_id == "admin"
    assert "POLL ERROR: connection timeout" in msg.content
    assert session_key == "telegram:admin"
    assert record is True


async def test_run_bound_deterministic_skill_script_destination_routing_stdout_and_stderr(
    tmp_path: Path,
) -> None:
    # Create skill script that outputs to both stdout and stderr
    script_dir = tmp_path / "skills" / "leads-skill" / "scripts"
    script_dir.mkdir(parents=True)
    script_file = script_dir / "leads.py"
    script_file.write_text(
        "import sys\n"
        "sys.stdout.write('LEADS PROCESSED: 3\\n')\n"
        "sys.stderr.write('WARN: rate limited on batch 2\\n')\n"
    )

    recorder = _MockRecorder()
    delivered_messages: list[tuple[OutboundMessage, bool, str | None]] = []

    async def mock_deliver(msg: OutboundMessage, *, record: bool = False, session_key: str | None = None) -> None:
        delivered_messages.append((msg, record, session_key))

    job = CronJob(
        id="skill-leads-dest",
        name="skill-leads-dest",
        schedule=CronSchedule(kind="every", every_ms=60000),
        payload=CronPayload(
            kind="skill_script",
            skill_name="leads-skill",
            script_name="leads.py",
            session_key="slack:admin-channel",
            origin_channel="slack",
            origin_chat_id="admin-channel",
            target_channel="slack",
            target_chat_id="staff-channel",
        ),
    )

    result = await run_bound_deterministic_cron_job(
        job,
        workspace=tmp_path,
        deliver_callback=mock_deliver,
        cron=recorder,
    )

    assert "LEADS PROCESSED: 3" in (result or "")
    assert "WARN: rate limited" in (result or "")

    # 2 messages: stdout to staff-channel, stderr to admin-channel
    assert len(delivered_messages) == 2

    target_msg, _, target_key = delivered_messages[0]
    assert target_msg.chat_id == "staff-channel"
    assert target_msg.content == "LEADS PROCESSED: 3"
    assert "WARN:" not in target_msg.content

    origin_msg, _, origin_key = delivered_messages[1]
    assert origin_msg.chat_id == "admin-channel"
    assert "WARN: rate limited on batch 2" in origin_msg.content
    assert "⚠️ Scheduled task 'skill-leads-dest' [stderr]:" in origin_msg.content


def test_cron_tool_add_workflow_recurring_ssot(tmp_path: Path) -> None:
    from nanobot.workflow.schema import WorkflowDefinition
    from nanobot.workflow.service import WorkflowService

    cron_service = CronService(tmp_path / "cron" / "jobs.json")
    wf_service = WorkflowService(workflows_dir=tmp_path / "workflows", cron_service=cron_service)

    # 1. Create a workflow
    wf = WorkflowDefinition(
        id="monitor_flow",
        name="Monitor Flow",
        start_at="done",
        states={"done": {"type": "succeed"}},
    )
    wf_service.save_workflow(wf)
    assert wf_service.get_workflow("monitor_flow") is not None

    tool = CronTool(cron_service, default_timezone="UTC", workflow_service=wf_service)

    # 2. Schedule workflow recurring via cron tool (SSOT facade)
    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                workflow_id="monitor_flow",
                cron_expr="*/15 * * * *",
            )
        )

    assert "Workflow 'monitor_flow' recurring schedule set to '*/15 * * * *'" in result
    assert "single source of truth (SSOT)" in result

    # 3. Verify workflow definition's trigger was updated
    saved_wf = wf_service.get_workflow("monitor_flow")
    assert saved_wf is not None
    assert saved_wf.trigger is not None
    assert saved_wf.trigger.cron == "*/15 * * * *"
    assert saved_wf.trigger.enabled is True

    # 4. Verify system cron job is registered
    jobs = cron_service.list_jobs()
    system_job = next((j for j in jobs if j.id == "workflow:monitor_flow"), None)
    assert system_job is not None
    assert system_job.payload.kind == "workflow"
    assert system_job.payload.workflow_id == "monitor_flow"

    # 5. Verify list output
    listing = tool._list_jobs()
    assert "Workflow: monitor_flow (deterministic execution)" in listing

    # 6. Remove via cron tool clears the trigger
    remove_res = asyncio.run(tool.execute(action="remove", job_id="workflow:monitor_flow"))
    assert "Removed schedule for workflow 'monitor_flow'" in remove_res
    cleared_wf = wf_service.get_workflow("monitor_flow")
    assert cleared_wf is not None
    assert cleared_wf.trigger is None


def test_cron_tool_add_workflow_one_shot(tmp_path: Path) -> None:
    from nanobot.workflow.schema import WorkflowDefinition
    from nanobot.workflow.service import WorkflowService

    cron_service = CronService(tmp_path / "cron" / "jobs.json")
    wf_service = WorkflowService(workflows_dir=tmp_path / "workflows", cron_service=cron_service)

    wf = WorkflowDefinition(
        id="oneshot_flow",
        name="One-Shot Flow",
        start_at="done",
        states={"done": {"type": "succeed"}},
    )
    wf_service.save_workflow(wf)

    tool = CronTool(cron_service, default_timezone="UTC", workflow_service=wf_service)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                workflow_id="oneshot_flow",
                at="2035-06-01T10:00:00",
            )
        )

    assert "Scheduled one-shot deterministic run for workflow 'oneshot_flow'" in result
    assert "delete automatically after run" in result

    # One-shot does NOT alter the permanent definition trigger
    saved_wf = wf_service.get_workflow("oneshot_flow")
    assert saved_wf is not None
    assert saved_wf.trigger is None

    # Job is in cron store with delete_after_run
    jobs = cron_service.list_jobs()
    oneshot_job = next((j for j in jobs if j.payload.workflow_id == "oneshot_flow"), None)
    assert oneshot_job is not None
    assert oneshot_job.delete_after_run is True
    assert oneshot_job.payload.kind == "workflow"


def test_cron_tool_workflow_not_found(tmp_path: Path) -> None:
    from nanobot.workflow.service import WorkflowService

    cron_service = CronService(tmp_path / "cron" / "jobs.json")
    wf_service = WorkflowService(workflows_dir=tmp_path / "workflows", cron_service=cron_service)
    tool = CronTool(cron_service, default_timezone="UTC", workflow_service=wf_service)

    with request_context(
        RequestContext(channel="telegram", chat_id="123", session_key="telegram:123")
    ):
        result = asyncio.run(
            tool.execute(
                action="add",
                workflow_id="nonexistent_flow",
                cron_expr="0 9 * * *",
            )
        )
    assert "Error: workflow 'nonexistent_flow' not found" in result






