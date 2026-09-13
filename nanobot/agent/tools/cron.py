"""Cron tool for scheduling reminders and tasks."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext, current_request_context
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob, CronJobState, CronSchedule
from nanobot.session.keys import UNIFIED_SESSION_KEY

_CRON_PARAMETERS = tool_parameters_schema(
    action=StringSchema("Action to perform", enum=["add", "list", "remove"]),
    name=StringSchema(
        "Optional short human-readable label for the job "
        "(e.g., 'weather-monitor', 'daily-standup'). Defaults to first 30 chars of message or command."
    ),
    message=StringSchema(
        "REQUIRED when action='add' (unless command, skill_name, or workflow_id is provided). "
        "Instruction for the agent to execute when the job triggers as an LLM agent turn "
        "(e.g., 'Send a reminder to WeChat: xxx' or 'Check system status and report'). "
        "Not used for action='list' or action='remove'."
    ),
    command=StringSchema(
        "Optional shell command to execute deterministically without invoking the LLM "
        "(e.g., 'python scripts/check_health.py')."
    ),
    workflow_id=StringSchema(
        "Optional workflow ID to execute deterministically without invoking the LLM (e.g. 'morning_triage'). "
        "Recurring schedules update the workflow definition's trigger (single source of truth), "
        "while one-shot schedules ('at') register an ephemeral run that automatically deletes after completion."
    ),
    skill_name=StringSchema(
        "Optional skill name for executing a pre-approved skill script deterministically "
        "(e.g., 'poll-lead-sheets'). Used together with script_name."
    ),
    script_name=StringSchema(
        "Optional script name within the skill's scripts/ directory (e.g., 'poll_lead_sheets.py'). "
        "Used together with skill_name."
    ),
    args=ArraySchema(
        items=StringSchema("Command line argument for skill script"),
        description="Optional list of string arguments passed to the skill script.",
    ),
    channel=StringSchema(
        "Optional target channel (e.g. 'telegram', 'discord', 'slack') for destination routing. "
        "When provided, stdout is delivered to the target while stderr/errors route to the creator's session. "
        "Defaults to current session."
    ),
    chat_id=StringSchema(
        "Optional target chat/group/channel ID for destination routing. "
        "Defaults to current session."
    ),
    thread_id=StringSchema(
        "Optional topic or thread ID for destination routing (e.g. Telegram message_thread_id, Slack thread_ts)."
    ),
    direct=BooleanSchema(
        description=(
            "Optional flag. If true with 'message', delivers the message directly as a static "
            "notification without invoking an LLM agent turn."
        )
    ),
    record_session=BooleanSchema(
        description=(
            "Optional flag. If true (default), records the delivered message into the destination "
            "session history so conversational context is preserved if a user replies. Set false for ephemeral pings."
        )
    ),
    quiet=BooleanSchema(
        description=(
            "Optional flag. If true, skips delivering notification to the originating session "
            "when execution completes with no output, but still delivers if there is output or an error."
        )
    ),
    every_seconds=IntegerSchema(description="Interval in seconds (for recurring tasks)"),
    cron_expr=StringSchema("Cron expression like '0 9 * * *' (for scheduled tasks)"),
    tz=StringSchema(
        "Optional IANA timezone for cron expressions (e.g. 'America/Vancouver'). "
        "When omitted with cron_expr, the tool's default timezone applies."
    ),
    at=StringSchema(
        "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00'). "
        "Naive values use the tool's default timezone."
    ),
    job_id=StringSchema("REQUIRED when action='remove'. Job ID to remove (obtain via action='list')."),
    required=["action"],
    description=(
        "Action-specific parameters: add requires a schedule (every_seconds, cron_expr, or at) "
        "plus one of: message (for agent turn), command (for shell command), "
        "workflow_id (for deterministic workflow execution), or "
        "skill_name + script_name (for skill script); remove requires job_id; list only needs action. "
        "Per-action requirements are enforced at runtime (see field descriptions) so the "
        "top-level schema stays compatible with providers (e.g. OpenAI Codex/Responses) that "
        "reject oneOf/anyOf/allOf/enum/not at the root of function parameters."
    ),
)


@tool_parameters(_CRON_PARAMETERS)
class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""

    def __init__(
        self,
        cron_service: CronService,
        default_timezone: str = "UTC",
        workflow_service: Any | None = None,
        workspace: str | Path | None = None,
    ):
        self._cron = cron_service
        self._default_timezone = default_timezone
        self._workflow_service = workflow_service
        self._workspace = workspace
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    def _get_workflow_service(self) -> Any:
        if self._workflow_service is None:
            from nanobot.workflow.service import WorkflowService

            workflows_dir = (Path(self._workspace) / "workflows") if self._workspace else None
            self._workflow_service = WorkflowService(workflows_dir, cron_service=self._cron)
        return self._workflow_service

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.cron_service is not None

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        cron_service = ctx.cron_service
        if cron_service is None:
            raise RuntimeError("CronTool requires an initialized cron service")
        workspace = Path(ctx.workspace) if ctx.workspace else None
        workflows_dir = (workspace / "workflows") if workspace else None
        agent_ctrl = getattr(ctx, "runtime_control", None)
        loop_obj = (
            getattr(ctx, "agent_loop", None)
            or getattr(agent_ctrl, "_AgentRuntimeControl__target", None)
            or getattr(agent_ctrl, "target", None)
            or getattr(agent_ctrl, "agent_loop", None)
        )
        wf_service = None
        try:
            from nanobot.workflow.service import WorkflowService

            wf_service = WorkflowService(workflows_dir, agent_loop=loop_obj, cron_service=cron_service)
        except Exception:
            pass
        return cls(
            cron_service=cron_service,
            default_timezone=ctx.timezone,
            workflow_service=wf_service,
            workspace=workspace,
        )

    @staticmethod
    def _request_route() -> tuple[str, str, str, dict[str, Any]]:
        """Return routing from the authoritative request snapshot."""
        ctx = current_request_context()
        if ctx is None:
            return "", "", "", {}
        raw_key = f"{ctx.channel}:{ctx.chat_id}" if ctx.channel and ctx.chat_id else ""
        session_key = (
            raw_key if ctx.session_key == UNIFIED_SESSION_KEY else (ctx.session_key or "")
        )
        return session_key, ctx.channel or "", ctx.chat_id or "", dict(ctx.metadata or {})

    def set_cron_context(self, active: bool) -> Token[bool]:
        """Mark whether the tool is executing inside a cron job callback."""
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token: Token[bool]) -> None:
        """Restore previous cron context."""
        self._in_cron_context.reset(token)

    @staticmethod
    def _validate_timezone(tz: str) -> str | None:
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(tz)
        except (KeyError, Exception):
            return ToolResult.error(f"Error: unknown timezone '{tz}'")
        return None

    def _display_timezone(self, schedule: CronSchedule) -> str:
        """Pick the most human-meaningful timezone for display."""
        return schedule.tz or self._default_timezone

    @staticmethod
    def _format_timestamp(ms: int, tz_name: str) -> str:
        from zoneinfo import ZoneInfo

        dt = datetime.fromtimestamp(ms / 1000, tz=ZoneInfo(tz_name))
        return f"{dt.isoformat()} ({tz_name})"

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule reminders and recurring tasks. Actions: add, list, remove. "
            f"If tz is omitted, cron expressions and naive ISO times default to {self._default_timezone}."
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        errors = super().validate_params(params)
        action = params.get("action")
        if action == "add":
            has_message = bool(str(params.get("message") or "").strip())
            has_command = bool(str(params.get("command") or "").strip())
            has_workflow = bool(str(params.get("workflow_id") or "").strip())
            skill_name = str(params.get("skill_name") or "").strip()
            script_name = str(params.get("script_name") or "").strip()
            has_skill = bool(skill_name and script_name)

            if not (has_message or has_command or has_skill or has_workflow):
                errors.append("message is required when action='add'")
            if (skill_name and not script_name) or (script_name and not skill_name):
                errors.append("both 'skill_name' and 'script_name' are required when scheduling a skill script")

            channel = str(params.get("channel") or "").strip()
            chat_id = str(params.get("chat_id") or "").strip()
            if (channel and not chat_id) or (chat_id and not channel):
                errors.append("both 'channel' and 'chat_id' are required when specifying a destination")
        if action == "remove" and not str(params.get("job_id") or "").strip():
            errors.append("job_id is required when action='remove'")
        return errors

    async def execute(
        self,
        action: str,
        name: str | None = None,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        command: str | None = None,
        skill_name: str | None = None,
        script_name: str | None = None,
        args: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
        direct: bool = False,
        record_session: bool = True,
        quiet: bool = False,
        workflow_id: str | None = None,
    ) -> str:
        if action == "add":
            if self._in_cron_context.get():
                return ToolResult.error("Error: cannot schedule new jobs from within a cron job execution")
            return self._add_job(
                name=name,
                message=message,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                tz=tz,
                at=at,
                command=command,
                skill_name=skill_name,
                script_name=script_name,
                args=args,
                channel=channel,
                chat_id=chat_id,
                thread_id=thread_id,
                direct=direct,
                record_session=record_session,
                quiet=quiet,
                workflow_id=workflow_id,
            )
        elif action == "list":
            return self._list_jobs()
        elif action == "remove":
            return self._remove_job(job_id)
        return f"Unknown action: {action}"

    def _add_job(
        self,
        name: str | None,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        command: str | None = None,
        skill_name: str | None = None,
        script_name: str | None = None,
        args: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
        direct: bool = False,
        record_session: bool = True,
        quiet: bool = False,
        workflow_id: str | None = None,
    ) -> str:
        command_clean = (command or "").strip()
        skill_clean = (skill_name or "").strip()
        script_clean = (script_name or "").strip()
        msg_clean = (message or "").strip()
        wf_clean = (workflow_id or "").strip()
        target_channel = (channel or "").strip() or None
        target_chat_id = (chat_id or "").strip() or None
        target_thread_id = (thread_id or "").strip() or None

        from typing import Literal

        if direct and msg_clean:
            kind: Literal["agent_turn", "exec_command", "skill_script", "direct_message", "workflow"] = "direct_message"
            default_name = f"msg: {msg_clean[:25]}"
        elif wf_clean:
            kind = "workflow"
            default_name = f"workflow: {wf_clean}"
        elif command_clean:
            kind = "exec_command"
            default_name = f"exec: {command_clean[:24]}"
        elif skill_clean and script_clean:
            kind = "skill_script"
            default_name = f"skill: {skill_clean}/{script_clean}"
        elif msg_clean:
            kind = "agent_turn"
            default_name = msg_clean[:30]
        else:
            return ToolResult.error(
                "Error: cron action='add' requires a non-empty 'message' parameter "
                "describing what to do when the job triggers (e.g. the reminder text), "
                "or 'command', or 'workflow_id', or 'skill_name' + 'script_name'. Retry including message=\"...\"."
            )

        # If scheduling a workflow, validate existence and apply SSOT for recurring
        if wf_clean:
            try:
                wf_service = self._get_workflow_service()
                wf = wf_service.get_workflow(wf_clean)
                if wf is None:
                    return ToolResult.error(f"Error: workflow '{wf_clean}' not found")
            except Exception as exc:
                return ToolResult.error(f"Error checking workflow '{wf_clean}': {exc}")

            if cron_expr or every_seconds:
                effective_cron = cron_expr
                if not effective_cron and every_seconds:
                    if every_seconds % 60 == 0:
                        mins = every_seconds // 60
                        effective_cron = f"*/{mins} * * * *" if mins < 60 else "0 * * * *"
                    else:
                        return ToolResult.error(
                            "Error: workflow cron schedules require minute resolution "
                            "(e.g. cron_expr='*/15 * * * *' or every_seconds divisible by 60)."
                        )
                effective_tz = tz or self._default_timezone
                if err := self._validate_timezone(effective_tz):
                    return err
                from nanobot.workflow.schema import TriggerConfig

                wf.trigger = TriggerConfig(cron=effective_cron, tz=effective_tz, enabled=True)
                wf_service.save_workflow(wf)
                return (
                    f"Workflow '{wf_clean}' recurring schedule set to '{effective_cron}' ({effective_tz}). "
                    "Updated workflow trigger definition as single source of truth (SSOT) and synchronized system cron."
                )

        session_key, origin_channel, origin_chat_id, origin_metadata = self._request_route()
        if not session_key or not origin_channel or not origin_chat_id:
            return ToolResult.error("Error: scheduled cron jobs must be created from a chat session")
        if tz and not cron_expr:
            return ToolResult.error("Error: tz can only be used with cron_expr")
        if tz:
            if err := self._validate_timezone(tz):
                return err

        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            effective_tz = tz or self._default_timezone
            if err := self._validate_timezone(effective_tz):
                return err
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=effective_tz)
        elif at:
            from zoneinfo import ZoneInfo

            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return ToolResult.error(f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS")
            if dt.tzinfo is None:
                if err := self._validate_timezone(self._default_timezone):
                    return err
                dt = dt.replace(tzinfo=ZoneInfo(self._default_timezone))
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return ToolResult.error("Error: either every_seconds, cron_expr, or at is required")

        job = self._cron.add_job(
            name=name or default_name,
            schedule=schedule,
            message=msg_clean or wf_clean,
            delete_after_run=delete_after,
            session_key=session_key,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            origin_metadata=origin_metadata,
            kind=kind,
            command=command_clean or None,
            skill_name=skill_clean or None,
            script_name=script_clean or None,
            workflow_id=wf_clean or None,
            args=args or [],
            quiet=quiet,
            target_channel=target_channel,
            target_chat_id=target_chat_id,
            target_thread_id=target_thread_id,
            record_session=record_session,
        )
        if kind == "workflow":
            return (
                f"Scheduled one-shot deterministic run for workflow '{wf_clean}' at {at} (job ID: {job.id}). "
                "Will execute deterministically without invoking the LLM and delete automatically after run."
            )
        return f"Created job '{job.name}' (id: {job.id})"

    def _format_timing(self, schedule: CronSchedule) -> str:
        """Format schedule as a human-readable timing string."""
        if schedule.kind == "cron":
            tz = f" ({schedule.tz})" if schedule.tz else ""
            return f"cron: {schedule.expr}{tz}"
        if schedule.kind == "every" and schedule.every_ms:
            ms = schedule.every_ms
            if ms % 3_600_000 == 0:
                return f"every {ms // 3_600_000}h"
            if ms % 60_000 == 0:
                return f"every {ms // 60_000}m"
            if ms % 1000 == 0:
                return f"every {ms // 1000}s"
            return f"every {ms}ms"
        if schedule.kind == "at" and schedule.at_ms:
            return f"at {self._format_timestamp(schedule.at_ms, self._display_timezone(schedule))}"
        return schedule.kind

    def _format_state(self, state: CronJobState, schedule: CronSchedule) -> list[str]:
        """Format job run state as display lines."""
        lines: list[str] = []
        display_tz = self._display_timezone(schedule)
        if state.last_run_at_ms:
            info = (
                f"  Last run: {self._format_timestamp(state.last_run_at_ms, display_tz)}"
                f" — {state.last_status or 'unknown'}"
            )
            if state.last_error:
                info += f" ({state.last_error})"
            lines.append(info)
        if state.next_run_at_ms:
            lines.append(f"  Next run: {self._format_timestamp(state.next_run_at_ms, display_tz)}")
        return lines

    @staticmethod
    def _system_job_purpose(job: CronJob) -> str:
        if job.name == "dream":
            return "Dream memory consolidation for long-term memory."
        return "System-managed internal job."

    def _list_jobs(self) -> str:
        jobs = self._cron.list_jobs()
        if not jobs:
            return "No scheduled jobs."
        lines: list[str] = []
        for j in jobs:
            timing = self._format_timing(j.schedule)
            parts = [f"- {j.name} (id: {j.id}, {timing})"]
            if j.payload.kind == "system_event":
                parts.append(f"  Purpose: {self._system_job_purpose(j)}")
                parts.append("  Protected: visible for inspection, but cannot be removed.")
            elif j.payload.kind == "workflow":
                wf_target = j.payload.workflow_id or j.payload.message
                parts.append(f"  Workflow: {wf_target} (deterministic execution)")
            elif j.payload.kind == "direct_message":
                parts.append(f"  Direct Message: {j.payload.message}")
            elif j.payload.kind == "exec_command":
                parts.append(f"  Command: {j.payload.command or j.payload.message}")
            elif j.payload.kind == "skill_script":
                parts.append(f"  Skill Script: {j.payload.skill_name}/{j.payload.script_name}")
            if j.payload.target_channel and j.payload.target_chat_id:
                thread_info = (
                    f" (thread: {j.payload.target_thread_id})"
                    if j.payload.target_thread_id
                    else ""
                )
                parts.append(f"  Target: {j.payload.target_channel}:{j.payload.target_chat_id}{thread_info}")
            if not j.payload.record_session:
                parts.append("  Record Session: False")
            if j.payload.quiet:
                parts.append("  Quiet: True (suppresses empty notifications)")
            parts.extend(self._format_state(j.state, j.schedule))
            lines.append("\n".join(parts))
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return ToolResult.error("Error: job_id is required for remove")
        target_id = job_id
        if not self._cron.get_job(target_id) and self._cron.get_job(f"workflow:{target_id}"):
            target_id = f"workflow:{target_id}"
        if target_id.startswith("workflow:"):
            wf_id = target_id.removeprefix("workflow:")
            try:
                wf_service = self._get_workflow_service()
                wf = wf_service.get_workflow(wf_id)
                if wf and wf.trigger:
                    wf.trigger = None
                    wf_service.save_workflow(wf)
                    return f"Removed schedule for workflow '{wf_id}'."
            except Exception:
                pass
        result = self._cron.remove_job(target_id)
        if result == "removed":
            return f"Removed job {target_id}"
        if result == "protected":
            job = self._cron.get_job(target_id)
            if job and job.name == "dream":
                return (
                    "Cannot remove job `dream`.\n"
                    "This is a system-managed Dream memory consolidation job for long-term memory.\n"
                    "It remains visible so you can inspect it, but it cannot be removed."
                )
            return (
                f"Cannot remove job `{target_id}`.\n"
                "This is a protected system-managed cron job."
            )
        return f"Job {target_id} not found"
