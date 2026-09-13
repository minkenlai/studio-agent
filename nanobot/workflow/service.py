"""Workflow service for managing workflow definitions, persistence, and execution runs."""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.config.paths import get_config_path
from nanobot.workflow.engine import WorkflowEngine, WorkflowExecutionError
from nanobot.workflow.schema import (
    ChoiceState,
    ExecState,
    FailState,
    LLMState,
    SucceedState,
    TaskState,
    ToolState,
    WorkflowDefinition,
    WorkflowRunRecord,
)


class WorkflowService:
    """Service for persisting workflows, tracking runs, and executing workflows."""

    def __init__(
        self,
        workflows_dir: Path | str | None = None,
        *,
        engine: WorkflowEngine | None = None,
        agent_loop: Any | None = None,
        cron_service: Any | None = None,
    ) -> None:
        if workflows_dir is not None:
            self.workflows_dir = Path(workflows_dir).expanduser().resolve()
        else:
            self.workflows_dir = (get_config_path().parent / "workflows").resolve()

        self.runs_dir = self.workflows_dir / "runs"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.agent_loop = agent_loop
        self.cron_service = cron_service
        self.engine = engine or WorkflowEngine(dispatcher=self._default_dispatcher)

    async def _default_dispatcher(
        self, action: str, prompt: str | None, params: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Default action dispatcher integrating with AgentLoop and tools when available."""
        act = (action or "").strip().lower()

        # 1. Pass / no-op
        # 1. Pass / No-op
        if act == "pass":
            return params.get("result", {"status": "ok"})

        # 2. Exec / Shell command
        if act == "exec":
            cmd = params.get("command") or prompt
            if not cmd:
                raise WorkflowExecutionError("Missing required parameter 'command' for exec action")
            import subprocess

            proc = subprocess.run(
                str(cmd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=params.get("timeout", 60),
            )
            stdout_str = proc.stdout.strip()
            res: dict[str, Any] = {
                "exit_code": proc.returncode,
                "stdout": stdout_str,
                "stderr": proc.stderr.strip(),
            }
            if stdout_str.startswith(("{", "[")):
                try:
                    import json

                    parsed = json.loads(stdout_str)
                    if isinstance(parsed, dict):
                        res["json"] = parsed
                        for k, v in cast(dict[str, Any], parsed).items():
                            if k not in res:
                                res[k] = v
                    else:
                        res["json"] = parsed
                except Exception:
                    pass
            return res

        # 3. Direct agent tool execution
        if act == "tool":
            tool_name = params.get("tool") or params.get("name")
            if not tool_name:
                raise WorkflowExecutionError("Missing required parameter 'tool' for tool action")
            if self.agent_loop is not None and getattr(self.agent_loop, "tools", None) is not None:
                tool = self.agent_loop.tools.get(str(tool_name))
                if tool is not None:
                    raw_args = params.get("args")
                    tool_args: dict[str, Any] = (
                        cast(dict[str, Any], raw_args)
                        if isinstance(raw_args, dict)
                        else {k: v for k, v in params.items() if k not in ("name", "tool", "args")}
                    )
                    res = await tool.execute(**tool_args)
                    return str(res)
                raise WorkflowExecutionError(f"Tool '{tool_name}' not found in registered tools")
            raise WorkflowExecutionError("Agent loop or tool registry not available to execute tool action")

        # 4. Message dispatch
        if act == "send_message":
            channel = params.get("channel", "system")
            chat_id = params.get("chat_id", "default")
            content = params.get("content", prompt or params.get("message") or "")
            if self.agent_loop is not None and getattr(self.agent_loop, "bus", None) is not None:
                from nanobot.bus.events import OutboundMessage

                outbound = OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=str(content),
                )
                await self.agent_loop.bus.publish(outbound)
                return {"status": "sent", "channel": channel, "chat_id": chat_id}
            logger.info("[Workflow Message] [{}:{}] {}", channel, chat_id, content)
            return {"status": "logged", "channel": channel, "chat_id": chat_id, "content": content}

        # 5. LLM turn / Prompt execution
        if act == "prompt":
            prompt_text = prompt or params.get("prompt")
            if not prompt_text:
                raise WorkflowExecutionError("Missing required parameter 'prompt' for prompt action")

            if self.agent_loop is not None:
                channel = params.get("channel", "workflow")
                chat_id = params.get("chat_id", "system")
                session_key = params.get("session_key", f"workflow:{context.get('_workflow_id', 'flow')}")
                outbound = await self.agent_loop.process_direct(
                    content=str(prompt_text),
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                )
                return outbound.content if outbound else ""
            raise WorkflowExecutionError(
                "Agent loop is not available to execute LLM turn. Ensure WorkflowService is bound to an active AgentLoop."
            )

        raise WorkflowExecutionError(
            f"Unsupported action '{action}'. Valid actions are: 'prompt', 'tool', 'exec', 'send_message', 'pass'."
        )

    # ------------------------------------------------------------------
    # Workflow Definition Persistence
    # ------------------------------------------------------------------

    def _file_for_workflow(self, workflow_id: str) -> Path:
        clean_id = "".join(c for c in workflow_id if c.isalnum() or c in ("-", "_")).strip() or "unnamed"
        return self.workflows_dir / f"{clean_id}.json"

    def list_workflows(self) -> list[dict[str, Any]]:
        """List all workflow definitions."""
        results: list[dict[str, Any]] = []
        for file in sorted(self.workflows_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "id" in data:
                    results.append(cast(dict[str, Any], data))
            except Exception as exc:
                logger.warning("Failed to parse workflow file {}: {}", file, exc)
        return results

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None:
        """Load a workflow definition by ID."""
        file = self._file_for_workflow(workflow_id)
        if not file.is_file():
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            return WorkflowDefinition.model_validate(data)
        except Exception as exc:
            logger.warning("Failed to load workflow {}: {}", workflow_id, exc)
            return None

    def validate_workflow(
        self, definition: dict[str, Any] | WorkflowDefinition
    ) -> dict[str, Any]:
        """Validate a workflow definition for schema compliance and graph integrity."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema parsing
        wf: WorkflowDefinition | None = None
        if isinstance(definition, WorkflowDefinition):
            wf = definition
        else:
            try:
                wf = WorkflowDefinition.model_validate(definition)
            except Exception as exc:
                raw_id = definition.get("id")
                return {
                    "valid": False,
                    "errors": [f"Schema validation failed: {exc}"],
                    "warnings": [],
                    "workflow_id": str(raw_id) if raw_id else None,
                    "state_count": 0,
                    "states": [],
                }

        state_names = set(wf.states.keys())
        if not state_names:
            errors.append("Workflow has no states defined.")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "workflow_id": wf.id,
                "state_count": 0,
                "states": [],
            }

        # 2. Start state check
        if wf.start_at not in state_names:
            errors.append(f"Initial state 'start_at': '{wf.start_at}' is not defined in 'states'.")

        # 3. Transitions & Parameter validation per state
        for name, state in wf.states.items():
            if isinstance(state, ChoiceState):
                if not state.choices and not state.default:
                    errors.append(f"Choice state '{name}' has no choice rules and no default transition.")
                for idx, rule in enumerate(state.choices):
                    if rule.next not in state_names:
                        errors.append(
                            f"Choice state '{name}' rule #{idx + 1} targets unknown state '{rule.next}'."
                        )
                if state.default and state.default not in state_names:
                    errors.append(
                        f"Choice state '{name}' default targets unknown state '{state.default}'."
                    )
            elif isinstance(state, (FailState, SucceedState)):
                # Terminal states do not have next transitions
                pass
            else:
                is_end = getattr(state, "end", False)
                next_target = getattr(state, "next", None)
                if not is_end:
                    if not next_target:
                        errors.append(
                            f"State '{name}' is non-terminal (end=false) but has no 'next' transition specified."
                        )
                    elif next_target not in state_names:
                        errors.append(
                            f"State '{name}' transitions to unknown state '{next_target}'."
                        )

            # Check required parameters for task variants
            if isinstance(state, ExecState):
                if not state.command:
                    errors.append(f"Exec state '{name}' is missing required 'command' field.")
            elif isinstance(state, LLMState):
                if not state.prompt:
                    errors.append(f"LLM state '{name}' is missing required 'prompt' field.")
            elif isinstance(state, ToolState):
                if not state.tool:
                    errors.append(f"Tool state '{name}' is missing required 'tool' name field.")
            elif isinstance(state, TaskState):
                act = (state.action or "").strip().lower()
                cmd = state.command or state.params.get("command") or state.params.get("cmd")
                prompt = state.prompt or state.params.get("prompt")
                tool = state.tool or state.params.get("tool") or state.params.get("name")

                if act == "exec" and not cmd and not state.prompt:
                    errors.append(f"Task state '{name}' (action='exec') is missing required 'command'.")
                elif act == "tool" and not tool:
                    errors.append(f"Task state '{name}' (action='tool') is missing required 'tool' name.")
                elif act == "prompt" and not prompt and not state.action:
                    errors.append(f"Task state '{name}' (action='prompt') is missing required 'prompt'.")

        # 4. Graph reachability check (warn about unreachable states)
        reachable: set[str] = set()
        queue = [wf.start_at] if wf.start_at in state_names else []
        while queue:
            curr = queue.pop(0)
            if curr in reachable or curr not in state_names:
                continue
            reachable.add(curr)
            curr_state = wf.states[curr]
            if isinstance(curr_state, ChoiceState):
                for rule in curr_state.choices:
                    if rule.next in state_names and rule.next not in reachable:
                        queue.append(rule.next)
                if curr_state.default and curr_state.default in state_names and curr_state.default not in reachable:
                    queue.append(curr_state.default)
            elif hasattr(curr_state, "next") and getattr(curr_state, "next", None):
                nxt = str(getattr(curr_state, "next"))
                if nxt in state_names and nxt not in reachable:
                    queue.append(nxt)

        unreachable = sorted(list(state_names - reachable))
        if unreachable:
            warnings.append(
                f"The following states are defined but unreachable from 'start_at': {', '.join(unreachable)}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "workflow_id": wf.id,
            "state_count": len(state_names),
            "states": sorted(list(state_names)),
        }


    def save_workflow(self, workflow: WorkflowDefinition) -> WorkflowDefinition:
        """Persist or update a workflow definition."""
        file = self._file_for_workflow(workflow.id)
        content = workflow.model_dump_json(indent=2)
        tmp = file.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, file)
        logger.info("Saved workflow '{}' to {}", workflow.id, file)
        self.sync_cron_jobs()
        return workflow

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow definition."""
        file = self._file_for_workflow(workflow_id)
        if file.is_file():
            file.unlink(missing_ok=True)
            logger.info("Deleted workflow '{}'", workflow_id)
            if self.cron_service is not None:
                with suppress(Exception):
                    self.cron_service.remove_system_job(f"workflow:{workflow_id}")
                    store = self.cron_service._require_store()
                    store.jobs = [
                        j
                        for j in store.jobs
                        if not (
                            j.payload.kind == "workflow"
                            and (j.payload.workflow_id == workflow_id or j.payload.message == workflow_id)
                        )
                    ]
                    self.cron_service._save_store()
            return True
        return False

    def sync_cron_jobs(self) -> None:
        """Synchronize workflow cron triggers with the CronService."""
        if self.cron_service is None:
            return
        from nanobot.cron.types import CronJob, CronPayload, CronSchedule

        for wf_dict in self.list_workflows():
            wf_id = str(wf_dict.get("id", ""))
            trigger_raw = wf_dict.get("trigger")
            trigger: dict[str, Any] = cast(dict[str, Any], trigger_raw) if isinstance(trigger_raw, dict) else {}
            cron_expr = trigger.get("cron")
            is_enabled = trigger.get("enabled", True)
            job_id = f"workflow:{wf_id}"
            if cron_expr and is_enabled:
                tz = trigger.get("tz")
                self.cron_service.register_system_job(
                    CronJob(
                        id=job_id,
                        name=f"Workflow: {wf_dict.get('name', wf_id)}",
                        schedule=CronSchedule(kind="cron", expr=str(cron_expr), tz=str(tz) if tz else None),
                        payload=CronPayload(
                            kind="workflow",
                            workflow_id=wf_id,
                            message=wf_id,
                        ),
                    )
                )
            else:
                with suppress(Exception):
                    self.cron_service.remove_system_job(job_id)

    # ------------------------------------------------------------------
    # Execution Runs
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow_id: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowRunRecord:
        """Execute a workflow and save its run telemetry record."""
        workflow = self.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        init_ctx = dict(initial_context or {})
        init_ctx["_workflow_id"] = workflow.id

        record = await self.engine.run(workflow, init_ctx)
        self._save_run(record)
        return record

    def _file_for_run(self, run_id: str) -> Path:
        clean_id = "".join(c for c in run_id if c.isalnum() or c in ("-", "_")).strip() or "unknown_run"
        return self.runs_dir / f"{clean_id}.json"

    def _save_run(self, record: WorkflowRunRecord) -> None:
        """Persist a run record to disk."""
        file = self._file_for_run(record.run_id)
        content = record.model_dump_json(indent=2)
        tmp = file.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, file)

    def get_run(self, run_id: str, workflow_id: str | None = None) -> WorkflowRunRecord | None:
        """Get an execution run by ID."""
        file = self._file_for_run(run_id)
        if not file.is_file():
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            record = WorkflowRunRecord.model_validate(data)
            if workflow_id and record.workflow_id != workflow_id:
                return None
            return record
        except Exception as exc:
            logger.warning("Failed to load workflow run {}: {}", run_id, exc)
            return None

    def list_runs(
        self, workflow_id: str | None = None, limit: int = 50
    ) -> list[WorkflowRunRecord]:
        """List execution runs, sorted from newest to oldest."""
        runs: list[WorkflowRunRecord] = []
        files = sorted(self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for file in files:
            if len(runs) >= limit:
                break
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                record = WorkflowRunRecord.model_validate(data)
                if workflow_id is None or record.workflow_id == workflow_id:
                    runs.append(record)
            except Exception:
                continue
        return runs
