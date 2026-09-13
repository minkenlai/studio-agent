"""Tool for creating, managing, and running declarative state machine workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.schema import (
    BooleanSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.workflow.schema import TriggerConfig, WorkflowDefinition
from nanobot.workflow.service import WorkflowService


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "Action to perform: 'list' (list workflows), 'get' (view workflow definition), "
            "'save' (create or update workflow definition), 'delete' (remove workflow), "
            "'run' (execute a workflow), 'runs' (view execution run logs), 'validate' (check workflow validity), "
            "or 'schedule' (configure automated recurring cron trigger for a workflow)."
        ),
        workflow_id=StringSchema(
            "Unique workflow identifier (e.g. 'morning_triage'). Required for get, delete, run, and schedule."
        ),
        cron=StringSchema(
            "Standard cron expression (e.g. '0 9 * * *' or '*/15 * * * *') for 'schedule' action."
        ),
        tz=StringSchema(
            "Optional IANA timezone (e.g. 'UTC', 'America/Los_Angeles') for 'schedule' action."
        ),
        enabled=BooleanSchema(
            description="Whether automated trigger schedule is enabled (defaults to true). Used with 'schedule' action."
        ),
        clear=BooleanSchema(
            description="Optional flag for 'schedule' action. If true, removes the automated trigger schedule from the workflow."
        ),
        definition=ObjectSchema(
            description=(
                "Workflow definition dictionary containing 'id', 'name', 'start_at', and 'states'.\n"
                "Optional top-level fields: 'trigger' ({'cron': '*/15 * * * *', 'tz': 'UTC', 'enabled': true}), 'description'.\n"
                "State types & output contracts:\n"
                "1. 'exec' (or 'task' with action='exec'): Runs shell command. "
                "Output shape: {'exit_code': int, 'stdout': str, 'stderr': str, 'json': object|null}. "
                "If stdout is valid JSON, parsed fields are accessible via '$.data.json.<field>' or directly '$.data.<field>'.\n"
                "2. 'llm' (or 'task' with action='prompt'): Executes LLM turn. Stores generated text at 'result_path'. "
                "Use '{{$.path}}' or '${$.path}' to interpolate variables into prompt (e.g. 'Summarize: {{$.data.json.sessions}}'). Bare '$.var' without braces is not interpolated.\n"
                "3. 'tool' (or 'task' with action='tool'): Executes an agent tool: {'tool': 'read_file', 'args': {'path': 'file.txt'}, 'result_path': '$.file'}.\n"
                "4. 'choice': Conditional branching: {'choices': [{'variable': '$.data.exit_code', 'equals': 0, 'next': 'NextState'}], 'default': 'Fallback'}.\n"
                "5. 'pass': Static data injection: {'result': {...}, 'result_path': '$.data', 'next': '...'}.\n"
                "6. 'succeed': Terminal success state {'type': 'succeed'}.\n"
                "7. 'fail': Terminal failure state {'type': 'fail', 'error': 'Reason'}."
            ),
            additional_properties=True,
        ),
        initial_context=ObjectSchema(
            description="Optional initial key-value dictionary passed to workflow state (used with 'run').",
            additional_properties=True,
        ),
        run_id=StringSchema(
            "Optional specific run ID to view step-by-step telemetry (used with 'runs')."
        ),
        required=["action"],
    )
)
class ManageWorkflowTool(Tool):
    """Tool for creating, updating, inspecting, and triggering state machine workflows."""

    @property
    def name(self) -> str:
        return "manage_workflow"

    @property
    def description(self) -> str:
        return (
            "Create, inspect, modify, run, and monitor declarative task workflows. "
            "Workflows are state machines of tasks and conditional choices, allowing chaining "
            "outputs between LLM turns, tool calls, and notifications."
        )

    def __init__(self, service: WorkflowService | None = None, workflows_dir: Path | str | None = None) -> None:
        if service is not None:
            self.service = service
        else:
            self.service = WorkflowService(workflows_dir)

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        wf_cfg = getattr(ctx.config, "workflows", None)
        return bool(getattr(wf_cfg, "enabled", True))

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        workspace = Path(ctx.workspace) if ctx.workspace else None
        workflows_dir = (workspace / "workflows") if workspace else None
        agent_ctrl = getattr(ctx, "runtime_control", None)
        loop_obj = (
            getattr(ctx, "agent_loop", None)
            or getattr(agent_ctrl, "_AgentRuntimeControl__target", None)
            or getattr(agent_ctrl, "target", None)
            or getattr(agent_ctrl, "agent_loop", None)
        )
        cron_service = getattr(ctx, "cron_service", None)
        service = WorkflowService(workflows_dir, agent_loop=loop_obj, cron_service=cron_service)
        return cls(service=service)

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        workflow_id = str(kwargs.get("workflow_id", "")).strip()
        definition = kwargs.get("definition")
        initial_context = kwargs.get("initial_context")
        run_id = kwargs.get("run_id")

        if action == "list":
            workflows = self.service.list_workflows()
            if not workflows:
                return ToolResult("No workflows configured.")
            lines = [f"Found {len(workflows)} workflow(s):"]
            for wf in workflows:
                trigger_raw = wf.get("trigger")
                trigger: dict[str, Any] = cast(dict[str, Any], trigger_raw) if isinstance(trigger_raw, dict) else {}
                cron_str = f" [cron: {trigger.get('cron')}]" if trigger.get("cron") else ""
                lines.append(f"  • {wf.get('id')}: {wf.get('name')} (start: {wf.get('start_at')}){cron_str}")
            return ToolResult("\n".join(lines))

        elif action == "get":
            if not workflow_id:
                return ToolResult.error("workflow_id is required for 'get' action.")
            wf = self.service.get_workflow(workflow_id)
            if wf is None:
                return ToolResult.error(f"Workflow '{workflow_id}' not found.")
            return ToolResult(wf.model_dump_json(indent=2))

        elif action in ("save", "create", "update"):
            if not isinstance(definition, dict):
                return ToolResult.error("definition object is required when saving a workflow.")
            def_dict = cast(dict[str, Any], definition)
            raw_id = def_dict.get("id")
            if not workflow_id and raw_id:
                workflow_id = str(raw_id)
            if not workflow_id:
                return ToolResult.error("Workflow definition must have an 'id'.")
            def_dict["id"] = workflow_id
            try:
                parsed = WorkflowDefinition.model_validate(def_dict)
                self.service.save_workflow(parsed)
                return ToolResult(
                    f"Workflow '{parsed.id}' ('{parsed.name}') successfully saved with {len(parsed.states)} states."
                )
            except Exception as exc:
                return ToolResult.error(f"Invalid workflow definition: {exc}")

        elif action == "delete":
            if not workflow_id:
                return ToolResult.error("workflow_id is required for 'delete' action.")
            if self.service.delete_workflow(workflow_id):
                return ToolResult(f"Workflow '{workflow_id}' successfully deleted.")
            return ToolResult.error(f"Workflow '{workflow_id}' not found.")

        elif action == "run":
            if not workflow_id:
                return ToolResult.error("workflow_id is required for 'run' action.")
            try:
                init_ctx: dict[str, Any] | None = (
                    cast(dict[str, Any], initial_context) if isinstance(initial_context, dict) else None
                )
                run_record = await self.service.run_workflow(
                    workflow_id,
                    initial_context=init_ctx,
                )
                status_emoji = "✅" if run_record.status == "succeeded" else "❌"
                return ToolResult(
                    f"{status_emoji} Workflow '{workflow_id}' run completed with status: {run_record.status}\n"
                    f"  Run ID       : {run_record.run_id}\n"
                    f"  Duration     : {run_record.total_duration_ms:.1f}ms\n"
                    f"  Steps run    : {len(run_record.steps)}\n"
                    f"  Final context: {json.dumps(run_record.final_context, indent=2, ensure_ascii=False)}\n"
                    f"{'  Error: ' + run_record.error if run_record.error else ''}"
                )
            except Exception as exc:
                return ToolResult.error(f"Failed to execute workflow '{workflow_id}': {exc}")

        elif action == "runs":
            if run_id:
                record = self.service.get_run(str(run_id).strip())
                if record is None:
                    return ToolResult.error(f"Run '{run_id}' not found.")
                return ToolResult(record.model_dump_json(indent=2))
            runs = self.service.list_runs(workflow_id=workflow_id if workflow_id else None)
            if not runs:
                return ToolResult("No workflow runs found.")
            lines = [f"Recent runs ({len(runs)}):"]
            for r in runs:
                lines.append(
                    f"  • [{r.status.upper()}] {r.run_id} (wf: {r.workflow_id}, steps: {len(r.steps)}, {r.total_duration_ms:.1f}ms) @ {r.started_at}"
                )
            return ToolResult("\n".join(lines))

        elif action == "validate":
            target_def: dict[str, Any] | WorkflowDefinition | None = None
            if isinstance(definition, dict):
                target_def = cast(dict[str, Any], definition)
            elif workflow_id:
                target_def = self.service.get_workflow(workflow_id)
                if target_def is None:
                    return ToolResult.error(f"Workflow '{workflow_id}' not found to validate.")
            else:
                return ToolResult.error("Either 'definition' or 'workflow_id' is required for 'validate' action.")

            res = self.service.validate_workflow(target_def)
            wf_label = res.get("workflow_id") or "unnamed"
            warnings_list = cast(list[str], res.get("warnings", []))
            errors_list = cast(list[str], res.get("errors", []))
            states_list = cast(list[str], res.get("states", []))
            if res["valid"]:
                msg = f"✅ Workflow '{wf_label}' is valid ({res['state_count']} states: {', '.join(states_list)})."
                if warnings_list:
                    msg += "\nWarnings:\n" + "\n".join(f"  ⚠️ {w}" for w in warnings_list)
                return ToolResult(msg)
            else:
                err_lines = "\n".join(f"  ❌ {e}" for e in errors_list)
                msg = f"❌ Workflow '{wf_label}' validation failed with {len(errors_list)} error(s):\n{err_lines}"
                if warnings_list:
                    msg += "\nWarnings:\n" + "\n".join(f"  ⚠️ {w}" for w in warnings_list)
                return ToolResult.error(msg)

        elif action == "schedule":
            if not workflow_id:
                return ToolResult.error("workflow_id is required for 'schedule' action.")
            wf = self.service.get_workflow(workflow_id)
            if wf is None:
                return ToolResult.error(f"Workflow '{workflow_id}' not found.")
            cron_expr = kwargs.get("cron") or kwargs.get("cron_expr")
            tz_str = kwargs.get("tz")
            enabled = kwargs.get("enabled", True)
            clear = kwargs.get("clear", False)

            if clear:
                wf.trigger = None
                self.service.save_workflow(wf)
                return ToolResult(f"Schedule cleared for workflow '{workflow_id}'.")
            if cron_expr is None and not enabled and wf.trigger:
                wf.trigger.enabled = False
                self.service.save_workflow(wf)
                return ToolResult(f"Schedule disabled for workflow '{workflow_id}'.")
            if not cron_expr and wf.trigger and enabled:
                wf.trigger.enabled = True
                self.service.save_workflow(wf)
                return ToolResult(f"Schedule enabled for workflow '{workflow_id}' ({wf.trigger.cron}).")
            if not cron_expr:
                return ToolResult.error(
                    "cron expression (e.g. '0 9 * * *' or '*/15 * * * *') is required for 'schedule' action."
                )

            wf.trigger = TriggerConfig(
                cron=str(cron_expr),
                tz=str(tz_str) if tz_str else None,
                enabled=bool(enabled),
            )
            self.service.save_workflow(wf)
            tz_msg = f" ({tz_str})" if tz_str else ""
            status_msg = "enabled" if enabled else "disabled"
            return ToolResult(
                f"Workflow '{workflow_id}' recurring schedule set to '{cron_expr}'{tz_msg} ({status_msg}). "
                "Synchronized with deterministic system cron."
            )

        return ToolResult.error(
            f"Unknown action: {action}. Use 'list', 'get', 'save', 'delete', 'run', 'runs', 'validate', or 'schedule'."
        )
