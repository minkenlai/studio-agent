# Task Workflows

<!-- Meta description: Create, schedule, and visually inspect graph-based task workflows with conditional branching and JSONPath data passing in nanobot. -->

Task workflows in nanobot provide a declarative, graph-based state machine engine inspired by Amazon States Language (ASL-Lite). Workflows allow multi-step automations where tasks can pass results to downstream steps, evaluate conditional branching paths, loop safely under circuit-breaker limits, run on recurring cron schedules, and be visually inspected or edited in the WebUI.

---

## Key Features

- **Declarative Graph Structure**: Workflows are defined in a JSON state machine format consisting of discrete states (`task`, `choice`, `pass`, `fail`, `succeed`).
- **Data Passing & Interpolation**: Results from tasks can be scoped and passed to downstream tasks using JSONPath (`result_path`, `parameters`, `payload_template`) and template syntax (`{{$.step1.summary}}`).
- **Conditional Branching**: `choice` states evaluate data comparisons (`equals`, `numeric_gt`, `numeric_lt`, `contains`, `is_null`, and boolean matches) to select the next state.
- **Circuit-Breaker Safety**: Configurable `max_steps` (default: 100) prevents infinite loops, and `timeout_seconds` guards against hung executions.
- **Scheduled Triggers**: Scheduled workflows sync directly with nanobot's `CronService` to kick off runs on intervals or cron expressions.
- **Agent Tool (`manage_workflow`)**: LLM agents can inspect, author, execute, and monitor workflows conversationally.
- **WebUI Workflow Studio**: Interactive React Flow visualizer with Dagre auto-layout, condition edge labels, execution replay, state inspector, and JSON/visual editor at `#/workflows`.

---

## State Machine Schema

A workflow is stored as a JSON document in `~/.nanobot/workflows/<workflow_id>.json`:

```json
{
  "id": "daily-report-flow",
  "name": "Daily Report Workflow",
  "description": "Fetches project updates, analyzes blockers, and alerts if needed",
  "start_at": "FetchMetrics",
  "timeout_seconds": 300,
  "max_steps": 50,
  "triggers": [
    {
      "type": "cron",
      "cron": "0 9 * * 1-5"
    }
  ],
  "states": {
    "FetchMetrics": {
      "type": "task",
      "action": "prompt",
      "prompt": "Summarize outstanding pull requests and failing CI runs in repo.",
      "result_path": "$.metrics",
      "next": "CheckBlockers"
    },
    "CheckBlockers": {
      "type": "choice",
      "choices": [
        {
          "variable": "$.metrics.has_blockers",
          "equals": true,
          "next": "AlertTeam"
        }
      ],
      "default": "LogClean"
    },
    "AlertTeam": {
      "type": "task",
      "action": "send_message",
      "params": {
        "channel": "slack",
        "chat_id": "team-alerts",
        "message": "Blockers detected in morning check: {{$.metrics.summary}}"
      },
      "next": "Complete"
    },
    "LogClean": {
      "type": "pass",
      "result": { "status": "clean" },
      "result_path": "$.status",
      "next": "Complete"
    },
    "Complete": {
      "type": "succeed"
    }
  }
}
```

---

## State Types

Workflows support standard Amazon States Language (ASL) task states as well as first-class shortcut states for concise definitions:

| State Type | Purpose | Key Fields |
|---|---|---|
| `task` | Standard ASL task executing an action or resource | `action` (`prompt`, `exec`, `tool`, `send_message`, `pass`), `resource`, `command`, `tool`, `args`, `prompt`, `params`, `result_path`, `next`, `end` |
| `llm` | First-class shortcut for an LLM prompt turn | `prompt`, `params`, `result_path`, `next`, `end` |
| `exec` | First-class shortcut for executing a shell command | `command`, `params`, `result_path`, `next`, `end` |
| `tool` | First-class shortcut for invoking an agent tool | `tool` (tool name), `args`, `params`, `result_path`, `next`, `end` |
| `choice` | Evaluates conditional logic to branch | `choices` (list of comparison rules), `default` |
| `pass` | Transforms data or injects static payload | `result`, `result_path`, `next`, `end` |
| `fail` | Explicitly terminates run with error status | `error`, `cause` |
| `succeed` | Explicitly terminates run with success status | `comment` |

### Deterministic Action Dispatching

For `task`, `llm`, `exec`, and `tool` states, execution is strictly deterministic:
- **`prompt` / `llm`**: Submits the interpolated `prompt` to the live agent loop (`process_direct`).
- **`exec`**: Runs the shell `command` in a subprocess with timeout guard, returning `{exit_code, stdout, stderr}`.
- **`tool`**: Executes the named agent tool (`tool`) with provided arguments (`args`).
- **`send_message`**: Publishes an outbound message to the configured channel (`channel`, `chat_id`, `content`).
- **`pass`**: Passes or transforms state data into `result_path` without external execution.

If a required field is missing or an unrecognized action is provided, execution returns an explicit, deterministic error explaining the exact missing parameter.

### Choice State Operators

Comparison rules in `choices` support:
- `equals` / `not_equals`: Exact string, number, or boolean equality.
- `numeric_gt` / `numeric_gte`: Greater than (or equal to) comparisons.
- `numeric_lt` / `numeric_lte`: Less than (or equal to) comparisons.
- `contains`: Substring matching or array membership.
- `starts_with` / `ends_with`: String prefix and suffix matching.
- `is_null` / `is_present`: Field nullability and presence checks.

---

## Managing Workflows via the Agent

Agents can manage and execute workflows using the `manage_workflow` tool:

```python
# List available workflows
manage_workflow(action="list")

# Get definition
manage_workflow(action="get", workflow_id="daily-report-flow")

# Validate workflow definition before saving (checks schema, dangling transitions, and reachability)
manage_workflow(
    action="validate",
    definition={
        "id": "quick-check",
        "name": "Quick Health Check",
        "start_at": "RunTests",
        "states": {
            "RunTests": {
                "type": "exec",
                "command": "pytest -q tests/test_sanity.py",
                "result_path": "$.test_out",
                "next": "Analyze"
            },
            "Analyze": {
                "type": "llm",
                "prompt": "Evaluate test results: {{$.test_out.stdout}}",
                "result_path": "$.analysis",
                "end": True
            }
        }
    }
)

# Create or update a workflow (supports standard ASL task or first-class shortcuts)
manage_workflow(
    action="save",
    definition={
        "id": "quick-check",
        "name": "Quick Health Check",
        "start_at": "RunTests",
        "states": {
            "RunTests": {
                "type": "exec",
                "command": "pytest -q tests/test_sanity.py",
                "result_path": "$.test_out",
                "next": "Analyze"
            },
            "Analyze": {
                "type": "llm",
                "prompt": "Evaluate test results: {{$.test_out.stdout}}",
                "result_path": "$.analysis",
                "end": True
            }
        }
    }
)

# Trigger a workflow run with initial context
manage_workflow(
    action="run",
    workflow_id="quick-check",
    initial_context={"env": "production"}
)

# View recent execution runs
manage_workflow(action="runs", workflow_id="quick-check")

# Configure automated recurring schedule directly (SSOT)
manage_workflow(action="schedule", workflow_id="quick-check", cron="*/15 * * * *", tz="America/Los_Angeles")

# Disable or re-enable schedule without altering workflow definition
manage_workflow(action="schedule", workflow_id="quick-check", enabled=False)
manage_workflow(action="schedule", workflow_id="quick-check", enabled=True)

# Clear schedule
manage_workflow(action="schedule", workflow_id="quick-check", clear=True)
```

### Scheduling Workflows Deterministically (No LLM Chat Noise)

Workflows execute **deterministically in the background** without wrapping them in an LLM agent turn. Wrapping a workflow inside an agent turn prompt (e.g. `cron(action="add", message="Run workflow X")`) forces the LLM to wake up on every tick, run the workflow via tool call, and generate a chat message (like *"no activity"*), cluttering the chat.

Instead, use native workflow scheduling:

1. **Directly via `manage_workflow(action="schedule")`**:
   Sets the canonical schedule in the workflow definition (`trigger: {"cron": "...", "tz": "...", "enabled": true}`) and immediately registers/synchronizes the system cron job.
2. **Via `cron` Tool**:
   - **Recurring (`cron_expr`)**: `cron(action="add", workflow_id="quick-check", cron_expr="0 9 * * 1-5")` acts as an SSOT facade that updates the workflow definition's `trigger` directly.
   - **One-time (`at`)**: `cron(action="add", workflow_id="quick-check", at="2026-09-15T09:00:00")` schedules an ephemeral one-shot run that runs deterministically and auto-deletes upon completion without modifying the workflow definition.
   - **Removal**: `cron(action="remove", job_id="workflow:quick-check")` removes the schedule and unregisters the job.

---

## WebUI Workflow Studio

The WebUI includes a dedicated Workflow Studio accessible from the sidebar navigation or by visiting `#/workflows`:

1. **Workflow Selector & Actions**:
   - Filter and switch between workflows.
   - Run workflows on-demand with custom initial JSON context.
   - Create new workflows or edit existing definitions in the Visual/JSON Editor.
2. **Graph Visualizer**:
   - Built on React Flow with Dagre auto-layout.
   - Toggle between Vertical (Top-to-Bottom) and Horizontal (Left-to-Right) layouts.
   - Displays action icons, status badges, condition edge labels, and step execution glow.
3. **Execution Replay**:
   - Selecting a run from the history sidebar visualizes the exact execution path taken on the graph.
   - Traversed steps display execution latency and completion status.
4. **State Inspector**:
   - Click any node to open the inspector drawer.
   - Inspect node parameters, prompts, choice conditions, input data, and output data.

---

## Storage and Persistence

- **Workflow Definitions**: `~/.nanobot/workflows/<workflow_id>.json`
- **Execution Run Records**: `~/.nanobot/workflows/runs/<run_id>.json`
- **Durability**: Writes use atomic file operations (`.tmp` write + POSIX rename) to guarantee persistence across power loss or application restarts.
- **Identifier Sanitization**: Workflow and run IDs are sanitized defensively to prevent path traversal.

---

## Configuration

Workflows are enabled by default and can be configured in `~/.nanobot/config.json`:

```json
{
  "workflows": {
    "enabled": true
  }
}
```

| Setting | Default | Description |
|---|---|---|
| `workflows.enabled` | `true` | Enable declarative task workflows, agent tool, and cron sync. |

When `"enabled": false` (e.g. on lightweight, public, or customer-facing Guide nodes):
- The `manage_workflow` tool is excluded from the agent's tool registry, saving ~150 prompt tokens per turn and closing the tool calling surface.
- Gateway cron synchronization for workflows is skipped.
- WebUI workflow API endpoints return a disabled response.

