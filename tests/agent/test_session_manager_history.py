import json
from pathlib import Path

from nanobot.session.manager import Session, SessionManager


def _assert_no_orphans(history: list[dict]) -> None:
    """Assert every tool result in history has a matching assistant tool_call."""
    declared = {
        tc["id"]
        for m in history
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    orphans = [
        m.get("tool_call_id")
        for m in history
        if m.get("role") == "tool" and m.get("tool_call_id") not in declared
    ]
    assert orphans == [], f"orphan tool_call_ids: {orphans}"


def _tool_turn(prefix: str, idx: int) -> list[dict]:
    """Helper: one assistant with 2 tool_calls + 2 tool results."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"{prefix}_{idx}_a",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
                {
                    "id": f"{prefix}_{idx}_b",
                    "type": "function",
                    "function": {"name": "y", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": f"{prefix}_{idx}_a", "name": "x", "content": "ok"},
        {"role": "tool", "tool_call_id": f"{prefix}_{idx}_b", "name": "y", "content": "ok"},
    ]


# --- Original regression test (from PR 2075) ---


def test_get_history_drops_orphan_tool_results_when_window_cuts_tool_calls():
    session = Session(key="telegram:test")
    session.messages.append({"role": "user", "content": "old turn"})
    for i in range(20):
        session.messages.extend(_tool_turn("old", i))
    session.messages.append({"role": "user", "content": "problem turn"})
    for i in range(25):
        session.messages.extend(_tool_turn("cur", i))
    session.messages.append({"role": "user", "content": "new telegram question"})

    history = session.get_history(max_messages=100)
    _assert_no_orphans(history)


# --- Positive test: legitimate pairs survive trimming ---


def test_legitimate_tool_pairs_preserved_after_trim():
    """Complete tool-call groups within the window must not be dropped."""
    session = Session(key="test:positive")
    session.messages.append({"role": "user", "content": "hello"})
    for i in range(5):
        session.messages.extend(_tool_turn("ok", i))
    session.messages.append({"role": "assistant", "content": "done"})

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    tool_ids = [m["tool_call_id"] for m in history if m.get("role") == "tool"]
    assert len(tool_ids) == 10
    assert history[0]["role"] == "user"


def test_retain_recent_legal_suffix_keeps_recent_messages():
    session = Session(key="test:trim")
    for i in range(10):
        session.messages.append({"role": "user", "content": f"msg{i}"})

    session.retain_recent_legal_suffix(4)

    assert len(session.messages) == 4
    assert session.messages[0]["content"] == "msg6"
    assert session.messages[-1]["content"] == "msg9"


def test_retain_recent_legal_suffix_adjusts_staged_boundary():
    """retain drops 6 -> SB shifts down by 6 (recomputed, never above len, never below 0)."""
    session = Session(key="test:trim-cons")
    for i in range(10):
        session.messages.append({"role": "user", "content": f"msg{i}"})
    session.staged_boundary = 7

    session.retain_recent_legal_suffix(4)

    assert len(session.messages) == 4
    assert session.messages[0]["content"] == "msg6"
    assert session.staged_boundary == 1


def test_retain_recent_legal_suffix_sb_never_negative():
    """If SB would go below 0 after a drop, it clamps to 0."""
    session = Session(key="test:trim-cons-clamp")
    for i in range(10):
        session.messages.append({"role": "user", "content": f"msg{i}"})
    session.staged_boundary = 3

    session.retain_recent_legal_suffix(4)  # drops 6 > SB

    assert len(session.messages) == 4
    assert session.staged_boundary == 0


def test_retain_recent_legal_suffix_zero_clears_session():
    session = Session(key="test:trim-zero")
    for i in range(10):
        session.messages.append({"role": "user", "content": f"msg{i}"})
    session.staged_boundary = 5

    session.retain_recent_legal_suffix(0)

    assert session.messages == []
    assert session.staged_boundary == 0


def test_retain_recent_legal_suffix_keeps_legal_tool_boundary():
    session = Session(key="test:trim-tools")
    session.messages.append({"role": "user", "content": "old"})
    session.messages.extend(_tool_turn("old", 0))
    session.messages.append({"role": "user", "content": "keep"})
    session.messages.extend(_tool_turn("keep", 0))
    session.messages.append({"role": "assistant", "content": "done"})

    session.retain_recent_legal_suffix(4)

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "keep"


# --- last_consolidated > 0 ---


def test_orphan_trim_with_last_consolidated():
    """Orphan trimming works correctly when session is partially consolidated."""
    session = Session(key="test:consolidated")
    for i in range(10):
        session.messages.append({"role": "user", "content": f"old {i}"})
        session.messages.extend(_tool_turn("cons", i))
    session.last_consolidated = 30

    session.messages.append({"role": "user", "content": "recent"})
    for i in range(15):
        session.messages.extend(_tool_turn("new", i))
    session.messages.append({"role": "user", "content": "latest"})

    history = session.get_history(max_messages=20)
    _assert_no_orphans(history)
    assert all(m.get("role") != "tool" or m["tool_call_id"].startswith("new_") for m in history)


# --- Edge: no tool messages at all ---


def test_no_tool_messages_unchanged():
    session = Session(key="test:plain")
    for i in range(5):
        session.messages.append({"role": "user", "content": f"q{i}"})
        session.messages.append({"role": "assistant", "content": f"a{i}"})

    history = session.get_history(max_messages=6)
    assert len(history) == 6
    _assert_no_orphans(history)


# --- Edge: all leading messages are orphan tool results ---


def test_all_orphan_prefix_stripped():
    """If the window starts with orphan tool results and nothing else, they're all dropped."""
    session = Session(key="test:all-orphan")
    session.messages.append(
        {"role": "tool", "tool_call_id": "gone_1", "name": "x", "content": "ok"}
    )
    session.messages.append(
        {"role": "tool", "tool_call_id": "gone_2", "name": "y", "content": "ok"}
    )
    session.messages.append({"role": "user", "content": "fresh start"})
    session.messages.append({"role": "assistant", "content": "hi"})

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    assert history[0]["role"] == "user"
    assert len(history) == 2


# --- Edge: empty session ---


def test_empty_session_history():
    session = Session(key="test:empty")
    history = session.get_history(max_messages=500)
    assert history == []


# --- Window cuts mid-group: assistant present but some tool results orphaned ---


def test_window_cuts_mid_tool_group():
    """If the window starts between an assistant's tool results, the partial group is trimmed."""
    session = Session(key="test:mid-cut")
    session.messages.append({"role": "user", "content": "setup"})
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "split_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                {"id": "split_b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
            ],
        }
    )
    session.messages.append(
        {"role": "tool", "tool_call_id": "split_a", "name": "x", "content": "ok"}
    )
    session.messages.append(
        {"role": "tool", "tool_call_id": "split_b", "name": "y", "content": "ok"}
    )
    session.messages.append({"role": "user", "content": "next"})
    session.messages.extend(_tool_turn("intact", 0))
    session.messages.append({"role": "assistant", "content": "final"})

    # Window of 6 should cut off the "setup" user msg and the assistant with split_a/split_b,
    # leaving orphan tool results for split_a at the front.
    history = session.get_history(max_messages=6)
    _assert_no_orphans(history)


# --- Back-orphan trim tests (_find_legal_end) ---


def test_trailing_assistant_with_no_tool_results_is_trimmed():
    """An assistant message with tool_calls but no following tool results must be trimmed."""
    session = Session(key="test:back-orphan-basic")
    session.messages.append({"role": "user", "content": "do something"})
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "dangling_1",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
            ],
        }
    )
    # No tool result follows — simulates a crash mid-turn

    history = session.get_history(max_messages=500)
    # The dangling assistant message should be trimmed
    assert all(m.get("role") != "assistant" or not m.get("tool_calls") for m in history), (
        "Dangling assistant tool_calls should have been trimmed from the tail"
    )


def test_trailing_assistant_with_partial_tool_results_is_trimmed():
    """If only some tool results are present (partial), the whole assistant+results block is trimmed."""
    session = Session(key="test:back-orphan-partial")
    session.messages.append({"role": "user", "content": "go"})
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "partial_a",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
                {
                    "id": "partial_b",
                    "type": "function",
                    "function": {"name": "y", "arguments": "{}"},
                },
            ],
        }
    )
    # Only one of two tool results present
    session.messages.append(
        {"role": "tool", "tool_call_id": "partial_a", "name": "x", "content": "ok"}
    )

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    # The incomplete block should be gone
    assert not any(m.get("tool_call_id") == "partial_a" for m in history), (
        "Partial tool result block should have been trimmed"
    )


def test_complete_tool_turn_at_tail_is_preserved():
    """A fully resolved tool turn at the tail must NOT be trimmed."""
    session = Session(key="test:back-clean-tail")
    session.messages.append({"role": "user", "content": "go"})
    session.messages.extend(_tool_turn("complete", 0))
    session.messages.append({"role": "assistant", "content": "all done"})

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    assert len(history) == 5  # user + assistant(tool_calls) + 2 tool results + assistant(text)
    assert history[-1]["content"] == "all done"


def test_back_orphan_trim_preserves_earlier_complete_turns():
    """Trimming the dangling tail must not affect earlier complete tool turns."""
    session = Session(key="test:back-orphan-preserve-earlier")
    session.messages.append({"role": "user", "content": "first"})
    session.messages.extend(_tool_turn("good", 0))
    session.messages.append({"role": "user", "content": "second"})
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "dangling_x",
                    "type": "function",
                    "function": {"name": "z", "arguments": "{}"},
                },
            ],
        }
    )
    # No tool result — dangling tail

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)
    # Earlier complete turn must still be present
    assert any(m.get("tool_call_id") == "good_0_a" for m in history), (
        "Earlier complete tool turn should be preserved"
    )
    # Dangling tail must be gone
    assert not any(
        m.get("role") == "assistant"
        and m.get("tool_calls")
        and any(tc["id"] == "dangling_x" for tc in m["tool_calls"])
        for m in history
    ), "Dangling assistant tail should be trimmed"


def test_find_legal_end_clean_tail_returns_full_length():
    """_find_legal_end returns len(messages) when the tail is clean."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert Session._find_legal_end(messages) == 2


def test_find_legal_end_dangling_returns_trimmed_index():
    """_find_legal_end returns the index of the dangling assistant message."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        # No tool result
    ]
    assert Session._find_legal_end(messages) == 1


def test_retain_recent_legal_suffix_trims_back_orphan():
    """retain_recent_legal_suffix must also trim dangling tool_calls from the back."""
    session = Session(key="test:retain-back-orphan")
    session.messages.append({"role": "user", "content": "go"})
    session.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "dangle_r",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                },
            ],
        }
    )

    session.retain_recent_legal_suffix(10)

    history = session.get_history(max_messages=500)
    _assert_no_orphans(history)


# --- context_prefix_len tag-scan (pinned prefix) ---


def _tag(content: str, legacy: bool = False) -> dict:
    if legacy:
        return {"role": "user", "content": content, "_compact_summary": True}
    return {
        "role": "user",
        "content": content,
        "_session_context": True,
        "_compact_summary": True,
    }


def test_context_prefix_len_no_tags_is_zero():
    msgs = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a1"},
    ]
    assert Session.context_prefix_len(msgs) == 0
    assert Session.context_prefix_len([]) == 0


def test_context_prefix_len_single_tag_then_user():
    msgs = [_tag("ctx"), {"role": "user", "content": "u1"}]
    assert Session.context_prefix_len(msgs) == 1


def test_context_prefix_len_three_tags():
    msgs = [_tag("ctx0"), _tag("ctx1"), _tag("ctx2"), {"role": "user", "content": "u1"}]
    assert Session.context_prefix_len(msgs) == 3


def test_context_prefix_len_legacy_only_tag_counts():
    msgs = [_tag("ctx", legacy=True), _tag("ctx"), {"role": "user", "content": "u1"}]
    assert Session.context_prefix_len(msgs) == 2


def test_context_prefix_len_stops_at_first_untagged():
    # A tag after a non-tagged message counts only the leading run.
    msgs = [
        _tag("ctx0"),
        {"role": "user", "content": "u1"},
        _tag("ctx-orphan"),
    ]
    assert Session.context_prefix_len(msgs) == 1


# --- get_history tag-scan ---


def test_get_history_preserves_leading_dual_tagged_prefix():
    session = Session(key="test:prefix")
    session.messages = [
        _tag("[Previous session context]: old work"),
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    history = session.get_history(max_messages=500)

    assert len(history) == 3
    assert history[0]["content"].startswith("[Previous session context]")
    assert history[1]["role"] == "user"


def test_get_history_drops_leading_untagged_non_user():
    """Leading untagged assistant/tool messages are dropped; output starts at first user msg."""
    session = Session(key="test:lead-drop")
    session.messages = [
        {"role": "assistant", "content": "orphan-a"},
        {"role": "tool", "tool_call_id": "x", "name": "f", "content": "orphan-r"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "a1"},
    ]

    history = session.get_history(max_messages=500)

    assert history[0]["role"] == "user"
    assert history[0]["content"] == "first user"


def test_get_history_max_messages_caps_slicing():
    session = Session(key="test:cap")
    for i in range(10):
        session.add_message("user", f"u{i}")

    history = session.get_history(max_messages=4)

    assert len(history) == 4
    assert [m["content"] for m in history] == ["u6", "u7", "u8", "u9"]


def test_get_history_small_window_slices_prefix_out():
    """The max_messages slice is applied first; a prefix outside the window is not restored."""
    session = Session(key="test:cap-ctx")
    session.messages = [
        _tag("ctx"),
        {"role": "user", "content": "u0"},
        {"role": "user", "content": "u1"},
    ]

    history = session.get_history(max_messages=2)

    # sliced = last 2 = [u0, u1]; no tagged prefix inside the slice
    assert [m["content"] for m in history] == ["u0", "u1"]


# --- D10 load/save migration: last_consolidated -> staged_boundary ---


def _write_session_jsonl(tmp_path: Path, key: str, metadata_extra: dict, n_msgs: int = 5) -> Path:
    mgr = SessionManager(tmp_path)
    path = mgr._get_session_path(key)
    lines = [
        {
            "_type": "metadata",
            "key": key,
            "created_at": "2026-08-01T00:00:00",
            "updated_at": "2026-08-01T00:00:00",
            "metadata": {},
            **metadata_extra,
        }
    ]
    lines += [{"role": "user", "content": f"msg{i}"} for i in range(n_msgs)]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_load_migrates_legacy_last_consolidated(tmp_path):
    _write_session_jsonl(tmp_path, "t:legacy", {"last_consolidated": 5})

    session = SessionManager(tmp_path)._load("t:legacy")

    assert session.staged_boundary == 5
    assert len(session.messages) == 5


def test_load_staged_boundary_wins_over_legacy(tmp_path):
    _write_session_jsonl(tmp_path, "t:both", {"staged_boundary": 2, "last_consolidated": 5})

    session = SessionManager(tmp_path)._load("t:both")

    assert session.staged_boundary == 2


def test_load_no_pointer_keys_defaults_zero(tmp_path):
    _write_session_jsonl(tmp_path, "t:none", {})

    session = SessionManager(tmp_path)._load("t:none")

    assert session.staged_boundary == 0


def test_save_writes_staged_boundary_not_last_consolidated(tmp_path):
    mgr = SessionManager(tmp_path)
    session = Session(key="t:save", staged_boundary=4)
    for i in range(6):
        session.add_message("user", f"msg{i}")

    mgr.save(session)
    first_line = mgr._get_session_path("t:save").read_text(encoding="utf-8").splitlines()[0]
    meta = json.loads(first_line)

    assert meta["staged_boundary"] == 4
    assert "last_consolidated" not in meta
