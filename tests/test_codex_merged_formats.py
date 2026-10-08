"""Synthetic integration regressions for the Jerry guards + completed-item parser."""
import json
from pathlib import Path

from ai_session_export.sources.codex import parse_codex_session_file


def _event(kind: str, payload: dict) -> dict:
    return {"timestamp": "2026-10-08T10:00:00Z", "type": kind, "payload": payload}


def _session(path: Path, turns: list[dict], **metadata) -> Path:
    events = [
        _event("session_meta", {"id": "synthetic-merged-format", "cwd": "/fixture", **metadata}),
        _event("turn_context", {"model": "fixture-model"}),
        *turns,
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return path


def _response(role: str, text: str) -> dict:
    return _event("response_item", {"type": "message", "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}]})


def _completed(role: str, text: str, extra: list[dict] | None = None) -> dict:
    return _event("event_msg", {"type": "item_completed", "item": {
        "type": "UserMessage" if role == "user" else "AgentMessage",
        "content": [*(extra or []), {"type": "text" if role == "user" else "Text", "text": text}],
    }})


def test_completed_items_win_over_response_item_duplicates_without_repetition_loss(tmp_path: Path) -> None:
    # Repeated genuine turns must survive; the mirrored response encoding must not double them.
    turns = []
    for _ in range(2):
        turns.extend([_response("user", "Repeat the fixture"),
            _completed("user", "Repeat the fixture"),
            _response("assistant", "Repeated fixture reply"),
            _completed("assistant", "Repeated fixture reply")])
    turns.append(_event("event_msg", {"type": "token_count", "info": {}}))
    parsed = parse_codex_session_file(_session(tmp_path / "rollout.jsonl", turns), {})
    assert parsed is not None
    assert [(message.role, message.content) for message in parsed.record.messages] == [
        ("user", "Repeat the fixture"), ("assistant", "Repeated fixture reply"),
        ("user", "Repeat the fixture"), ("assistant", "Repeated fixture reply"),
    ]


def test_completed_user_context_guard_retains_user_quoted_prefixes(tmp_path: Path) -> None:
    # Filter only distinct known injected content items, using the existing Fork guard.
    # Text quoting/mentioning a wrapper inside genuine speech remains intact.
    text = 'Explain "<environment_context>" and keep my quoted text.'
    turns = [_completed("user", text, [
        {"type": "text", "text": "<environment_context>\n<cwd>/fixture</cwd>\n</environment_context>"},
        {"type": "text", "text": "# AGENTS.md instructions for /fixture\nfixture rules"},
    ]), _completed("assistant", "Fixture explanation")]
    parsed = parse_codex_session_file(_session(tmp_path / "rollout.jsonl", turns), {})
    assert parsed is not None
    assert [message.content for message in parsed.record.messages] == [text, "Fixture explanation"]
    assert parsed.record.title == text


def test_completed_items_keep_internal_thread_guard_and_allow_automation(tmp_path: Path) -> None:
    turns = [_completed("user", "Fixture question"), _completed("assistant", "Fixture answer")]
    for source in ["guardian_review", "subagent"]:
        assert parse_codex_session_file(_session(tmp_path / "rollout.jsonl", turns, thread_source=source), {}) is None
    assert parse_codex_session_file(_session(tmp_path / "rollout.jsonl", turns,
        source={"subagent": {"other": "fixture"}}), {}) is None
    assert parse_codex_session_file(_session(tmp_path / "rollout.jsonl", turns, thread_source="automation"), {}) is not None
