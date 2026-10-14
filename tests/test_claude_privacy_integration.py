"""Local-only privacy and non-mutation checks for truncation/backfill integration."""
from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

from ai_session_export.sources.claude_code import export_claude_code


def _fixtures(root: Path) -> tuple[Path, Path, Path]:
    project = root / "projects" / "fixture-project"
    project.mkdir(parents=True)
    session = project / "fixture-main.jsonl"
    rows = [
        {"type": "user", "uuid": "fixture-user", "parentUuid": "missing-parent",
         "timestamp": "2026-10-13T12:00:00Z", "sessionId": "fixture-main",
         "message": {"role": "user", "content": "Local fixture question"}},
        {"type": "assistant", "uuid": "fixture-answer", "parentUuid": "fixture-user",
         "timestamp": "2026-10-13T12:01:00Z", "sessionId": "fixture-main",
         "message": {"role": "assistant", "content": "Local fixture answer", "model": "fixture-model"}},
    ]
    session.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    sibling = project / "fixture-main" / "subagents" / "fixture-agent.jsonl"
    sibling.parent.mkdir(parents=True)
    sibling.write_text(json.dumps({"timestamp": "2026-10-12T12:00:00Z",
        "type": "user", "message": {"content": "SYNTHETIC_SUBAGENT_PRIVATE_MARKER"}}) + "\n")
    backfill = root / "backfill" / "fixture-main.jsonl"
    backfill.parent.mkdir()
    backfill.write_text(json.dumps({"created_at": "2026-10-12T13:00:00Z",
        "role": "system", "text": "SYNTHETIC_SYSTEM_PRIVATE_MARKER"}) + "\n")
    return session, sibling, backfill


def _export(root: Path, state: dict, *, dry_run: bool):
    return export_claude_code(root / "archive", state, full=False, dry_run=dry_run,
        since_date=None, project_dirs=(root / "projects",),
        history_files=(root / "absent-history.jsonl",), backfill_dir=root / "backfill")


def test_diagnostics_never_import_subagent_body_or_network(tmp_path: Path, monkeypatch) -> None:
    sources = _fixtures(tmp_path)
    original = {p: p.read_bytes() for p in sources}

    def deny_network(*args, **kwargs):
        raise AssertionError("Local export must never open a network socket")

    monkeypatch.setattr(socket, "socket", deny_network)
    result = _export(tmp_path, {}, dry_run=False)
    assert result["exported"] == 1
    output = next((tmp_path / "archive").glob("*.md")).read_text()
    diagnostics = json.dumps(result)
    assert "subagent_activity_before_transcript" in output
    assert "2026-10-12T12:00:00Z" in output
    assert "PRIVATE_MARKER" not in output + diagnostics
    assert {p: p.read_bytes() for p in sources} == original


def test_backfill_dry_run_preserves_existing_archive_and_state(tmp_path: Path) -> None:
    sources = _fixtures(tmp_path)
    state = {}
    assert _export(tmp_path, state, dry_run=False)["exported"] == 1
    archive = next((tmp_path / "archive").glob("*.md"))
    original_archive = archive.read_bytes()
    original_state = copy.deepcopy(state)
    sources[-1].write_text(json.dumps({"created_at": "2026-10-12T13:00:00Z",
        "role": "user", "text": "Earlier synthetic question"}) + "\n")
    original_sources = {p: p.read_bytes() for p in sources}
    assert _export(tmp_path, state, dry_run=True)["exported"] == 1
    assert state == original_state
    assert archive.read_bytes() == original_archive
    assert list((tmp_path / "archive").glob("*.md")) == [archive]
    assert {p: p.read_bytes() for p in sources} == original_sources
