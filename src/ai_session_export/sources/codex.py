from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from ..markdown import render_markdown
from ..models import MessageTurn, SessionRecord
from ..utils import parse_iso_timestamp, should_skip_session, unique_output_path


DEFAULT_CODEX_SESSION_DIRS = (
    Path.home() / ".codex" / "sessions",
    Path.home() / ".codex" / "archived_sessions",
)
DEFAULT_CODEX_SESSION_INDEX = Path.home() / ".codex" / "session_index.jsonl"
SESSION_ID_PATTERN = re.compile(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$")
ROLLOUT_DATE_PATTERN = re.compile(r"^rollout-(\d{4}-\d{2}-\d{2})T")

# Newer Codex rollouts (observed 2026-09) no longer emit `event_msg` user/agent
# messages; the transcript lives in `response_item` records with
# `payload.type == "message"` and a `role`. User turns there carry injected
# workspace context as separate content items, which are not user speech.
INJECTED_USER_CONTENT_PREFIXES = (
    "<environment_context>",
    "<recommended_plugins>",
    "<user_instructions>",
    "<permissions",
    "<turn_aborted>",
    "<system-reminder>",
    "<image",
    "# AGENTS.md instructions",
)
# Guardian auto-review and spawned subagent threads are Codex-internal, not
# Jerry's conversations. `thread_source` is the stable marker; a dict-valued
# `source` is the older encoding of the same thing.
SKIPPED_THREAD_SOURCES = {"subagent", "guardian_review"}


class ParsedCodexSession(NamedTuple):
    record: SessionRecord
    latest_timestamp_ms: int


def _iter_session_files(session_dirs: tuple[Path, ...]) -> list[Path]:
    files: set[Path] = set()
    for root in session_dirs:
        if root.is_dir():
            files.update(root.rglob("rollout-*.jsonl"))
    return sorted(files)


def _item_content_text(item: dict, *, drop_injected: bool = False) -> str:
    """Join the text parts of an item_completed message.

    Content entries are `{"type": "text"|"Text", "text": ...}`; the casing differs
    between user and agent items, so match on the payload key instead of the type.
    """
    parts = []
    for entry in item.get("content") or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if text and not (drop_injected and text.startswith(INJECTED_USER_CONTENT_PREFIXES)):
            parts.append(text)
    return "\n".join(parts).strip()


def _load_session_titles(index_file: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    if not index_file.is_file():
        return titles

    with index_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            session_id = str(entry.get("id") or "").strip()
            title = str(entry.get("thread_name") or "").strip()
            if session_id and title:
                titles[session_id] = title
    return titles


def _session_id_from_path(file_path: Path) -> str:
    match = SESSION_ID_PATTERN.search(file_path.stem)
    return match.group(1) if match else ""


def _date_from_path(file_path: Path) -> date | None:
    match = ROLLOUT_DATE_PATTERN.match(file_path.name)
    return date.fromisoformat(match.group(1)) if match else None


def _is_internal_thread(meta: dict[str, Any]) -> bool:
    if isinstance(meta.get("source"), dict):
        return True
    return str(meta.get("thread_source") or "").strip() in SKIPPED_THREAD_SOURCES


def _message_item_text(payload: dict[str, Any], *, drop_injected: bool) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("input_text", "output_text", "text"):
                continue
            parts.append(str(item.get("text") or ""))
    else:
        return ""
    kept: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if drop_injected and text.startswith(INJECTED_USER_CONTENT_PREFIXES):
            continue
        kept.append(text)
    return "\n\n".join(kept).strip()


def parse_codex_session_file(file_path: Path, titles: dict[str, str]) -> ParsedCodexSession | None:
    session_id = ""
    cwd = ""
    models: set[str] = set()
    messages: list[MessageTurn] = []
    current_model: str | None = None
    pending_user_turns: list[int] = []
    message_origins: list[str] = []
    first_user_text = ""
    started_at: date | None = None
    latest_timestamp_ms = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                # Active Codex rollouts may have an incomplete final line.
                continue

            timestamp = parse_iso_timestamp(str(event.get("timestamp") or ""))
            event_ts_ms = None
            if timestamp is not None:
                event_ts_ms = int(timestamp.timestamp() * 1000)
                latest_timestamp_ms = max(latest_timestamp_ms, event_ts_ms)
                if started_at is None:
                    started_at = timestamp.date()

            event_type = event.get("type")
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue

            if event_type == "session_meta":
                if _is_internal_thread(payload):
                    return None
                session_id = str(payload.get("id") or payload.get("session_id") or session_id)
                cwd = str(payload.get("cwd") or cwd)
                continue

            if event_type == "turn_context":
                model = str(payload.get("model") or "").strip()
                if model:
                    current_model = model
                    models.add(model)
                    for index in pending_user_turns:
                        messages[index] = messages[index]._replace(model=model)
                    pending_user_turns.clear()
                cwd = str(payload.get("cwd") or cwd)
                continue

            payload_type = payload.get("type")
            if event_type == "response_item" and payload_type == "message":
                role = str(payload.get("role") or "")
                if role == "user":
                    text = _message_item_text(payload, drop_injected=True)
                    if not text:
                        continue
                    messages.append(
                        MessageTurn(role="user", content=text, time_created=event_ts_ms, model=current_model)
                    )
                    message_origins.append("item")
                    pending_user_turns.append(len(messages) - 1)
                elif role == "assistant":
                    text = _message_item_text(payload, drop_injected=False)
                    if text:
                        messages.append(
                            MessageTurn(role="assistant", content=text, time_created=event_ts_ms, model=current_model)
                        )
                        message_origins.append("item")
                    pending_user_turns.clear()
                continue

            if event_type != "event_msg":
                continue

            role: str | None = None
            text = ""

            if payload_type == "user_message":
                role, text = "user", str(payload.get("message") or "").strip()
            elif payload_type == "agent_message":
                role, text = "assistant", str(payload.get("message") or "").strip()
            elif payload_type == "item_completed":
                # Current rollout format: turns arrive as completed items rather than
                # user_message/agent_message events. Anything other than the two
                # message kinds (Reasoning, command output, ...) stays private.
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "UserMessage":
                    role = "user"
                elif item_type == "AgentMessage":
                    role = "assistant"
                else:
                    continue
                text = _item_content_text(item, drop_injected=role == "user")

            if role is None:
                continue

            if role == "user":
                if not text:
                    continue
                messages.append(
                    MessageTurn(role="user", content=text, time_created=event_ts_ms, model=current_model)
                )
                message_origins.append("legacy")
                pending_user_turns.append(len(messages) - 1)
            else:
                if text:
                    messages.append(
                        MessageTurn(role="assistant", content=text, time_created=event_ts_ms, model=current_model)
                    )
                    message_origins.append("legacy")
                pending_user_turns.clear()

    # A rollout that carries event_msg message turns (legacy messages or current
    # completed items) is authoritative. Only fall back to response_item turns
    # when no event_msg message turns exist, so a file with both encodings is
    # not exported twice over.
    if "legacy" in message_origins:
        messages = [m for m, origin in zip(messages, message_origins) if origin == "legacy"]
    first_user_text = next((m.content for m in messages if m.role == "user"), "")

    if not session_id or not first_user_text or not messages or started_at is None:
        return None

    title = titles.get(session_id) or first_user_text.splitlines()[0][:120]
    if should_skip_session(title):
        return None

    return ParsedCodexSession(
        record=SessionRecord(
            source="codex",
            session_id=session_id,
            title=title,
            date=started_at.isoformat(),
            messages=messages,
            project_directory=cwd,
            models_used=sorted(models),
        ),
        latest_timestamp_ms=latest_timestamp_ms,
    )


def export_codex(
    output_dir: Path,
    state: dict[str, Any],
    *,
    full: bool,
    dry_run: bool,
    since_date: date | None,
    session_dirs: tuple[Path, ...] = DEFAULT_CODEX_SESSION_DIRS,
    session_index: Path = DEFAULT_CODEX_SESSION_INDEX,
) -> dict[str, Any]:
    session_files = _iter_session_files(session_dirs)
    titles = _load_session_titles(session_index)
    source_state = state.setdefault("codex", {})
    session_state: dict[str, dict[str, Any]] = source_state.setdefault("sessions", {})

    exported = 0
    scanned = 0
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in session_files:
        scanned += 1
        path_date = _date_from_path(file_path)
        if since_date and path_date and path_date < since_date:
            continue
        path_session_id = _session_id_from_path(file_path)
        path_state = session_state.get(path_session_id, {})
        path_output = str(path_state.get("output_file") or "")
        path_output_exists = bool(path_output) and (output_dir / path_output).is_file()
        file_mtime_ns = file_path.stat().st_mtime_ns
        if (
            not full
            and path_session_id
            and int(path_state.get("source_mtime_ns", 0)) == file_mtime_ns
            and path_output_exists
        ):
            continue

        parsed = parse_codex_session_file(file_path, titles)
        if parsed is None:
            continue
        if since_date and date.fromisoformat(parsed.record.date) < since_date:
            continue

        previous = session_state.get(parsed.record.session_id, {})
        previous_timestamp = int(previous.get("latest_timestamp", 0))
        previous_output = str(previous.get("output_file") or "")
        output_path = output_dir / previous_output if previous_output else None
        output_exists = output_path is not None and output_path.is_file()
        if not full and parsed.latest_timestamp_ms <= previous_timestamp and output_exists:
            if not dry_run:
                previous["source_mtime_ns"] = file_mtime_ns
                session_state[parsed.record.session_id] = previous
            continue

        if output_path is None:
            output_path = unique_output_path(output_dir, parsed.record.date, parsed.record.title)
        if not dry_run:
            output_path.write_text(render_markdown(parsed.record), encoding="utf-8")
            session_state[parsed.record.session_id] = {
                "latest_timestamp": parsed.latest_timestamp_ms,
                "output_file": output_path.name,
                "source_mtime_ns": file_mtime_ns,
            }
        exported += 1

    return {"source": "codex", "scanned": scanned, "exported": exported}
