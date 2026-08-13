"""Source adapters used to produce small, model-readable compile review views."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vikingbot.compile.coverage import CollectionSnapshot, ReviewUnit

_CODEX_ADAPTER_ID = "codex-rollout-v1"
_CODEX_PATH_RE = re.compile(
    r"(?:^|/)(?:sessions|archived_sessions)/"
    r"(?P<dir_year>\d{4})/(?P<dir_month>\d{2})/(?P<dir_day>\d{2})/"
    r"rollout-(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-"
    r"(?P<thread>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"(?:_(?P<rollout>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}))?"
    r"\.jsonl(?:\.zst)?$",
    re.IGNORECASE,
)
_CODEX_IMPORTED_PATH_RE = re.compile(
    r"(?:^|/)(?P<dir_month>\d{2})/(?P<dir_day>\d{2})/"
    r"rollout-(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-"
    r"(?P<thread>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"(?:_(?P<rollout>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}))?"
    r"\.jsonl(?:\.zst)?$",
    re.IGNORECASE,
)
_KNOWN_ROLLOUT_TYPES = frozenset(
    {
        "compacted",
        "event_msg",
        "inter_agent_communication",
        "inter_agent_communication_metadata",
        "response_item",
        "session_meta",
        "turn_context",
        "world_state",
    }
)
_SAFE_RESPONSE_ITEM_TYPES = frozenset(
    {
        "functioncall",
        "functioncalloutput",
        "local_shell_call",
        "reasoning",
        "websearchcall",
    }
)
_SAFE_EVENT_TYPES = frozenset(
    {
        "agentreasoning",
        "contextcompacted",
        "execapprovalrequest",
        "execcommandbegin",
        "execcommandend",
        "mcpstartupcomplete",
        "taskstarted",
        "tokencount",
        "turnaborted",
        "turncomplete",
    }
)
_SAFE_COMPLETED_ITEM_TYPES = frozenset(
    {"commandexecution", "filechange", "mcptoolcall", "reasoning", "todoupdate", "websearch"}
)


class AdapterError(ValueError):
    """Raised when an input cannot be safely interpreted by an adapter."""


RawRollout = str | bytes | Mapping[str, Any] | Iterable[str | bytes | Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AdapterProbe:
    matched: bool
    adapter_id: str
    reason: str
    thread_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewMessage:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "subagent"}:
            raise AdapterError(f"unsupported review message role {self.role!r}")
        if not self.text.strip():
            raise AdapterError("review message text must be non-empty")


@dataclass(frozen=True, slots=True)
class ReviewView:
    unit_id: str
    session_id: str
    messages: tuple[ReviewMessage, ...]

    def render(self) -> str:
        chunks = [f"# Codex session {self.session_id}"]
        for message in self.messages:
            chunks.append(f"[{message.role.upper()}]\n{message.text}")
        return "\n\n".join(chunks).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class AdaptedCollection:
    adapter_id: str
    snapshot: CollectionSnapshot
    views: tuple[ReviewView, ...]
    warnings: tuple[str, ...] = ()

    def view_by_unit_id(self, unit_id: str) -> ReviewView:
        for view in self.views:
            if view.unit_id == unit_id:
                return view
        raise AdapterError(f"unknown adapted unit {unit_id!r}")


@dataclass(frozen=True, slots=True)
class _ParsedLine:
    line_number: int
    value: Mapping[str, Any] | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _PathIdentity:
    thread_id: str


class CodexRolloutAdapter:
    """Strict adapter for Codex's local rollout JSONL representation.

    Detection combines the canonical dated path and filename with the first
    ``session_meta`` record.  A generic JSONL file never matches on extension
    alone.  Callers own I/O and may pass decoded records directly.
    """

    adapter_id = _CODEX_ADAPTER_ID

    def is_candidate_path(self, path: str | Path) -> bool:
        """Cheap path-only prefilter; ``probe`` remains the strict recognizer."""

        return _codex_path_identity(path) is not None

    def detect(self, path: str | Path, raw: RawRollout) -> bool:
        return self.probe(path, raw).matched

    def probe(self, path: str | Path, raw: RawRollout) -> AdapterProbe:
        parsed = _parse_raw(raw)
        return self._probe_parsed(path, parsed)

    def canonicalize(self, raw: RawRollout, *, path: str | Path | None = None) -> str:
        """Return only user, substantive assistant and plaintext sub-agent text.

        When ``path`` is supplied the full strict detector is applied.  Without
        it, canonicalization still requires a valid leading Codex session-meta
        record; this form is intended for a caller that already ran ``probe``.
        """

        parsed = _parse_raw(raw)
        if path is not None:
            probe = self._probe_parsed(path, parsed)
            if not probe.matched:
                raise AdapterError(probe.reason)
            thread_id = probe.thread_id or "unknown"
            first = _first_nonempty(parsed)
            assert first is not None and first.value is not None
            meta = first.value["payload"]
        else:
            first = _first_nonempty(parsed)
            meta = _session_meta(first.value if first else None, expected_thread_id=None)
            if meta is None:
                raise AdapterError("raw input does not start with valid Codex session_meta")
            thread_id = str(meta["id"])
        messages, warnings = _extract_messages(parsed, meta)
        if warnings:
            raise AdapterError(
                "Codex rollout contains unknown or malformed records that cannot be safely "
                "omitted from its review view"
            )
        if not messages:
            return ""
        view = ReviewView(
            unit_id=f"codex::{thread_id}",
            session_id=thread_id,
            messages=messages,
        )
        return view.render()

    def adapt(
        self,
        *,
        source_id: str,
        path: str | Path,
        raw: RawRollout,
    ) -> AdaptedCollection:
        parsed = _parse_raw(raw)
        probe = self._probe_parsed(path, parsed)
        if not probe.matched:
            raise AdapterError(probe.reason)
        first = _first_nonempty(parsed)
        assert first is not None and first.value is not None
        meta = first.value["payload"]
        thread_id = probe.thread_id or str(meta["id"])
        source_id = str(source_id).strip()
        if not source_id:
            raise AdapterError("source_id must be non-empty")

        messages, warnings = _extract_messages(parsed, meta)
        unit_id = f"{source_id}::codex::{thread_id}"
        view = ReviewView(unit_id=unit_id, session_id=thread_id, messages=messages)
        canonical = view.render().encode("utf-8")
        unit = ReviewUnit(
            unit_id=unit_id,
            source_id=source_id,
            locator=str(path),
            kind="codex_session",
            substantive=bool(messages),
            content_sha256=hashlib.sha256(canonical).hexdigest(),
        )
        return AdaptedCollection(
            adapter_id=self.adapter_id,
            snapshot=CollectionSnapshot(source_id=source_id, units=(unit,)),
            views=(view,),
            warnings=warnings,
        )

    def _probe_parsed(
        self,
        path: str | Path,
        parsed: tuple[_ParsedLine, ...],
    ) -> AdapterProbe:
        identity = _codex_path_identity(path)
        if identity is None:
            return AdapterProbe(False, self.adapter_id, "path is not a canonical Codex rollout")
        first = _first_nonempty(parsed)
        if first is None:
            return AdapterProbe(False, self.adapter_id, "rollout is empty")
        if first.error or first.value is None:
            return AdapterProbe(
                False,
                self.adapter_id,
                "first non-empty rollout record is not valid JSON",
            )
        meta = _session_meta(first.value, expected_thread_id=identity.thread_id)
        if meta is None:
            return AdapterProbe(
                False,
                self.adapter_id,
                "first rollout record is not matching Codex session_meta",
            )
        return AdapterProbe(True, self.adapter_id, "matched", identity.thread_id)


def _codex_path_identity(path: str | Path) -> _PathIdentity | None:
    normalized = str(path).replace("\\", "/")
    match = _CODEX_PATH_RE.search(normalized)
    imported = False
    if match is None:
        match = _CODEX_IMPORTED_PATH_RE.search(normalized)
        imported = match is not None
    if match is None:
        return None
    stamp = match.group("stamp")
    try:
        datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%S")
        thread_id = str(uuid.UUID(match.group("thread")))
        if match.group("rollout"):
            uuid.UUID(match.group("rollout"))
    except ValueError:
        return None
    expected_date = (
        f"{stamp[:4]}-{match.group('dir_month')}-{match.group('dir_day')}"
        if imported
        else "-".join((match.group("dir_year"), match.group("dir_month"), match.group("dir_day")))
    )
    if stamp[:10] != expected_date:
        return None
    return _PathIdentity(thread_id=thread_id)


def _session_meta(
    record: Mapping[str, Any] | None,
    *,
    expected_thread_id: str | None,
) -> Mapping[str, Any] | None:
    if not isinstance(record, Mapping) or record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return None
    try:
        thread_id = str(uuid.UUID(str(payload["id"])))
    except (KeyError, ValueError, AttributeError):
        return None
    if expected_thread_id is not None and thread_id != expected_thread_id:
        return None
    for field in ("timestamp", "cwd", "originator", "cli_version"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            return None
    source = payload.get("source")
    if not (
        isinstance(source, str) and source.strip() or isinstance(source, Mapping) and bool(source)
    ):
        return None
    return payload


def _parse_raw(raw: RawRollout) -> tuple[_ParsedLine, ...]:
    if isinstance(raw, Mapping):
        values: Iterable[str | bytes | Mapping[str, Any]] = (raw,)
    elif isinstance(raw, bytes):
        try:
            values = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return (_ParsedLine(1, None, "invalid UTF-8"),)
    elif isinstance(raw, str):
        values = raw.splitlines()
    else:
        values = raw

    parsed: list[_ParsedLine] = []
    for line_number, value in enumerate(values, start=1):
        if isinstance(value, Mapping):
            parsed.append(_ParsedLine(line_number, value))
            continue
        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError:
                parsed.append(_ParsedLine(line_number, None, "invalid UTF-8"))
                continue
        else:
            text = str(value)
        if not text.strip():
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            parsed.append(_ParsedLine(line_number, None, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(decoded, Mapping):
            parsed.append(_ParsedLine(line_number, None, "JSON record is not an object"))
            continue
        parsed.append(_ParsedLine(line_number, decoded))
    return tuple(parsed)


def _first_nonempty(parsed: tuple[_ParsedLine, ...]) -> _ParsedLine | None:
    return parsed[0] if parsed else None


def _extract_messages(
    parsed: tuple[_ParsedLine, ...],
    meta: Mapping[str, Any],
) -> tuple[tuple[ReviewMessage, ...], tuple[str, ...]]:
    messages: list[ReviewMessage] = []
    warnings: list[str] = []
    child_session = _is_subagent_session(meta)
    for parsed_line in parsed[1:]:
        if parsed_line.error or parsed_line.value is None:
            warnings.append(f"ignored malformed rollout record at line {parsed_line.line_number}")
            continue
        record = parsed_line.value
        record_type = record.get("type")
        if not isinstance(record_type, str):
            warnings.append(f"ignored untyped rollout record at line {parsed_line.line_number}")
            continue
        if record_type not in _KNOWN_ROLLOUT_TYPES:
            warnings.append(
                f"ignored unknown rollout type {record_type!r} at line {parsed_line.line_number}"
            )
            continue
        try:
            extracted = _messages_from_record(record, child_session=child_session)
        except AdapterError as exc:
            warnings.append(f"unsafe rollout record at line {parsed_line.line_number}: {exc}")
            continue
        for message in extracted:
            if messages and (messages[-1].role, messages[-1].text) == (
                message.role,
                message.text,
            ):
                continue
            messages.append(message)
    return tuple(messages), tuple(warnings)


def _messages_from_record(
    record: Mapping[str, Any],
    *,
    child_session: bool,
) -> tuple[ReviewMessage, ...]:
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise AdapterError(f"{record_type!r} payload is not an object")
    if record_type == "response_item":
        return _messages_from_response_item(payload, child_session=child_session)
    if record_type == "event_msg":
        return _messages_from_event(payload, child_session=child_session)
    if record_type == "compacted":
        messages: list[ReviewMessage] = []
        replacement_history = payload.get("replacement_history")
        if replacement_history is not None and not isinstance(replacement_history, list):
            raise AdapterError("compacted replacement_history is not a list")
        if isinstance(replacement_history, list):
            for item in replacement_history:
                if not isinstance(item, Mapping):
                    raise AdapterError("compacted replacement item is not an object")
                messages.extend(
                    _messages_from_response_item(item, child_session=child_session)
                )
        summary = _plain_text(payload.get("message"))
        if summary:
            messages.append(
                ReviewMessage(
                    role="subagent" if child_session else "assistant",
                    text=summary,
                )
            )
        return tuple(messages)
    if record_type == "inter_agent_communication":
        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            raise AdapterError("inter-agent content is not plaintext")
        text = _plain_text(content)
        if text:
            return (ReviewMessage(role="subagent", text=text),)
    return ()


def _messages_from_response_item(
    payload: Mapping[str, Any],
    *,
    child_session: bool,
) -> tuple[ReviewMessage, ...]:
    item_type = _normalized_tag(payload.get("type"))
    if item_type == "message":
        role = _normalized_tag(payload.get("role"))
        if role in {"system", "developer"}:
            # This drops system/developer prompts and their embedded tool schemas.
            return ()
        if role not in {"user", "assistant"}:
            raise AdapterError(f"unknown response message role {role!r}")
        text = _content_text(payload.get("content"))
        if not text:
            return ()
        output_role = "subagent" if role == "assistant" and child_session else role
        return (ReviewMessage(role=output_role, text=text),)
    if item_type == "agentmessage":
        text = _content_text(payload.get("content"))
        if text:
            return (ReviewMessage(role="subagent", text=text),)
    if item_type in _SAFE_RESPONSE_ITEM_TYPES:
        return ()
    raise AdapterError(f"unknown response item type {item_type!r}")


def _messages_from_event(
    payload: Mapping[str, Any],
    *,
    child_session: bool,
) -> tuple[ReviewMessage, ...]:
    event_type = _normalized_tag(payload.get("type"))
    if event_type == "usermessage":
        text = _plain_text(payload.get("message")) or _content_text(payload.get("content"))
        return (ReviewMessage("user", text),) if text else ()
    if event_type == "agentmessage":
        text = _plain_text(payload.get("message")) or _content_text(payload.get("content"))
        if not text:
            return ()
        role = "subagent" if child_session else "assistant"
        return (ReviewMessage(role, text),)
    if event_type == "itemcompleted":
        item = payload.get("item")
        if not isinstance(item, Mapping):
            raise AdapterError("completed event item is not an object")
        item_type = _normalized_tag(item.get("type"))
        if item_type == "usermessage":
            text = _content_text(item.get("content"))
            return (ReviewMessage("user", text),) if text else ()
        if item_type == "agentmessage":
            text = _content_text(item.get("content"))
            if text:
                role = "subagent" if child_session else "assistant"
                return (ReviewMessage(role, text),)
        if item_type == "subagentactivity":
            text = _plain_text(item.get("message")) or _content_text(item.get("content"))
            return (ReviewMessage("subagent", text),) if text else ()
        if item_type in _SAFE_COMPLETED_ITEM_TYPES:
            return ()
        raise AdapterError(f"unknown completed item type {item_type!r}")
    if event_type == "subagentactivity":
        text = _plain_text(payload.get("message")) or _content_text(payload.get("content"))
        return (ReviewMessage("subagent", text),) if text else ()
    if event_type in _SAFE_EVENT_TYPES:
        return ()
    raise AdapterError(f"unknown event type {event_type!r}")


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _plain_text(content)
    if not isinstance(content, list):
        raise AdapterError("message content is neither text nor a block list")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise AdapterError("message content contains a non-object block")
        block_type = _normalized_tag(item.get("type"))
        if block_type not in {"inputtext", "outputtext", "text"}:
            raise AdapterError(f"unknown message content block type {block_type!r}")
        text = _plain_text(item.get("text"))
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else None


def _plain_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalized_tag(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _is_subagent_session(meta: Mapping[str, Any]) -> bool:
    if any(meta.get(field) for field in ("parent_thread_id", "agent_path", "agent_role")):
        return True
    source = meta.get("source")
    if isinstance(source, str):
        return "subagent" in _normalized_tag(source)
    if isinstance(source, Mapping):
        return any("subagent" in _normalized_tag(key) for key in source)
    return False


__all__ = [
    "AdaptedCollection",
    "AdapterError",
    "AdapterProbe",
    "CodexRolloutAdapter",
    "ReviewMessage",
    "ReviewView",
]
