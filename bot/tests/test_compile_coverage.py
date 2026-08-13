import copy
import json
from pathlib import Path

import pytest
from vikingbot.compile.adapters import AdapterError, CodexRolloutAdapter
from vikingbot.compile.coverage import (
    CollectionSnapshot,
    CoverageError,
    CoverageLedger,
    CoverageStatus,
    ReadReceipt,
    ReviewUnit,
    VerifiedSummary,
)


def _unit(
    source_id: str,
    unit_id: str,
    body: str,
    *,
    kind: str = "document",
    substantive: bool = True,
) -> tuple[ReviewUnit, str]:
    import hashlib

    return (
        ReviewUnit(
            unit_id=unit_id,
            source_id=source_id,
            locator=f"viking://resources/{source_id}/{unit_id}",
            kind=kind,
            substantive=substantive,
            content_sha256=hashlib.sha256(body.encode()).hexdigest(),
        ),
        body,
    )


def _coverage_fixture():
    summary, summary_body = _unit(
        "source-a", "summary-a", "Verified source A summary", kind="summary"
    )
    leaf_a, _ = _unit("source-a", "leaf-a", "A")
    leaf_b, _ = _unit("source-a", "leaf-b", "B")
    source_a = CollectionSnapshot(
        source_id="source-a",
        units=(summary, leaf_a, leaf_b),
        verified_summaries=(
            VerifiedSummary(
                summary_unit_id=summary.unit_id,
                covered_unit_ids=(leaf_a.unit_id, leaf_b.unit_id),
                verifier="test-adapter-v1",
            ),
        ),
    )
    source_b_unit, source_b_body = _unit("source-b", "doc-b", "Source B body")
    source_b = CollectionSnapshot(source_id="source-b", units=(source_b_unit,))
    return source_a, source_b, summary_body, source_b_body


def test_ledger_tracks_receipted_reads_and_verified_summary_coverage():
    source_a, source_b, summary_body, source_b_body = _coverage_fixture()
    ledger = CoverageLedger.from_sources((source_a, source_b))

    assert ledger.summary()["counts"] == {
        "pending": 4,
        "reviewed": 0,
        "covered": 0,
        "skipped": 0,
        "total": 4,
    }
    assert ledger.gate_error()

    ledger.mark_reviewed_by_uri(
        source_b.units[0].locator,
        ReadReceipt.for_content(source_b, "doc-b", source_b_body),
    )
    summary_receipt = ReadReceipt.for_content(source_a, "summary-a", summary_body)
    ledger.mark_reviewed("summary-a", summary_receipt)
    assert ledger.cover(("leaf-a", "leaf-b"), "summary-a") == ("leaf-a", "leaf-b")

    assert ledger.status("summary-a") == CoverageStatus.REVIEWED
    assert ledger.status("leaf-a") == CoverageStatus.COVERED
    assert ledger.status("doc-b") == CoverageStatus.REVIEWED
    assert ledger.summary()["counts"] == {
        "pending": 0,
        "reviewed": 2,
        "covered": 2,
        "skipped": 0,
        "total": 4,
    }
    assert ledger.is_complete()
    assert ledger.gate_error() is None
    assert tuple(unit.unit_id for unit in ledger.units) == (
        "summary-a",
        "leaf-a",
        "leaf-b",
        "doc-b",
    )
    assert ledger.by_id["doc-b"].source_id == "source-b"


def test_covered_requires_a_snapshot_verified_and_read_summary():
    summary, summary_body = _unit("source", "summary", "summary", kind="summary")
    leaf, _ = _unit("source", "leaf", "leaf")
    unverified = CollectionSnapshot(source_id="source", units=(summary, leaf))
    ledger = CoverageLedger((unverified,))

    receipt = ReadReceipt.for_content(unverified, "summary", summary_body)
    ledger.record_read(receipt)
    with pytest.raises(CoverageError, match="not a verified summary"):
        ledger.cover("leaf", "summary")

    verified = CollectionSnapshot(
        source_id="source",
        units=(summary, leaf),
        verified_summaries=(VerifiedSummary("summary", ("leaf",), "fixture"),),
    )
    ledger = CoverageLedger((verified,))
    with pytest.raises(CoverageError, match="must be read"):
        ledger.cover("leaf", "summary")
    with pytest.raises(CoverageError, match="different unit"):
        ledger.cover_from_summary(
            "summary",
            ReadReceipt.for_content(verified, "leaf", "leaf"),
        )


def test_read_receipt_must_be_complete_nonempty_and_match_discovered_content():
    unit, body = _unit("source", "doc", "expected")
    snapshot = CollectionSnapshot(source_id="source", units=(unit,))
    ledger = CoverageLedger((snapshot,))

    with pytest.raises(CoverageError, match="partial reads"):
        ledger.record_read(ReadReceipt.for_content(snapshot, "doc", body, complete=False))
    with pytest.raises(CoverageError, match="content digest"):
        ledger.record_read(ReadReceipt.for_content(snapshot, "doc", "different"))
    assert ledger.status("doc") == CoverageStatus.PENDING


def test_current_iteration_evidence_cannot_unlock_same_iteration_submission():
    unit, body = _unit("source", "doc", "expected")
    snapshot = CollectionSnapshot(source_id="source", units=(unit,))
    ledger = CoverageLedger((snapshot,))

    ledger.record_read(
        ReadReceipt.for_content(snapshot, "doc", body),
        iteration=3,
    )

    assert "not usable until its tool result is delivered" in (
        ledger.gate_error(before_iteration=3) or ""
    )
    assert ledger.gate_error(before_iteration=4) is None


@pytest.mark.parametrize(
    "reason",
    [
        "",
        "skip",
        "not relevant",
        "This item is out of scope",
        "N/A",
        "low value",
        "与当前任务无关",
        "banana",
        "123",
    ],
)
def test_skip_rejects_empty_or_purely_generic_reasons(reason):
    unit, _ = _unit("source", "doc", "body")
    ledger = CoverageLedger.from_sources({"source": (unit,)})

    with pytest.raises(CoverageError, match="skip reason|source-specific"):
        ledger.skip("doc", reason)


def test_all_skipped_source_cannot_pass_gate_but_specific_reason_is_recorded():
    unit, _ = _unit("source", "doc", "body")
    ledger = CoverageLedger.from_sources((unit,))

    ledger.skip("doc", "Generated lockfile duplicates dependency metadata in manifest.toml")

    assert ledger.status("doc") == CoverageStatus.SKIPPED
    assert ledger.summary()["counts"]["skipped"] == 1
    assert "no substantive reviewed or covered unit" in (ledger.gate_error() or "")


def test_full_read_can_correct_an_agent_skip_without_dead_ending_the_gate():
    unit, body = _unit("source", "doc", "body")
    snapshot = CollectionSnapshot("source", (unit,))
    ledger = CoverageLedger((snapshot,))
    ledger.skip("doc", "Generated-looking document requires verification against source.md")

    ledger.record_read(ReadReceipt.for_content(snapshot, "doc", body))

    assert ledger.status("doc") == CoverageStatus.REVIEWED
    assert ledger.is_complete()


def test_concrete_non_english_skip_reason_is_accepted():
    unit, _ = _unit("source", "doc", "body")
    ledger = CoverageLedger.from_sources((unit,))

    ledger.skip("doc", "该文件是构建生成的锁文件，内容已由 manifest.toml 覆盖")

    assert ledger.status("doc") == CoverageStatus.SKIPPED


def test_coverage_internal_serialization_round_trips_and_detects_tampering():
    source_a, source_b, summary_body, source_b_body = _coverage_fixture()
    ledger = CoverageLedger((source_a, source_b))
    ledger.cover_from_summary(
        "summary-a",
        ReadReceipt.for_content(source_a, "summary-a", summary_body),
    )
    ledger.record_read(ReadReceipt.for_content(source_b, "doc-b", source_b_body))

    serialized = ledger.to_internal_dict()
    restored = CoverageLedger.from_internal_dict(json.loads(json.dumps(serialized)))
    assert restored.to_internal_dict() == serialized
    assert restored.summary() == ledger.summary()

    tampered = copy.deepcopy(serialized)
    tampered["snapshots"][0]["verified_summaries"][0]["covered_unit_ids"].append("doc-b")
    with pytest.raises(CoverageError, match="membership digest"):
        CoverageLedger.from_internal_dict(tampered)


THREAD_ID = "019bc371-82cf-7d82-ad0b-96d026aaca73"
ROLLOUT_PATH = Path(
    f"/tmp/.codex/sessions/2026/01/15/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"
)
DREAM_ROLLOUT_PATH = Path(
    f"/tmp/dream-sessions/01/15/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"
)


def _meta(**overrides):
    payload = {
        "id": THREAD_ID,
        "timestamp": "2026-01-15T15:55:54.000Z",
        "cwd": "/workspace",
        "originator": "codex_cli_rs",
        "cli_version": "0.105.0",
        "source": "cli",
    }
    payload.update(overrides)
    return {"timestamp": payload["timestamp"], "type": "session_meta", "payload": payload}


def _jsonl(*records):
    return "\n".join(json.dumps(record) for record in records) + "\n"


def test_codex_adapter_requires_path_filename_and_session_meta_strong_signals():
    adapter = CodexRolloutAdapter()
    raw = _jsonl(_meta())

    assert adapter.detect(ROLLOUT_PATH, raw)
    assert adapter.probe(ROLLOUT_PATH, raw).thread_id == THREAD_ID
    assert not adapter.detect(Path("/tmp/random.jsonl"), raw)
    assert not adapter.detect(
        ROLLOUT_PATH,
        _jsonl({"type": "message", "payload": {"role": "user", "text": "hi"}}),
    )
    mismatched = _meta(id="019bc371-82cf-7d82-ad0b-96d026aacaff")
    assert not adapter.detect(ROLLOUT_PATH, _jsonl(mismatched))
    assert not adapter.detect(
        Path(f"/tmp/.codex/sessions/2026/01/14/rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"),
        raw,
    )


def test_codex_adapter_strictly_recognizes_imported_dream_session_paths():
    adapter = CodexRolloutAdapter()
    raw = _jsonl(_meta())

    assert adapter.detect(DREAM_ROLLOUT_PATH, raw)
    assert adapter.probe(DREAM_ROLLOUT_PATH, raw).thread_id == THREAD_ID
    assert not adapter.detect(
        Path(
            f"/tmp/dream-sessions/01/14/"
            f"rollout-2026-01-15T15-55-54-{THREAD_ID}.jsonl"
        ),
        raw,
    )
    assert not adapter.detect(
        DREAM_ROLLOUT_PATH,
        _jsonl(_meta(id="019bc371-82cf-7d82-ad0b-96d026aacaff")),
    )
    assert not adapter.detect(Path("/tmp/dream-sessions/01/15/history.jsonl"), raw)


def test_codex_adapter_filters_instructions_and_tools_but_keeps_useful_conversation():
    adapter = CodexRolloutAdapter()
    raw = (
        _jsonl(
            _meta(base_instructions={"text": "SECRET SYSTEM PROMPT AND TOOL SCHEMA"}),
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "SYSTEM TOOL SCHEMA"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "DEVELOPER INSTRUCTIONS"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Please fix the parser"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Please fix the parser"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"do not retain"}',
                    "call_id": "1",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Fixed validation and tests."}],
                },
            },
            {
                "type": "inter_agent_communication",
                "payload": {
                    "author": ["root", "reviewer"],
                    "recipient": ["root"],
                    "content": "Reviewer confirmed the regression test passes.",
                    "trigger_turn": False,
                },
            },
            {"type": "future_record", "payload": {"anything": True}},
        )
        + "{malformed-json\n"
    )

    adapted = adapter.adapt(source_id="codex-history", path=ROLLOUT_PATH, raw=raw)
    view = adapted.views[0]
    rendered = view.render()

    assert [message.role for message in view.messages] == ["user", "assistant", "subagent"]
    assert rendered.count("Please fix the parser") == 1
    assert "Fixed validation and tests." in rendered
    assert "Reviewer confirmed" in rendered
    assert "SYSTEM TOOL SCHEMA" not in rendered
    assert "DEVELOPER INSTRUCTIONS" not in rendered
    assert "SECRET SYSTEM PROMPT" not in rendered
    assert "do not retain" not in rendered
    assert any("unknown rollout type" in warning for warning in adapted.warnings)
    assert any("malformed rollout record" in warning for warning in adapted.warnings)
    assert adapted.snapshot.units[0].substantive

    ledger = CoverageLedger((adapted.snapshot,))
    receipt = ReadReceipt.for_content(adapted.snapshot, view.unit_id, rendered)
    ledger.record_read(receipt)
    assert ledger.is_complete()


def test_codex_adapter_handles_paginated_items_and_labels_child_output():
    adapter = CodexRolloutAdapter()
    child_meta = _meta(
        source={"subagent": {"thread_spawn": {"parent_thread_id": "parent", "depth": 1}}},
        parent_thread_id="019bc371-82cf-7d82-ad0b-96d026aacafe",
        agent_role="reviewer",
        history_mode="paginated",
    )
    raw = _jsonl(
        child_meta,
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "UserMessage",
                    "id": "u1",
                    "content": [{"type": "text", "text": "Inspect the tests"}],
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "AgentMessage",
                    "id": "a1",
                    "content": [{"type": "text", "text": "Tests are clean"}],
                },
            },
        },
    )

    adapted = adapter.adapt(source_id="children", path=ROLLOUT_PATH, raw=raw)
    assert [(item.role, item.text) for item in adapted.views[0].messages] == [
        ("user", "Inspect the tests"),
        ("subagent", "Tests are clean"),
    ]
    assert "[SUBAGENT]" in adapter.canonicalize(raw, path=ROLLOUT_PATH)


def test_codex_adapter_keeps_compacted_replacement_history_but_filters_developer_text():
    adapter = CodexRolloutAdapter()
    raw = _jsonl(
        _meta(),
        {
            "type": "compacted",
            "payload": {
                "message": "Useful compacted conclusion",
                "replacement_history": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Original task"}],
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Hidden schema"}],
                    },
                ],
            },
        },
    )

    view = adapter.canonicalize(raw, path=ROLLOUT_PATH)

    assert "Original task" in view
    assert "Useful compacted conclusion" in view
    assert "Hidden schema" not in view


def test_codex_adapter_tolerates_empty_and_unknown_records_without_false_content():
    adapter = CodexRolloutAdapter()
    raw = _jsonl(_meta(), {"type": "future_record", "payload": {"text": "not trusted"}})

    adapted = adapter.adapt(source_id="empty-child", path=ROLLOUT_PATH, raw=raw)
    assert adapted.views[0].messages == ()
    assert adapted.snapshot.units[0].substantive is False
    ledger = CoverageLedger((adapted.snapshot,))
    ledger.skip(
        adapted.snapshot.units[0].unit_id,
        "Metadata-only Codex session contains no user or assistant message records",
    )
    assert ledger.is_complete()

    assert adapter.canonicalize(_jsonl(_meta()), path=ROLLOUT_PATH) == ""
    with pytest.raises(AdapterError, match="unknown or malformed"):
        adapter.canonicalize(raw, path=ROLLOUT_PATH)


@pytest.mark.parametrize(
    "record",
    [
        {"type": "response_item", "payload": {"type": "future_message", "material": "x"}},
        {"type": "response_item", "payload": "material"},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "future_text", "text": "material"}],
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": {"text": "x"}},
        },
        {"type": "event_msg", "payload": {"type": "item_completed", "item": "material"}},
        {"type": "compacted", "payload": {"replacement_history": ["material"]}},
    ],
)
def test_codex_adapter_fails_closed_on_unknown_nested_message_shapes(record):
    adapter = CodexRolloutAdapter()

    with pytest.raises(AdapterError, match="unknown or malformed"):
        adapter.canonicalize(_jsonl(_meta(), record), path=ROLLOUT_PATH)


def test_canonicalize_without_prior_meta_or_with_non_codex_path_fails_closed():
    adapter = CodexRolloutAdapter()
    with pytest.raises(AdapterError, match="session_meta"):
        adapter.canonicalize(_jsonl({"type": "response_item", "payload": {}}))
    with pytest.raises(AdapterError, match="canonical Codex rollout"):
        adapter.canonicalize(_jsonl(_meta()), path="/tmp/history.jsonl")
