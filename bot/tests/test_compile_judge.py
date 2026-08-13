import asyncio
import json
from types import SimpleNamespace

import pytest
from vikingbot.agent.tools.base import ToolContext
from vikingbot.agent.tools.compile import RequestCompileExtensionTool, SubmitWikiBundleTool
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.compile.coverage import CoverageLedger
from vikingbot.compile.judge import CompileJudge
from vikingbot.compile.models import (
    CompileLimits,
    SanitizedCompileRequest,
    WikiBundleDraft,
)
from vikingbot.compile.service import BotCompileService, CompileCapabilities
from vikingbot.providers.base import LLMResponse


class _Provider:
    def __init__(self, responses=(), *, timeout=False):
        self.responses = list(responses)
        self.timeout = timeout
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.timeout:
            await asyncio.Event().wait()
        return LLMResponse(content=self.responses.pop(0))


def _request() -> SanitizedCompileRequest:
    return SanitizedCompileRequest.model_validate(
        {
            "from": ["viking://resources/source"],
            "to": "viking://resources/wiki",
            "skill": "viking://agent/skills/wiki",
            "reason": "Compile faithfully",
        }
    )


def _empty_bundle() -> WikiBundleDraft:
    return WikiBundleDraft.model_validate({"pages": [], "files": [], "links": []})


def _registry(quality_judge, *, limits=None, max_iterations=40):
    service = object.__new__(BotCompileService)
    service.limits = limits or CompileLimits()
    request_loop = SimpleNamespace(
        tools=ToolRegistry(), config=None, max_iterations=max_iterations
    )
    registry, _ = service._build_compile_registry(
        request_loop,
        roots=("viking://resources/source", "viking://resources/wiki"),
        target_uri="viking://resources/wiki",
        source_ids=set(),
        catalog_uris=set(),
        capabilities=CompileCapabilities(exec_enabled=False),
        coverage=CoverageLedger(()),
        quality_judge=quality_judge,
        request=_request(),
        skill_contract="Write a faithful Wiki.",
    )
    return request_loop, registry


@pytest.mark.asyncio
async def test_compile_judge_uses_a_fresh_two_message_context_without_tools():
    provider = _Provider(
        [
            json.dumps({"verdict": "pass", "rationale": "sound"}),
            json.dumps({"verdict": "approve", "rationale": "specific work remains"}),
        ]
    )
    judge = CompileJudge(provider, model="judge-model")

    await judge.judge_quality(
        reason="Compile",
        skill_contract="Write Wiki",
        coverage={"complete": True},
        bundle=_empty_bundle(),
    )
    await judge.judge_extension({"remaining_work": "Read the final source"})

    assert len(provider.calls) == 2
    for call in provider.calls:
        assert [message["role"] for message in call["messages"]] == ["system", "user"]
        assert len(call["messages"]) == 2
        assert call["tools"] == []
        assert call["model"] == "judge-model"


@pytest.mark.asyncio
async def test_judge_budget_preserves_structured_candidate_and_extension_actions():
    provider = _Provider(
        [
            json.dumps({"verdict": "pass", "rationale": "sound"}),
            json.dumps({"verdict": "approve", "rationale": "needed"}),
        ]
    )
    judge = CompileJudge(provider, model="judge-model", input_chars=12_000)
    hostile = ('"\\\n' * 20_000) + "tail"

    await judge.judge_quality(
        reason=hostile,
        skill_contract=hostile,
        coverage={"complete": True, "issues": [hostile]},
        bundle=_empty_bundle(),
        evidence=[{"excerpt": hostile}],
    )
    await judge.judge_extension(
        {
            "iteration": 38,
            "iteration_limit": 40,
            "coverage": {"counts": {"pending": 1000}, "issues": [hostile] * 100},
            "reason": "long task",
            "completed_work": "reviewed 900 units",
            "remaining_work": "review 100 units",
            "next_actions": "read and submit",
        }
    )

    quality_packet = json.loads(provider.calls[0]["messages"][1]["content"])
    preserved = quality_packet.get("packet", quality_packet)
    assert "candidate_bundle" in preserved
    assert "artifact_previews" in preserved
    assert len(provider.calls[0]["messages"][1]["content"]) <= 12_000
    extension_packet = json.loads(provider.calls[1]["messages"][1]["content"])
    assert extension_packet["remaining_work"] == "review 100 units"
    assert extension_packet["next_actions"] == "read and submit"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_verdict", "expected_feedback"),
    [
        ({"verdict": "pass", "rationale": "faithful"}, "pass", "faithful"),
        (
            {
                "verdict": "pass_with_warning",
                "rationale": "minor gap",
                "issues": ["short appendix"],
            },
            "pass_with_warning",
            "minor gap Issues: short appendix",
        ),
        (
            {
                "verdict": "revise",
                "rationale": "missing key fact",
                "revision_actions": ["add source result"],
            },
            "revise",
            "missing key fact Actions: add source result",
        ),
    ],
)
async def test_compile_judge_parses_quality_verdicts(payload, expected_verdict, expected_feedback):
    judge = CompileJudge(_Provider([json.dumps(payload)]), model="judge-model")

    verdict, feedback = await judge.judge_quality(
        reason="Compile",
        skill_contract="Write Wiki",
        coverage={"complete": True},
        bundle=_empty_bundle(),
    )

    assert verdict == expected_verdict
    assert feedback == expected_feedback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "feedback", "expected_warnings"),
    [
        ("pass", "", []),
        ("pass_with_warning", "Minor style issue", ["Minor style issue"]),
    ],
)
async def test_submission_accepts_quality_pass_variants(verdict, feedback, expected_warnings):
    async def quality_judge(bundle, payloads):
        assert bundle == _empty_bundle()
        assert payloads == []
        return verdict, feedback

    tool = SubmitWikiBundleTool(
        source_ids=set(),
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        quality_judge=quality_judge,
    )

    result = await tool.execute(ToolContext(), pages=[])

    assert result.startswith("Wiki bundle accepted")
    assert tool.warnings == expected_warnings
    assert tool.requires_partial is False


@pytest.mark.asyncio
async def test_duplicate_submission_cannot_clear_an_already_accepted_bundle():
    tool = SubmitWikiBundleTool(
        source_ids=set(),
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
    )

    accepted = await tool.execute(ToolContext(), pages=[])
    duplicate = await tool.execute(
        ToolContext(),
        pages=[{"not": "a valid page"}],
    )

    assert accepted.startswith("Wiki bundle accepted")
    assert duplicate == "Compile bundle was already accepted; duplicate submission ignored."
    assert tool.bundle == _empty_bundle()


@pytest.mark.asyncio
async def test_submission_requests_one_revision_then_accepts_unchanged_as_partial():
    calls = 0

    async def quality_judge(bundle, payloads):
        nonlocal calls
        assert bundle == _empty_bundle()
        assert payloads == []
        calls += 1
        return "revise", "Add the missing key conclusion"

    tool = SubmitWikiBundleTool(
        source_ids=set(),
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        quality_judge=quality_judge,
    )

    first = await tool.execute(ToolContext(), pages=[])
    second = await tool.execute(ToolContext(), pages=[])

    assert first.startswith("Error: Quality review found material issues")
    assert tool.candidate_bundle == _empty_bundle()
    assert second.startswith("Wiki bundle accepted")
    assert calls == 1
    assert tool.requires_partial is True
    assert tool.partial_reason == "The agent kept a result with unresolved quality findings."


@pytest.mark.asyncio
async def test_two_same_iteration_submits_cannot_bypass_revision_feedback():
    judged_titles = []

    async def quality_judge(bundle, payloads):
        del payloads
        judged_titles.append(bundle.pages[0].title)
        return (
            ("revise", "Fix the material issue")
            if len(judged_titles) == 1
            else ("pass", "")
        )

    tool = SubmitWikiBundleTool(
        source_ids={"src_1"},
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        quality_judge=quality_judge,
    )

    def page(title):
        return {
            "page_id": 1,
            "title": title,
            "page_type": "concept",
            "summary": f"Summary for {title}",
            "body_markdown": f"Body for {title}",
            "source_ids": ["src_1"],
            "path_hint": f"{title.casefold()}.md",
        }

    iteration_one = ToolContext(iteration=1, iteration_limit=10)
    first, preauthored_retry = await asyncio.gather(
        tool.execute(iteration_one, pages=[page("First")]),
        tool.execute(iteration_one, pages=[page("Second")]),
    )

    assert first.startswith("Error: Quality review found material issues")
    assert preauthored_retry.startswith("Error: Only one Compile submission")
    assert judged_titles == ["First"]

    accepted = await tool.execute(
        ToolContext(iteration=2, iteration_limit=10), pages=[page("Second")]
    )
    assert accepted.startswith("Wiki bundle accepted")
    assert judged_titles == ["First", "Second"]


@pytest.mark.asyncio
async def test_changed_artifact_bytes_trigger_a_second_quality_judgment():
    current_payload = b"first artifact bytes"
    judged_payloads = []

    class Sandbox:
        async def read_file_bytes(self, path, *, max_bytes=None):
            assert path == "output/artifact.bin"
            assert max_bytes == CompileLimits().output_total_bytes
            return current_payload

    class Manager:
        async def get_sandbox(self, session_key):
            assert session_key is not None
            return Sandbox()

    async def quality_judge(bundle, payloads):
        assert bundle.files[0].workspace_path == "output/artifact.bin"
        judged_payloads.append(payloads[0])
        return (
            ("revise", "Update the artifact")
            if len(judged_payloads) == 1
            else ("pass", "")
        )

    context = ToolContext(
        session_key=SimpleNamespace(type="compile"),
        sandbox_manager=Manager(),
    )
    tool = SubmitWikiBundleTool(
        source_ids=set(),
        catalog_uris=set(),
        target_uri="viking://resources/wiki",
        limits=CompileLimits(),
        quality_judge=quality_judge,
    )
    files = [{"path": "artifact.bin", "workspace_path": "output/artifact.bin"}]

    first = await tool.execute(context, pages=[], files=files)
    current_payload = b"second artifact bytes"
    second = await tool.execute(context, pages=[], files=files)

    assert first.startswith("Error: Quality review found material issues")
    assert second.startswith("Wiki bundle accepted")
    assert judged_payloads == [b"first artifact bytes", b"second artifact bytes"]
    assert tool.requires_partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["malformed", "timeout"])
async def test_service_retries_judge_failure_twice_then_fails_open(failure):
    provider = _Provider(
        ["not-json", "still-not-json"],
        timeout=failure == "timeout",
    )
    judge = CompileJudge(
        provider,
        model="judge-model",
        timeout_seconds=0.001,
    )
    _, registry = _registry(judge)
    submit = registry.get("submit_wiki_bundle")

    result = await submit.execute(ToolContext(), pages=[])

    assert result.startswith("Wiki bundle accepted")
    assert len(provider.calls) == 2
    assert len(submit.warnings) == 1
    assert submit.warnings[0].startswith("Quality review unavailable:")
    assert submit.requires_partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["deny", "failure"])
async def test_extension_is_near_limit_once_only_and_fail_closed(outcome):
    calls = []
    applied = []

    async def extend(packet):
        calls.append(packet)
        if outcome == "failure":
            raise asyncio.TimeoutError("judge timed out")
        return False, "remaining work is not specific"

    def apply_extension():
        applied.append(True)
        return 30

    tool = RequestCompileExtensionTool(
        coverage=CoverageLedger(()),
        extend=extend,
        apply_extension=apply_extension,
        near_limit=5,
    )
    arguments = {
        "reason": "More source analysis is needed",
        "completed_work": "Reviewed two documents",
        "remaining_work": "Review the final document",
        "next_actions": "Read it and submit",
    }

    too_early = await tool.execute(
        ToolContext(iteration=4, iteration_limit=10), **arguments
    )
    near_limit = await tool.execute(
        ToolContext(iteration=5, iteration_limit=10), **arguments
    )
    repeated = await tool.execute(
        ToolContext(iteration=6, iteration_limit=10), **arguments
    )

    assert too_early.startswith("Error: Request an extension only near")
    assert near_limit.startswith("Extension denied")
    if outcome == "failure":
        assert "independent review failed" in near_limit
    assert repeated.startswith("Error: The Compile iteration extension has already")
    assert len(calls) == 1
    assert applied == []


@pytest.mark.asyncio
async def test_service_extension_approval_updates_request_limit_once():
    class Judge:
        def __init__(self):
            self.calls = []

        async def judge_extension(self, packet):
            self.calls.append(packet)
            return True, "concrete work remains"

    judge = Judge()
    request_loop, registry = _registry(judge, max_iterations=40)
    extension = registry.get("request_compile_extension")
    arguments = {
        "reason": "One source remains",
        "completed_work": "Reviewed the first sources",
        "remaining_work": "Read the final source",
        "next_actions": "Read, synthesize, submit",
    }

    approved = await extension.execute(
        ToolContext(iteration=36, iteration_limit=40), **arguments
    )
    repeated = await extension.execute(
        ToolContext(iteration=37, iteration_limit=40), **arguments
    )

    assert approved.startswith("Extension approved")
    assert request_loop.max_iterations == 60
    assert len(judge.calls) == 1
    assert repeated.startswith("Error: The Compile iteration extension has already")
