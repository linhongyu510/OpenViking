"""Request-local tools used by the compile structured task."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from openviking.core.namespace import context_type_for_uri, relative_uri_path
from openviking.core.skill_loader import SkillLoader, validate_skill_format
from openviking.session.memory.dataclass import WikiLink
from openviking.session.memory.utils.link_renderer import LinkRenderer
from openviking.utils.path_safety import (
    safe_join_viking_uri,
    sanitize_relative_viking_path,
    validate_safe_viking_uri_path,
)
from openviking.utils.skill_processor import validate_skill_name
from openviking_cli.exceptions import OpenVikingError
from vikingbot.agent.tools.base import Tool, ToolContext
from vikingbot.compile.coverage import CoverageError, CoverageLedger, ReadReceipt
from vikingbot.compile.models import (
    COMPILE_STAGING_ROOT,
    COMPILE_WIKI_PAGE_ROOT,
    CompileLimits,
    WikiBundleDraft,
)
from vikingbot.compile.renderer import (
    is_reserved_wiki_page_uri,
    validate_declared_okf_markdown,
    validate_relative_file_path,
    validate_relative_page_path,
    wiki_page_path_from_title,
)

_LINK_FIELDS = frozenset({"f", "t", "link_type", "weight", "match_text", "description"})


def _normalize_workspace_path(path: str) -> str:
    normalized = sanitize_relative_viking_path(path)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _path_is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _uri_in_roots(uri: str, roots: tuple[str, ...]) -> bool:
    normalized = str(uri or "").strip().rstrip("/")
    if not normalized.startswith("viking://"):
        return False
    try:
        normalized = validate_safe_viking_uri_path(normalized)
    except ValueError:
        return False
    return any(
        normalized == root.rstrip("/") or bool(relative_uri_path(root, normalized))
        for root in roots
    )


def _skill_workspace_read_hint(uri: str) -> str | None:
    value = str(uri or "").strip()
    if value.startswith("viking://"):
        segments = value[len("viking://") :].split("/")
        if segments[:1] == ["skills"]:
            skills_at = 0
        elif segments[:2] == ["agent", "skills"]:
            skills_at = 1
        elif len(segments) >= 3 and segments[0] == "user" and segments[2] == "skills":
            skills_at = 2
        else:
            return None
        value = "/".join(segments[skills_at:])
    if not value.startswith("skills/"):
        return None
    try:
        return _normalize_workspace_path(value)
    except ValueError:
        return None


class CompileScopedTool(Tool):
    """Guard an existing OpenViking read tool without changing its implementation."""

    def __init__(
        self,
        tool: Tool,
        *,
        roots: tuple[str, ...],
        limits: CompileLimits,
        result_budget: dict[str, int],
        budget_lock: asyncio.Lock,
        coverage: CoverageLedger | None = None,
        review_views: Mapping[str, str | Callable[[], Awaitable[str]]] | None = None,
        evidence_cache: dict[str, str] | None = None,
    ):
        self._tool = tool
        self._roots = roots
        self._limits = limits
        self._result_budget = result_budget
        self._budget_lock = budget_lock
        self._coverage = coverage
        self._review_views = dict(review_views or {})
        self._evidence_cache = evidence_cache if evidence_cache is not None else {}

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._tool.parameters

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        uris: list[str] = []
        if self.name == "openviking_search":
            value = kwargs.get("target_uri")
            if not value:
                return "Error: Compile search requires target_uri within the task scope."
            uris.append(str(value))
        elif self.name in {"openviking_list", "openviking_grep", "openviking_glob"}:
            value = kwargs.get("uri")
            if not value or str(value).rstrip("/") in {"viking:", "viking://"}:
                return f"Error: Compile {self.name} requires uri within the task scope."
            uris.append(str(value))
            if self.name == "openviking_list" and kwargs.get("recursive"):
                kwargs["node_limit"] = min(
                    int(kwargs.get("node_limit") or self._limits.target_inventory_entries),
                    self._limits.target_inventory_entries,
                )
        elif self.name == "openviking_multi_read":
            values = kwargs.get("uris")
            if not isinstance(values, list) or not values:
                return "Error: Compile multi-read requires at least one URI."
            if len(values) > self._limits.tool_uri_count:
                return "Error: Compile multi-read URI limit exceeded."
            uris.extend(str(value) for value in values)

        if len(uris) > self._limits.tool_uri_count:
            return "Error: Compile tool URI limit exceeded."
        for uri in uris:
            workspace_path = _skill_workspace_read_hint(uri)
            if workspace_path:
                return (
                    "Error: Skill workspace files must be read with read_file using path "
                    f'"{workspace_path}", not with an openviking_* tool.'
                )
            if not _uri_in_roots(uri, self._roots):
                return f"Error: URI is outside the Compile task scope: {uri}"

        pending_receipts: dict[str, ReadReceipt] = {}
        pending_evidence: dict[str, str] = {}
        pending_non_substantive: list[tuple[tuple[str, ...], str]] = []
        if self.name == "openviking_multi_read" and self._coverage is not None:
            requested = [str(value).rstrip("/") for value in kwargs.get("uris", [])]
            for uri in requested:
                # Reads under the target catalog are useful for incremental
                # compilation but are not source review units.  They must stay
                # readable without manufacturing coverage receipts.
                try:
                    units = self._units_for_locator(uri)
                except CoverageError:
                    continue
                allowed = (
                    self._limits.adapted_input_bytes
                    if uri in self._review_views
                    else self._limits.tool_result_bytes
                )
                known_sizes = [
                    unit.discovered_bytes
                    for unit in units
                    if unit.discovered_bytes is not None
                ]
                if known_sizes and max(known_sizes) > allowed:
                    return (
                        "Error: Compile review unit exceeds its safe read budget; use a "
                        "Harness-verified summary or report a concrete filtering reason: "
                        f"{uri} ({max(known_sizes)} bytes > {allowed} bytes)."
                    )
            adapted = [uri for uri in requested if uri in self._review_views]
            delegated = [uri for uri in requested if uri not in self._review_views]
            sections: list[str] = []
            section_bytes = 0

            def append_section(*parts: str) -> bool:
                nonlocal section_bytes
                added = sum(len(part.encode("utf-8")) for part in parts) + len(parts)
                if section_bytes + added > self._limits.tool_result_bytes:
                    return False
                sections.extend(parts)
                section_bytes += added
                return True

            for uri in adapted:
                view = self._review_views[uri]
                content = await view() if callable(view) else view
                units = self._units_for_locator(uri)
                if not content.strip():
                    if not append_section(
                        f"--- START OF {uri} ---",
                        "[Harness: no substantive Codex conversation content]",
                        f"--- END OF {uri} ---",
                    ):
                        return "Error: Compile tool result exceeds the per-call size limit."
                    pending_non_substantive.append(
                        (
                            tuple(unit.unit_id for unit in units),
                            "Codex adapter inspected the full rollout and verified that it "
                            "contains only metadata or control records",
                        )
                    )
                    continue
                if not append_section(
                    f"--- START OF {uri} ---",
                    content,
                    f"--- END OF {uri} ---",
                ):
                    return "Error: Compile tool result exceeds the per-call size limit."
                for unit in units:
                    snapshot = next(
                        item
                        for item in self._coverage.snapshots
                        if item.source_id == unit.source_id
                    )
                    pending_receipts[unit.unit_id] = ReadReceipt.for_content(
                        snapshot, unit.unit_id, content
                    )
                    pending_evidence[unit.unit_id] = content
            if delegated:
                # Execute one URI per underlying call.  This preserves per-URI
                # success attribution: content from one untrusted source cannot
                # forge another URI's START/END markers and earn its receipt.
                for uri in delegated:
                    rendered_result = str(
                        await self._tool.execute(tool_context, **{**kwargs, "uris": [uri]})
                    )
                    if not append_section(rendered_result):
                        return "Error: Compile tool result exceeds the per-call size limit."
                    for receipt in self._receipts_for_single_read(uri, rendered_result):
                        pending_receipts[receipt.unit_id] = receipt
                        pending_evidence[receipt.unit_id] = self._single_read_content(
                            uri, rendered_result
                        )
            result = "\n".join(sections)
        else:
            result = await self._tool.execute(tool_context, **kwargs)
        if (
            isinstance(result, str)
            and result.startswith("Error")
            and not result.startswith("Error:")
        ):
            result = "Error: " + result[len("Error") :].lstrip(" :")
        rendered = str(result)
        size = len(rendered.encode("utf-8"))
        if size > self._limits.tool_result_bytes:
            return "Error: Compile tool result exceeds the per-call size limit."
        async with self._budget_lock:
            total = self._result_budget.get("bytes", 0) + size
            if total > self._limits.tool_total_result_bytes:
                return "Error: Compile task tool-result budget exceeded."
            self._result_budget["bytes"] = total
            # A read counts only after the exact result has passed every delivery
            # budget and is about to be returned to the model.  The batch method
            # pre-validates all receipts before mutating the ledger.
            if self._coverage is not None and pending_receipts:
                self._coverage.mark_reviewed(
                    tuple(pending_receipts),
                    pending_receipts,
                    iteration=tool_context.iteration,
                )
                self._evidence_cache.update(pending_evidence)
            if self._coverage is not None:
                for unit_ids, reason in pending_non_substantive:
                    self._coverage.mark_harness_non_substantive(
                        unit_ids,
                        reason,
                        iteration=tool_context.iteration,
                    )
        return rendered

    def _units_for_locator(self, uri: str):
        assert self._coverage is not None
        units = tuple(unit for unit in self._coverage.units if unit.locator == uri)
        if not units:
            raise CoverageError(f"read URI is not a discovered review unit: {uri}")
        return units

    @staticmethod
    def _single_read_content(uri: str, rendered: str) -> str:
        start = f"--- START OF {uri} ---\n"
        end = f"\n--- END OF {uri} ---"
        start_at = rendered.find(start)
        end_at = rendered.rfind(end)
        if start_at < 0 or end_at < start_at + len(start):
            return ""
        return rendered[start_at + len(start) : end_at]

    def _receipts_for_single_read(self, uri: str, rendered: str) -> tuple[ReadReceipt, ...]:
        if self._coverage is None or rendered.lstrip().startswith("Error"):
            return ()
        content = self._single_read_content(uri, rendered)
        if not content or content.lstrip().startswith("ERROR:"):
            return ()
        try:
            units = self._units_for_locator(uri)
        except CoverageError:
            return ()
        receipts: list[ReadReceipt] = []
        try:
            for unit in units:
                snapshot = next(
                    item
                    for item in self._coverage.snapshots
                    if item.source_id == unit.source_id
                )
                receipts.append(ReadReceipt.for_content(snapshot, unit.unit_id, content))
            return tuple(receipts)
        except CoverageError:
            # The unit remains pending when a discovered content digest no
            # longer matches what was delivered.
            return ()


class ReportCoverageTool(Tool):
    """Let the agent account for units without restating the full ledger."""

    def __init__(self, coverage: CoverageLedger):
        self.coverage = coverage

    @property
    def name(self) -> str:
        return "report_compile_coverage"

    @property
    def description(self) -> str:
        return (
            "Account for discovered Compile review units. Use skip only with a concrete, "
            "auditable filtering reason. Use cover only with a Harness-verified summary."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["skip", "cover", "status"]},
                "unit_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 200,
                },
                "reason": {"type": "string", "maxLength": 1200},
                "summary_unit_id": {"type": "string"},
                "source_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        action: str,
        unit_ids: list[str] | None = None,
        reason: str | None = None,
        summary_unit_id: str | None = None,
        source_id: str | None = None,
        offset: int = 0,
        limit: int = 100,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            if action == "skip":
                if len(unit_ids or []) > 200 or len(reason or "") > 1200:
                    return "Error: Coverage skip exceeds the per-call reporting limit."
                self.coverage.skip(
                    unit_ids or [], reason or "", iteration=tool_context.iteration
                )
            elif action == "cover":
                if len(unit_ids or []) > 200:
                    return "Error: Coverage cover exceeds the per-call reporting limit."
                self.coverage.cover(
                    unit_ids or [],
                    summary_unit_id or "",
                    iteration=tool_context.iteration,
                )
            elif action != "status":
                return f"Error: Unsupported coverage action: {action}"
        except CoverageError as exc:
            return f"Error: Invalid coverage report: {exc}"
        summary = self.coverage.summary()
        issues = [
            issue if len(issue) <= 600 else issue[:600] + "..."
            for issue in summary["issues"][:8]
        ]
        visible_units = [
            unit
            for unit in self.coverage.units
            if source_id is None or unit.source_id == source_id
        ][offset : offset + limit]
        return json.dumps(
            {
                "accepted": True,
                "counts": summary["counts"],
                "complete": summary["complete"],
                "issues": issues,
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "source_id": unit.source_id,
                        "locator": unit.locator,
                        "kind": unit.kind,
                        "status": self.coverage.status(unit.unit_id).value,
                    }
                    for unit in visible_units
                ],
                "next_offset": (
                    offset + len(visible_units)
                    if offset + len(visible_units) < len(
                        [
                            unit
                            for unit in self.coverage.units
                            if source_id is None or unit.source_id == source_id
                        ]
                    )
                    else None
                ),
            },
            ensure_ascii=False,
        )


class RequestCompileExtensionTool(Tool):
    """Request one bounded extension from an independent fresh-context judge."""

    def __init__(
        self,
        *,
        coverage: CoverageLedger,
        extend: Callable[[Mapping[str, Any]], Awaitable[tuple[bool, str]]],
        apply_extension: Callable[[], int],
        near_limit: int = 5,
    ):
        self.coverage = coverage
        self.extend = extend
        self.apply_extension = apply_extension
        self.near_limit = near_limit
        self.used = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "request_compile_extension"

    @property
    def description(self) -> str:
        return (
            "Near the normal iteration limit, request one bounded extension for genuinely "
            "long Compile work. The extension never increases the wall-clock deadline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        field = {"type": "string", "minLength": 1, "maxLength": 1200}
        return {
            "type": "object",
            "properties": {
                "reason": field,
                "completed_work": field,
                "remaining_work": field,
                "next_actions": field,
            },
            "required": ["reason", "completed_work", "remaining_work", "next_actions"],
        }

    async def execute(
        self,
        tool_context: ToolContext,
        reason: str,
        completed_work: str,
        remaining_work: str,
        next_actions: str,
        **kwargs: Any,
    ) -> str:
        del kwargs
        async with self._lock:
            if self.used:
                return "Error: The Compile iteration extension has already been decided."
            iteration = int(tool_context.iteration or 0)
            limit = int(tool_context.iteration_limit or 0)
            if limit <= 0 or limit - iteration > self.near_limit:
                return (
                    "Error: Request an extension only near the normal iteration limit; "
                    f"currently at iteration {iteration}/{limit}."
                )
            self.used = True
            packet = {
                "iteration": iteration,
                "iteration_limit": limit,
                "coverage": self.coverage.summary(),
                "reason": reason,
                "completed_work": completed_work,
                "remaining_work": remaining_work,
                "next_actions": next_actions,
            }
            try:
                approved, explanation = await self.extend(packet)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return f"Extension denied because the independent review failed: {exc}"
            if not approved:
                return f"Extension denied: {explanation[:600]}"
            new_limit = self.apply_extension()
            return (
                f"Extension approved. The request-local iteration limit is now {new_limit}; "
                "continue the remaining work and submit a bounded result."
            )


class SubmitWikiBundleTool(Tool):
    def __init__(
        self,
        *,
        source_ids: set[str],
        catalog_uris: set[str],
        file_catalog_uris: set[str] | None = None,
        target_uri: str,
        limits: CompileLimits,
        require_workspace_files: bool = False,
        require_workspace_pages: bool = False,
        workspace_baseline: set[str] | None = None,
        wiki_uri_resolver: Callable[[str], Awaitable[bool]] | None = None,
        exec_enabled: bool = True,
        coverage: CoverageLedger | None = None,
        quality_judge: (
            Callable[[WikiBundleDraft, list[bytes | None]], Awaitable[tuple[str, str]]] | None
        ) = None,
    ):
        self.source_ids = source_ids
        self.catalog_uris = catalog_uris
        self.file_catalog_uris = set(catalog_uris)
        self.file_catalog_uris.update(file_catalog_uris or ())
        self.target_uri = target_uri.rstrip("/")
        self.limits = limits
        self.require_workspace_files = require_workspace_files
        self.require_workspace_pages = require_workspace_pages
        self.workspace_baseline = (
            None
            if workspace_baseline is None
            else {_normalize_workspace_path(path) for path in workspace_baseline}
        )
        self.wiki_uri_resolver = wiki_uri_resolver
        self.exec_enabled = exec_enabled
        self.coverage = coverage
        self.quality_judge = quality_judge
        self.bundle: WikiBundleDraft | None = None
        self.file_payloads: list[bytes | None] = []
        self.skill_name: str | None = None
        self.warnings: list[str] = []
        self.candidate_bundle: WikiBundleDraft | None = None
        self.candidate_file_payloads: list[bytes | None] = []
        self.candidate_skill_name: str | None = None
        self.candidate_warnings: list[str] = []
        self.candidate_partial_reason: str | None = None
        self.candidate_coverage_issue: str | None = None
        self._judge_calls = 0
        self._revision_requested = False
        self._last_candidate_hash: str | None = None
        self._last_submit_iteration: int | None = None
        self._lock = asyncio.Lock()
        self.requires_partial = False
        self.partial_reason: str | None = None

    @property
    def _is_skill_target(self) -> bool:
        return context_type_for_uri(self.target_uri) == "skill"

    @property
    def name(self) -> str:
        return "submit_wiki_bundle"

    @property
    def description(self) -> str:
        artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
        workspace_notice = (
            f" Generate artifact files with {artifact_writers}, then reference them with "
            "workspace_path; do not inline file content."
            if self.require_workspace_files
            else ""
        )
        if self._is_skill_target:
            return (
                "Submit one complete OpenViking Skill package. Include every file under "
                "<skill-name>/ and include <skill-name>/SKILL.md."
                f"{workspace_notice}"
            )
        return (
            "Submit the final output only after every path and format explicitly required "
            "by the Skill is represented. Treat only actual Wiki content as Wiki pages and "
            f"preserve exact-path Skill outputs as artifact files.{workspace_notice}"
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if "raw" in params:
            message = "use the tool schema directly; do not wrap the payload in a JSON string"
            if self.require_workspace_files:
                artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
                message += (
                    f"; generate artifact files with {artifact_writers} and submit them using "
                    "workspace_path instead of inline content"
                )
            return [message]
        return super().validate_params(params)

    @property
    def parameters(self) -> dict[str, Any]:
        schema = WikiBundleDraft.model_json_schema()
        required = schema.setdefault("required", [])
        if "files" not in required:
            required.append("files")
        definitions = schema.get("$defs", {})
        if self.require_workspace_files:
            file_schema = definitions.get("CompileFileDraft", {})
            file_properties = file_schema.get("properties", {})
            if isinstance(file_properties, dict):
                file_properties.pop("content", None)
            file_required = file_schema.setdefault("required", [])
            if "content" in file_required:
                file_required.remove("content")
            if "workspace_path" not in file_required:
                file_required.append("workspace_path")
        if self._is_skill_target:
            schema["properties"].pop("pages", None)
            schema["properties"].pop("links", None)
            required[:] = [field for field in required if field not in {"pages", "links"}]
            definitions.pop("WikiPageDraft", None)
            definitions.pop("WikiLink", None)
            file_schema = definitions.get("CompileFileDraft", {})
            file_schema.get("properties", {}).pop("update_uri", None)
            file_required = file_schema.setdefault("required", [])
            if "path" not in file_required:
                file_required.append("path")
            schema.pop("title", None)
            return schema
        if self.require_workspace_pages:
            page_def = schema.get("$defs", {}).get("WikiPageDraft", {})
            page_properties = page_def.get("properties", {})
            if isinstance(page_properties, dict):
                page_properties.pop("body_markdown", None)
            page_required = page_def.setdefault("required", [])
            if "body_markdown" in page_required:
                page_required.remove("body_markdown")
            if "body_workspace_path" not in page_required:
                page_required.append("body_workspace_path")
        link_def = schema.get("$defs", {}).get("WikiLink", {})
        match_schema = link_def.get("properties", {}).get("match_text")
        if isinstance(match_schema, dict):
            match_schema["description"] = (
                "Exact anchor text that must either appear in the source page draft body "
                "outside frontmatter, code, existing Markdown links, and Citations, or "
                "already be part of a Markdown link to the target page."
            )
        schema.pop("title", None)
        return schema

    async def execute(
        self,
        tool_context: ToolContext,
        pages: list[dict[str, Any]] | None = None,
        files: list[dict[str, Any]] | None = None,
        links: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        async with self._lock:
            if (
                tool_context.iteration is not None
                and self._last_submit_iteration == tool_context.iteration
            ):
                return (
                    "Error: Only one Compile submission is evaluated per iteration. "
                    "Read the prior tool result before submitting again."
                )
            self._last_submit_iteration = tool_context.iteration
            if self.bundle is not None:
                return "Compile bundle was already accepted; duplicate submission ignored."
            return await self._execute_locked(
                tool_context, pages=pages, files=files, links=links
            )

    async def _execute_locked(
        self,
        tool_context: ToolContext,
        *,
        pages: list[dict[str, Any]] | None,
        files: list[dict[str, Any]] | None,
        links: list[dict[str, Any]] | None,
    ) -> str:
        self.bundle = None
        self.file_payloads = []
        self.skill_name = None
        self.warnings = []
        self.requires_partial = False
        self.partial_reason = None
        raw_links = links or []
        parsed_links: list[WikiLink] = []
        for index, link in enumerate(raw_links):
            if not isinstance(link, Mapping) or set(link) - _LINK_FIELDS:
                self.warnings.append(
                    f"Dropped invalid optional links[{index}]: unknown fields or non-object value."
                )
                continue
            try:
                parsed_links.append(WikiLink.model_validate(link))
            except ValidationError as exc:
                self.warnings.append(
                    f"Dropped invalid optional links[{index}]: {exc.errors()[0]['msg']}."
                )
        try:
            bundle = WikiBundleDraft.model_validate(
                {
                    "pages": pages or [],
                    "files": files or [],
                    "links": [link.model_dump() for link in parsed_links],
                }
            )
            await self._validate_workspace_manifest(
                bundle,
                tool_context=tool_context,
            )
            bundle = await self._materialize_page_bodies(bundle, tool_context=tool_context)
            bundle, payloads, link_warnings = await self._validate_bundle(
                bundle, tool_context=tool_context
            )
            self.warnings.extend(link_warnings)
        except (ValidationError, ValueError) as exc:
            kind = "Skill" if self._is_skill_target else "Wiki"
            return f"Error: Invalid {kind} bundle: {exc}"
        self.candidate_bundle = bundle
        self.candidate_file_payloads = payloads
        self.candidate_skill_name = self.skill_name
        self.candidate_warnings = list(self.warnings)
        if self.coverage is not None:
            gate_error = self.coverage.gate_error(before_iteration=tool_context.iteration)
            # Preserve the gate result from the exact submission turn.  A
            # concurrently executing read may update the live Ledger after the
            # candidate was authored, but that future evidence cannot justify
            # this candidate during iteration-limit recovery.
            self.candidate_coverage_issue = gate_error
            if gate_error:
                return (
                    "Error: Coverage gate rejected the bundle. Review or account for the "
                    f"remaining units, then resubmit. {gate_error[:2000]}"
                )
        else:
            self.candidate_coverage_issue = None

        candidate_hasher = hashlib.sha256(bundle.model_dump_json().encode("utf-8"))
        for payload in payloads:
            candidate_hasher.update(b"\0inline\0" if payload is None else payload)
        candidate_hash = candidate_hasher.hexdigest()
        if self.quality_judge is not None:
            if self._revision_requested and candidate_hash == self._last_candidate_hash:
                self.warnings.append(
                    "Quality review suggested a revision; the agent kept the submitted result."
                )
                self.requires_partial = True
                self.partial_reason = "The agent kept a result with unresolved quality findings."
            elif self._judge_calls >= 2:
                self.warnings.append(
                    "Quality review reached its two-call limit; the latest safe result was kept."
                )
                self.requires_partial = True
                self.partial_reason = "Quality review did not converge within one revision."
            else:
                try:
                    verdict, feedback = await self.quality_judge(bundle, payloads)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    verdict, feedback = (
                        "pass_with_warning",
                        f"Quality review was unavailable after retry: {exc}",
                    )
                self._judge_calls += 1
                if verdict == "revise" and self._judge_calls == 1:
                    self._revision_requested = True
                    self._last_candidate_hash = candidate_hash
                    self.partial_reason = feedback or "Quality review requested a revision."
                    self.candidate_partial_reason = self.partial_reason
                    return (
                        "Error: Quality review found material issues. You may revise once, or "
                        f"resubmit unchanged to keep this safe result. {feedback[:1600]}"
                    )
                if verdict == "revise":
                    self.warnings.append(
                        "Quality review still found material issues after the bounded revision."
                    )
                    self.requires_partial = True
                    self.partial_reason = (
                        "Material quality findings remain after the bounded revision."
                    )
                elif verdict == "pass_with_warning" and feedback:
                    self.warnings.append(feedback[:1600])
        self.bundle = bundle
        self.file_payloads = payloads
        if self._is_skill_target:
            return (
                f"Skill bundle accepted for '{self.skill_name}' with {len(bundle.files)} file(s)."
            )
        return (
            f"Wiki bundle accepted with {len(bundle.pages)} page(s) and "
            f"{len(bundle.files)} file(s)."
        )

    async def _list_workspace_files(
        self,
        *,
        tool_context: ToolContext,
    ) -> set[str]:
        if tool_context.sandbox_manager is None:
            raise ValueError("task sandbox is unavailable")
        sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
        files: set[str] = set()
        pending = [""]
        visited = 0
        while pending:
            directory = pending.pop()
            try:
                entries = await sandbox.list_dir(directory or ".")
            except Exception as exc:
                raise ValueError("task workspace could not be inspected") from exc
            for name, is_dir in entries:
                relative = _normalize_workspace_path(f"{directory}/{name}" if directory else name)
                visited += 1
                if visited > self.limits.target_inventory_entries:
                    raise ValueError("task workspace inventory limit exceeded")
                if _path_is_within(relative, COMPILE_STAGING_ROOT):
                    continue
                if name in {".git", "__pycache__"}:
                    continue
                if is_dir:
                    pending.append(relative)
                elif not relative.endswith((".pyc", ".pyo")):
                    files.add(relative)
        return files

    async def _validate_workspace_manifest(
        self,
        bundle: WikiBundleDraft,
        *,
        tool_context: ToolContext,
    ) -> None:
        if context_type_for_uri(self.target_uri) != "resource":
            return
        page_paths = {
            _normalize_workspace_path(page.body_workspace_path)
            for page in bundle.pages
            if page.body_workspace_path is not None
        }
        artifact_paths = {
            _normalize_workspace_path(file.workspace_path)
            for file in bundle.files
            if file.workspace_path is not None
        }
        errors: list[str] = []
        invalid_pages = sorted(
            path for path in page_paths if not _path_is_within(path, COMPILE_WIKI_PAGE_ROOT)
        )
        if self.require_workspace_pages and invalid_pages:
            errors.append(
                "Wiki page body workspace paths must be temporary files under "
                f"{COMPILE_WIKI_PAGE_ROOT}/, not Skill artifact paths: " + ", ".join(invalid_pages)
            )
        invalid_artifacts = sorted(
            path for path in artifact_paths if _path_is_within(path, COMPILE_STAGING_ROOT)
        )
        if invalid_artifacts:
            errors.append(
                "Skill artifact workspace paths must remain outside the Compile staging "
                "directory: " + ", ".join(invalid_artifacts)
            )

        if self.workspace_baseline is not None:
            current_files = await self._list_workspace_files(tool_context=tool_context)
            generated_artifacts = current_files - self.workspace_baseline
            missing_artifacts = sorted(generated_artifacts - artifact_paths)
            if missing_artifacts:
                errors.append(
                    "generated Skill artifacts are missing from files; preserve their "
                    "required paths and submit them unchanged: " + ", ".join(missing_artifacts)
                )
        if errors:
            raise ValueError("; ".join(errors))

    async def _read_workspace_bytes(
        self,
        workspace_path: str,
        *,
        tool_context: ToolContext,
        label: str,
        max_bytes: int,
    ) -> bytes:
        try:
            relative = _normalize_workspace_path(workspace_path)
            if tool_context.sandbox_manager is None:
                raise ValueError("task sandbox is unavailable")
            sandbox = await tool_context.sandbox_manager.get_sandbox(tool_context.session_key)
            return await sandbox.read_file_bytes(relative, max_bytes=max_bytes)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{label} workspace path could not be read: {workspace_path}") from exc

    async def _materialize_page_bodies(
        self,
        bundle: WikiBundleDraft,
        *,
        tool_context: ToolContext,
    ) -> WikiBundleDraft:
        artifact_workspace_paths = {
            _normalize_workspace_path(file.workspace_path)
            for file in bundle.files
            if file.workspace_path is not None
        }
        pages = []
        page_bytes = 0
        for page in bundle.pages:
            if self.require_workspace_pages and page.body_markdown is not None:
                raise ValueError(
                    f"page {page.page_id} body must be generated with write_file and "
                    "submitted using body_workspace_path instead of inline Markdown"
                )
            if page.body_workspace_path is None:
                assert page.body_markdown is not None
                page_bytes += len(page.body_markdown.encode("utf-8"))
                if page_bytes > self.limits.output_total_bytes:
                    raise ValueError("draft content size limit exceeded")
                pages.append(page)
                continue
            workspace_path = _normalize_workspace_path(page.body_workspace_path)
            if workspace_path in artifact_workspace_paths:
                raise ValueError(
                    f"page {page.page_id} body must be a separate reader-oriented "
                    "workspace file, not an exact artifact file"
                )
            raw = await self._read_workspace_bytes(
                workspace_path,
                tool_context=tool_context,
                label=f"page {page.page_id} body",
                max_bytes=self.limits.output_total_bytes - page_bytes,
            )
            page_bytes += len(raw)
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"page {page.page_id} body_workspace_path must contain UTF-8 Markdown"
                ) from exc
            pages.append(
                page.model_copy(update={"body_markdown": body, "body_workspace_path": None})
            )
        return bundle.model_copy(update={"pages": pages})

    async def _validate_bundle(
        self, bundle: WikiBundleDraft, *, tool_context: ToolContext
    ) -> tuple[WikiBundleDraft, list[bytes | None], list[str]]:
        target_type = context_type_for_uri(self.target_uri)
        if len(bundle.pages) > self.limits.output_pages:
            raise ValueError("page limit exceeded")
        if len(bundle.files) > self.limits.output_files:
            raise ValueError("file limit exceeded")
        if len(bundle.pages) + len(bundle.files) > self.limits.output_operations:
            raise ValueError("combined output operation limit exceeded")
        if not bundle.pages and bundle.links:
            raise ValueError("empty bundle must not contain links")
        if target_type == "skill" and (bundle.pages or bundle.links):
            raise ValueError("Skill targets only accept artifact files")
        if bundle.files and target_type not in {"resource", "skill"}:
            raise ValueError(
                "raw artifact files are only supported for Resource targets or exact "
                "Skill namespace targets; re-run ov compile with a supported target"
            )
        if self.require_workspace_files and any(file.content is not None for file in bundle.files):
            artifact_writers = "write_file or exec" if self.exec_enabled else "write_file"
            raise ValueError(
                f"artifact files must be generated with {artifact_writers} and submitted "
                "using workspace_path instead of inline content"
            )
        page_ids: set[int] = set()
        page_uris: dict[int, str] = {}
        final_uris: set[str] = set()
        total_bytes = 0
        for page in bundle.pages:
            if page.body_markdown is None:
                raise ValueError(f"page {page.page_id} body was not materialized")
            if page.page_id in page_ids:
                raise ValueError(f"duplicate page_id: {page.page_id}")
            page_ids.add(page.page_id)
            if not page.title.strip() or not page.page_type.strip() or not page.summary.strip():
                raise ValueError(f"page {page.page_id} has empty required fields")
            if "\n" in page.summary.strip() or "\r" in page.summary.strip():
                raise ValueError(f"page {page.page_id} summary must be one line")
            if page.body_markdown.lstrip().startswith("---"):
                raise ValueError(
                    f"page {page.page_id} must not include YAML frontmatter. If this is a "
                    "Skill-prescribed artifact, do not edit or strip its frontmatter; submit "
                    f"it through files and create a separate Wiki body under "
                    f"{COMPILE_WIKI_PAGE_ROOT}/"
                )
            if not page.source_ids or any(
                source_id not in self.source_ids for source_id in page.source_ids
            ):
                raise ValueError(f"page {page.page_id} has invalid source_ids")
            if page.update_uri:
                final_uri = page.update_uri.rstrip("/")
                if is_reserved_wiki_page_uri(final_uri):
                    raise ValueError(f"page {page.page_id} cannot update a reserved Wiki file")
                if not await self._is_wiki_uri(final_uri):
                    raise ValueError(
                        f"page {page.page_id} update_uri is not an existing OKF Wiki page"
                    )
                if page.path_hint:
                    raise ValueError(f"page {page.page_id} cannot rename an update")
            else:
                hint = page.path_hint or wiki_page_path_from_title(page.title)
                relative = validate_relative_page_path(hint)
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
                if final_uri in self.file_catalog_uris:
                    raise ValueError(f"page {page.page_id} path exists; use its update_uri")
            if final_uri in final_uris:
                raise ValueError(f"duplicate final Wiki path: {final_uri}")
            final_uris.add(final_uri)
            page_uris[page.page_id] = final_uri
            total_bytes += len(page.body_markdown.encode("utf-8"))

        file_payloads: list[bytes | None] = []
        for index, file in enumerate(bundle.files):
            if target_type == "skill":
                if file.update_uri:
                    raise ValueError("Skill bundles require relative path entries, not update_uri")
                relative = validate_relative_file_path(file.path or "")
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
            elif file.update_uri:
                final_uri = validate_safe_viking_uri_path(file.update_uri).rstrip("/")
                if is_reserved_wiki_page_uri(final_uri):
                    raise ValueError(f"file {index} cannot update a reserved file")
                if final_uri not in self.file_catalog_uris:
                    raise ValueError(f"file {index} update_uri is not in the catalog")
            else:
                relative = validate_relative_file_path(file.path or "")
                final_uri = safe_join_viking_uri(self.target_uri, relative).rstrip("/")
                if final_uri in self.file_catalog_uris:
                    raise ValueError(f"file {index} path exists; use its update_uri")
            if final_uri in final_uris:
                raise ValueError(f"duplicate final output path: {final_uri}")
            final_uris.add(final_uri)

            if file.content is not None:
                payload = None
                content_bytes = file.content.encode("utf-8")
            else:
                payload = await self._read_workspace_bytes(
                    file.workspace_path or "",
                    tool_context=tool_context,
                    label=f"file {index}",
                    max_bytes=self.limits.output_total_bytes - total_bytes,
                )
                content_bytes = payload
            total_bytes += len(content_bytes)
            if total_bytes > self.limits.output_total_bytes:
                raise ValueError("draft content size limit exceeded")
            if target_type == "resource":
                page_type = validate_declared_okf_markdown(final_uri, content_bytes)
                existing_wiki = bool(file.update_uri and await self._is_wiki_uri(final_uri))
                if existing_wiki and page_type is None:
                    raise ValueError(
                        f"file {index} updates an existing Wiki page and must retain "
                        "valid OKF frontmatter with a non-empty type"
                    )
            file_payloads.append(payload)

        if total_bytes > self.limits.output_total_bytes:
            raise ValueError("draft content size limit exceeded")
        if target_type == "skill":
            self.skill_name = self._validate_skill_bundle(bundle, file_payloads)
        page_by_id = {page.page_id: page for page in bundle.pages}
        valid_links: list[WikiLink] = []
        link_warnings: list[str] = []
        for index, link in enumerate(bundle.links):
            prefix = f"links[{index}]"
            error: str | None = None
            if link.f is None or link.t is None:
                error = "endpoints must be non-null"
            elif link.f == link.t:
                error = "must not be a self-link"
            elif link.f not in page_ids or link.t not in page_ids:
                error = "endpoints must reference bundle pages"
            elif not link.match_text:
                error = "match_text is required"
            else:
                source_page = page_by_id[link.f]
                if not LinkRenderer.can_render_link(
                    source_page.body_markdown,
                    link.match_text,
                    page_uris[link.f],
                    page_uris[link.t],
                ):
                    error = (
                        f"from page {link.f} has unsatisfied anchor {link.match_text!r}; "
                        "use exact unprotected text or an existing Markdown link to the target"
                    )
            if error is not None:
                link_warnings.append(f"Dropped invalid optional {prefix}: {error}.")
            else:
                valid_links.append(link)
        return (
            bundle.model_copy(update={"links": valid_links}),
            file_payloads,
            link_warnings,
        )

    async def _is_wiki_uri(self, uri: str) -> bool:
        if uri in self.catalog_uris:
            return True
        if uri not in self.file_catalog_uris or self.wiki_uri_resolver is None:
            return False
        if await self.wiki_uri_resolver(uri):
            self.catalog_uris.add(uri)
            return True
        return False

    @staticmethod
    def _validate_skill_bundle(bundle: WikiBundleDraft, file_payloads: list[bytes | None]) -> str:
        if not bundle.files:
            raise ValueError("Skill bundle must contain files")

        skill_names: set[str] = set()
        contents: dict[str, bytes] = {}
        for index, file in enumerate(bundle.files):
            relative = validate_relative_file_path(file.path or "")
            parts = relative.split("/")
            if len(parts) < 2:
                raise ValueError(f"file {index} must be under <skill-name>/, got: {relative}")
            skill_names.add(parts[0])
            payload = (
                file.content.encode("utf-8") if file.content is not None else file_payloads[index]
            )
            if payload is None:
                raise ValueError(f"file {index} has no materialized content")
            contents[relative] = payload

        if len(skill_names) != 1:
            raise ValueError("Skill bundle must contain exactly one top-level Skill directory")
        skill_name = next(iter(skill_names))
        skill_md_path = f"{skill_name}/SKILL.md"
        skill_md = contents.get(skill_md_path)
        if skill_md is None:
            raise ValueError(f"Skill bundle must include {skill_md_path}")
        try:
            skill_md_text = skill_md.decode("utf-8")
            parsed = SkillLoader.parse(skill_md_text, source_path=skill_md_path)
            parsed_name = validate_skill_name(parsed.get("name"))
        except (UnicodeDecodeError, ValueError, OpenVikingError, yaml.YAMLError) as exc:
            raise ValueError(str(exc)) from exc
        if parsed_name != skill_name:
            raise ValueError(f"Skill name '{parsed_name}' does not match directory '{skill_name}'")
        validation = validate_skill_format(
            skill_md_text,
            strict=True,
            skill_dir_name=skill_name,
            source_path=skill_md_path,
        )
        if not validation["valid"]:
            messages = [
                str(issue.get("message") or issue.get("rule") or "invalid Skill")
                for issue in validation["errors"]
            ]
            raise ValueError("; ".join(messages))
        return skill_name


__all__ = [
    "CompileScopedTool",
    "ReportCoverageTool",
    "RequestCompileExtensionTool",
    "SubmitWikiBundleTool",
]
