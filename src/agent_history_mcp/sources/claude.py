"""
Claude Code transcripts: ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl

Each line is a record (user / assistant / system / ai-title / ...). Assistant
content blocks include text, thinking, and tool_use; user content blocks carry
tool_result. Subagent transcripts live in <session-id>/subagents/*.jsonl and are
merged into the parent session as sidechain tool events.
"""

from __future__ import annotations

from pathlib import Path

from ..model import SessionBuilder
from . import SourceAdapter, SourceUnit, file_token, read_jsonl


def _claude_extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if isinstance(c, dict) and c.get("type") == "text":
            t = c.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts)


def iter_claude_session_files(claude_path: Path) -> list[Path]:
    projects_dir = claude_path / "projects"
    if not projects_dir.is_dir():
        return []
    return [
        path
        for path in sorted(projects_dir.rglob("*.jsonl"), reverse=True)
        if "subagents" not in path.parts
    ]


def subagent_files(session_file: Path) -> list[Path]:
    sub_dir = session_file.parent / session_file.stem / "subagents"
    if not sub_dir.is_dir():
        return []
    return sorted(sub_dir.glob("*.jsonl"))


def _feed_records(builder: SessionBuilder, objects: list[dict], sidechain: bool) -> dict[str, dict]:
    """Feed one transcript's records into the builder; returns usage keyed by message id."""
    usage_by_id: dict[str, dict] = {}
    for obj in objects:
        msg_type = obj.get("type", "")
        ts = obj.get("timestamp")
        if msg_type == "ai-title" and not sidechain:
            builder.title = obj.get("aiTitle", builder.title) or builder.title
            continue
        if msg_type == "system":
            if obj.get("subtype") == "compact_boundary" and not sidechain:
                builder.add_signal("compact", "context compacted", ts)
            continue
        if msg_type not in ("user", "assistant"):
            continue
        if not sidechain:
            builder.set_cwd(obj.get("cwd"))
            builder.set_branch(obj.get("gitBranch"))
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", msg_type)
        content = msg.get("content", [])

        if role == "assistant":
            model = msg.get("model")
            if model and not str(model).startswith("<"):
                builder.add_model(model)
            usage = msg.get("usage")
            usage_key = msg.get("id") or obj.get("requestId") or obj.get("uuid")
            if isinstance(usage, dict) and usage_key:
                usage_by_id[usage_key] = usage

        if not sidechain and not obj.get("isMeta"):
            text = _claude_extract_text(content)
            if text and not (text.strip().startswith("<") and len(text) > 2000):
                builder.add_message(role, text, ts)

        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                builder.add_tool_call(
                    block.get("name", ""),
                    block.get("input") or {},
                    ts,
                    call_id=block.get("id", ""),
                    sidechain=sidechain or bool(obj.get("isSidechain")),
                )
            elif block_type == "tool_result":
                builder.add_tool_result(
                    block.get("tool_use_id", ""),
                    block.get("content"),
                    ts,
                    is_error=True if block.get("is_error") else None,
                )
    return usage_by_id


def parse_claude_session_file(session_file: Path, include_subagents: bool = True) -> dict | None:
    builder = SessionBuilder("claude", session_file.stem, str(session_file))
    builder.title = session_file.stem[:20]
    usage = _feed_records(builder, read_jsonl(session_file), sidechain=False)

    if include_subagents:
        subagents = subagent_files(session_file)
        for sub_file in subagents:
            usage.update(_feed_records(builder, read_jsonl(sub_file), sidechain=True))
        if subagents:
            builder.events.sort(key=lambda e: (e.ts or "~", e.idx))
            for idx, event in enumerate(builder.events):
                event.idx = idx

    for item in usage.values():
        builder.add_tokens(
            (item.get("input_tokens") or 0)
            + (item.get("cache_read_input_tokens") or 0)
            + (item.get("cache_creation_input_tokens") or 0),
            item.get("output_tokens") or 0,
        )
    return builder.build()


def parse_claude_sessions(claude_path: Path) -> list[dict]:
    """
    Walk ~/.claude/projects/ and parse each top-level session JSONL.
    Subagent transcripts are folded into their parent session.
    """
    sessions = []
    for session_file in iter_claude_session_files(claude_path):
        session = parse_claude_session_file(session_file)
        if session:
            sessions.append(session)
    return sessions


class ClaudeAdapter(SourceAdapter):
    name = "claude"

    def discover(self) -> list[SourceUnit]:
        if not self.available():
            return []
        units = []
        for path in iter_claude_session_files(self.root):
            try:
                token = file_token(path)
                for sub in subagent_files(path):
                    token += "|" + file_token(sub)
            except OSError:
                continue
            units.append(SourceUnit(str(path), self.name, token, path))
        return units

    def parse(self, unit: SourceUnit) -> dict | None:
        return parse_claude_session_file(unit.locator)
