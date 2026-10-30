"""
Codex CLI rollouts: ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl

Each line is {"timestamp", "type", "payload"} where type is session_meta,
turn_context, response_item, event_msg, or compacted. response_item payloads
are tagged by their own "type": message, function_call, function_call_output,
custom_tool_call(_output), local_shell_call, web_search_call, reasoning, ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..model import SessionBuilder, output_text
from . import SourceAdapter, SourceUnit, file_token, read_jsonl

_CODEX_SKIP_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
    "<user_instructions>",
    "# AGENTS.md",
    "# Context from my IDE",
)


def _codex_extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if not isinstance(c, dict):
            continue
        for key in ("text", "input_text", "output_text"):
            v = c.get(key, "")
            if v:
                parts.append(v)
                break
    return "\n".join(parts)


def _codex_clean_user(text: str) -> str:
    m = re.search(r"## My request for Codex:\s*\n(.+)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def iter_codex_session_files(codex_path: Path) -> list[Path]:
    files: list[Path] = []
    for dirname in ("sessions", "archived_sessions"):
        sessions_dir = codex_path / dirname
        if sessions_dir.is_dir():
            files.extend(sessions_dir.rglob("*.jsonl"))
    return sorted(files, reverse=True)


def load_codex_title_index(codex_path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    index_file = codex_path / "session_index.jsonl"
    if index_file.exists():
        for obj in read_jsonl(index_file):
            sid = obj.get("id", "")
            name = obj.get("thread_name", "")
            if sid and name:
                index[sid] = name
    return index


def _date_from_path(path: Path) -> str:
    parts = list(path.parts)
    try:
        yi = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4)
        return f"{parts[yi]}-{parts[yi + 1]}-{parts[yi + 2]}"
    except (StopIteration, IndexError):
        return ""


def _unwrap_output(text: str) -> tuple[str, int | None]:
    """Tool outputs are often JSON {"output": ..., "metadata": {"exit_code": n}}."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text, None
    if not isinstance(data, dict):
        return text, None
    exit_code = None
    meta = data.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("exit_code"), int):
        exit_code = meta["exit_code"]
    elif isinstance(data.get("exit_code"), int):
        exit_code = data["exit_code"]
    output = data.get("output")
    return (output if isinstance(output, str) else text), exit_code


def parse_codex_session_file(path: Path, title_index: dict[str, str] | None = None) -> dict | None:
    fname = path.stem
    index = title_index or {}
    builder = SessionBuilder("codex", fname, str(path))
    builder.title = next((n for sid, n in index.items() if sid in fname), fname[:60])
    builder.session_date = _date_from_path(path)
    exit_codes: dict[str, int] = {}

    for obj in read_jsonl(path):
        record_type = obj.get("type")
        payload = obj.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        ts = obj.get("timestamp")

        if record_type == "session_meta":
            builder.set_cwd(payload.get("cwd"))
            git = payload.get("git") or {}
            if isinstance(git, dict):
                builder.set_branch(git.get("branch"))
            continue
        if record_type == "turn_context":
            builder.set_cwd(payload.get("cwd"))
            builder.add_model(payload.get("model"))
            continue
        if record_type == "compacted":
            builder.add_signal("compact", "context compacted", ts)
            continue
        if record_type == "event_msg":
            event_type = payload.get("type")
            if event_type == "token_count":
                info = payload.get("info") or {}
                total = info.get("total_token_usage") if isinstance(info, dict) else None
                if isinstance(total, dict):
                    builder.set_tokens(total.get("input_tokens"), total.get("output_tokens"))
            elif event_type == "exec_command_end":
                call_id = payload.get("call_id", "")
                if isinstance(payload.get("exit_code"), int):
                    exit_codes[call_id] = payload["exit_code"]
                    builder.set_exit_code(call_id, payload["exit_code"])
            elif event_type == "turn_aborted":
                builder.add_signal("interrupt", str(payload.get("reason", "")), ts)
            continue
        if record_type != "response_item":
            continue

        item_type = payload.get("type")
        if item_type in (None, "message"):
            role = payload.get("role", "")
            if role not in ("user", "assistant"):
                continue
            text = _codex_extract_text(payload.get("content", []))
            if not text or any(text.strip().startswith(pfx) for pfx in _CODEX_SKIP_PREFIXES):
                continue
            if role == "user":
                text = _codex_clean_user(text)
            builder.add_message(role, text, ts)
        elif item_type == "function_call":
            builder.add_tool_call(payload.get("name", ""), payload.get("arguments"), ts,
                                  call_id=payload.get("call_id", ""), namespace=payload.get("namespace"))
        elif item_type == "custom_tool_call":
            builder.add_tool_call(payload.get("name", ""), payload.get("input"), ts,
                                  call_id=payload.get("call_id", ""), namespace=payload.get("namespace"))
        elif item_type == "local_shell_call":
            action = payload.get("action") or {}
            builder.add_tool_call("local_shell", {"command": action.get("command")}, ts,
                                  call_id=payload.get("call_id") or "")
        elif item_type == "web_search_call":
            builder.add_tool_call("web_search_call", payload.get("action") or {}, ts)
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = payload.get("call_id") or ""
            text, exit_code = _unwrap_output(output_text(payload.get("output")))
            if exit_code is None:
                exit_code = exit_codes.get(call_id)
            builder.add_tool_result(call_id, text, ts, exit_code=exit_code, name=payload.get("name") or "")

    return builder.build()


def parse_codex_sessions(codex_path: Path) -> list[dict]:
    """
    Return a list of Codex session dicts.
    """
    index = load_codex_title_index(codex_path)
    sessions = []
    for path in iter_codex_session_files(codex_path):
        session = parse_codex_session_file(path, index)
        if session:
            sessions.append(session)
    return sessions


class CodexAdapter(SourceAdapter):
    name = "codex"

    def __init__(self, root):
        super().__init__(root)
        self._title_index: dict[str, str] | None = None

    def discover(self) -> list[SourceUnit]:
        if not self.available():
            return []
        self._title_index = load_codex_title_index(self.root)
        units = []
        for path in iter_codex_session_files(self.root):
            try:
                units.append(SourceUnit(str(path), self.name, file_token(path), path))
            except OSError:
                continue
        return units

    def parse(self, unit: SourceUnit) -> dict | None:
        if self._title_index is None:
            self._title_index = load_codex_title_index(self.root) if self.root else {}
        return parse_codex_session_file(unit.locator, self._title_index)
