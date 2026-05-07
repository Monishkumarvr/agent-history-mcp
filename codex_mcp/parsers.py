"""
Parsers for Codex CLI and Claude Code JSONL session files.

Both parsers emit the same unified session shape:
  {session_id, session_title, session_date, source, messages: [...]}
where each message has:
  {role, text, ts, session_id, session_title, session_date, source}
"""

import json
import re
from datetime import datetime
from pathlib import Path


def _fmt_ts(ts_raw: str) -> str:
    if not ts_raw:
        return ""
    try:
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_raw[:16]


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(errors="replace").splitlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out
    except Exception:
        return []


_CODEX_SKIP_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
    "# AGENTS.md",
    "# Context from my IDE",
)


def _codex_extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
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
    sessions_dir = codex_path / "sessions"
    if not sessions_dir.is_dir():
        return []
    return sorted(sessions_dir.rglob("*.jsonl"), reverse=True)


def load_codex_title_index(codex_path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    index_file = codex_path / "session_index.jsonl"
    if index_file.exists():
        for obj in _read_jsonl(index_file):
            sid = obj.get("id", "")
            name = obj.get("thread_name", "")
            if sid and name:
                index[sid] = name
    return index


def parse_codex_session_file(path: Path, title_index: dict[str, str] | None = None) -> dict | None:
    fname = path.stem

    try:
        parts = list(path.parts)
        yi = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4)
        date_str = f"{parts[yi]}-{parts[yi + 1]}-{parts[yi + 2]}"
    except (StopIteration, IndexError):
        date_str = "unknown"

    index = title_index or {}
    title = next((n for sid, n in index.items() if sid in fname), fname[:60])

    messages = []
    for obj in _read_jsonl(path):
        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload", {})
        role = payload.get("role", "")
        if role not in ("user", "assistant"):
            continue
        text = _codex_extract_text(payload.get("content", []))
        if not text:
            continue
        if any(text.strip().startswith(pfx) for pfx in _CODEX_SKIP_PREFIXES):
            continue
        if role == "user":
            text = _codex_clean_user(text)
        if text:
            messages.append({
                "role": role,
                "text": text,
                "ts": _fmt_ts(obj.get("timestamp", "")),
                "session_id": fname,
                "session_title": title,
                "session_date": date_str,
                "source": "codex",
            })

    if not messages:
        return None

    return {
        "session_id": fname,
        "session_title": title,
        "session_date": date_str,
        "source": "codex",
        "messages": messages,
    }


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


def _claude_extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "text":
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


def parse_claude_session_file(session_file: Path) -> dict | None:
    session_id = session_file.stem
    objects = _read_jsonl(session_file)

    title = session_id[:20]
    date_str = ""
    first_ts = ""

    for obj in objects:
        if obj.get("type") == "ai-title":
            title = obj.get("aiTitle", title)
        if not first_ts and obj.get("timestamp"):
            first_ts = obj["timestamp"]

    if first_ts:
        try:
            dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            date_str = dt.astimezone().strftime("%Y-%m-%d")
        except Exception:
            date_str = first_ts[:10]

    messages = []
    for obj in objects:
        msg_type = obj.get("type", "")
        if msg_type not in ("user", "assistant"):
            continue
        msg = obj.get("message", {})
        role = msg.get("role", msg_type)
        text = _claude_extract_text(msg.get("content", []))
        if not text:
            continue
        if text.strip().startswith("<") and len(text) > 2000:
            continue
        messages.append({
            "role": role,
            "text": text,
            "ts": _fmt_ts(obj.get("timestamp", "")),
            "session_id": session_id,
            "session_title": title,
            "session_date": date_str,
            "source": "claude",
        })

    if not messages:
        return None

    return {
        "session_id": session_id,
        "session_title": title,
        "session_date": date_str,
        "source": "claude",
        "messages": messages,
    }


def parse_claude_sessions(claude_path: Path) -> list[dict]:
    """
    Walk ~/.claude/projects/ and parse each top-level session JSONL.
    Skips subagent files.
    """
    sessions = []
    for session_file in iter_claude_session_files(claude_path):
        session = parse_claude_session_file(session_file)
        if session:
            sessions.append(session)
    return sessions
