"""
Parsers for Codex CLI and Claude Code JSONL session files.
Both parsers emit the same unified message dict:
  {role, text, ts, session_id, session_title, session_date, source}
"""

import json
import re
from datetime import datetime
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

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
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out
    except Exception:
        return []


# ── Codex parser ──────────────────────────────────────────────────────────────

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


def parse_codex_sessions(codex_path: Path) -> list[dict]:
    """
    Return a list of session dicts, each containing:
      {session_id, session_title, session_date, source, messages: [...]}
    """
    sessions_dir = codex_path / "sessions"
    if not sessions_dir.is_dir():
        return []

    # load title index
    index: dict[str, str] = {}
    index_file = codex_path / "session_index.jsonl"
    if index_file.exists():
        for obj in _read_jsonl(index_file):
            sid = obj.get("id", "")
            name = obj.get("thread_name", "")
            if sid and name:
                index[sid] = name

    sessions = []
    for path in sorted(sessions_dir.rglob("*.jsonl"), reverse=True):
        fname = path.stem

        # date from directory structure YYYY/MM/DD
        try:
            parts = list(path.parts)
            yi = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4)
            date_str = f"{parts[yi]}-{parts[yi+1]}-{parts[yi+2]}"
        except (StopIteration, IndexError):
            date_str = "unknown"

        title = next((n for sid, n in index.items() if sid in fname), fname[:60])

        messages = []
        for obj in _read_jsonl(path):
            if obj.get("type") != "response_item":
                continue
            p = obj.get("payload", {})
            role = p.get("role", "")
            if role not in ("user", "assistant"):
                continue
            text = _codex_extract_text(p.get("content", []))
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

        if messages:
            sessions.append({
                "session_id": fname,
                "session_title": title,
                "session_date": date_str,
                "source": "codex",
                "messages": messages,
            })

    return sessions


# ── Claude Code parser ────────────────────────────────────────────────────────

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


def parse_claude_sessions(claude_path: Path) -> list[dict]:
    """
    Walk ~/.claude/projects/ and parse each top-level session JSONL.
    Skips subagent files (they have no user messages).
    """
    projects_dir = claude_path / "projects"
    if not projects_dir.is_dir():
        return []

    sessions = []

    # each project is a flat dir; sessions are top-level *.jsonl files
    for session_file in sorted(projects_dir.rglob("*.jsonl"), reverse=True):
        # skip subagent files — they live in .../subagents/*.jsonl
        if "subagents" in session_file.parts:
            continue

        session_id = session_file.stem
        objects = _read_jsonl(session_file)

        # extract title and date
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
            t = obj.get("type", "")
            if t not in ("user", "assistant"):
                continue
            msg = obj.get("message", {})
            role = msg.get("role", t)
            text = _claude_extract_text(msg.get("content", []))
            if not text:
                continue
            # strip system context injections (tool results, file contents, etc.)
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

        if messages:
            sessions.append({
                "session_id": session_id,
                "session_title": title,
                "session_date": date_str,
                "source": "claude",
                "messages": messages,
            })

    return sessions
