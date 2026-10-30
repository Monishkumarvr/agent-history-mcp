"""
Keep what context compaction drops.

Compaction swaps a long conversation for a short summary; on real history it kept
a median 12% of tokens (Claude) and 23% (Codex). The transcript on disk still has
everything, so nothing has to be lost:

  - the PreCompact hook writes a checkpoint built from the transcript (no model call),
  - the SessionStart hook (source "compact") puts that checkpoint back into context,
  - session_recall searches the transcript for the exact text the summary dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

from .handoff import _clip, _format_digest, _is_prompt_text, digest_session
from .model import agent_home
from .redact import redact
from .sources.claude import parse_claude_session_file
from .sources.codex import parse_codex_session_file

RECALL_SOURCES = ("claude", "codex")
_ASK_COUNT = 15        # user messages kept in a checkpoint
_ASK_CHARS = 600       # per message; the full text stays reachable through session_recall
_WINDOW = 300          # chars of context on each side of a recall match
_MESSAGE_CHARS = 20000
_SAFE_ID_RE = re.compile(r"^[\w.-]+$")


def find_transcript(session_id: str, source: str, root: Path) -> Path | None:
    """Transcript file for a session id (a unique prefix works for Claude)."""
    session_id = (session_id or "").strip()
    if not _SAFE_ID_RE.match(session_id):  # the id goes into a glob pattern
        return None
    if source == "claude":
        matches = list(root.glob(f"projects/*/{session_id}*.jsonl"))
    elif source == "codex":
        matches = [p for d in ("sessions", "archived_sessions") for p in (root / d).rglob(f"*{session_id}*.jsonl")]
    else:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def load_session(path: Path, source: str) -> dict | None:
    # Subagent transcripts are skipped: compaction only drops the main conversation.
    if source == "claude":
        return parse_claude_session_file(path, include_subagents=False)
    return parse_codex_session_file(path)


def _user_asks(session: dict) -> list[tuple[int, str]]:
    return [
        (i, m["text"]) for i, m in enumerate(session["messages"])
        if m["role"] == "user" and _is_prompt_text(m["text"])
    ]


def build_checkpoint(session: dict, session_id: str, source: str, trigger: str = "") -> str:
    events = session["events"]
    row = {
        "source": source, "session_id": session_id, "title": session["session_title"],
        "started_at": session["started_at"], "ended_at": session["ended_at"],
        "git_branch": session["git_branch"],
    }
    digest = digest_session(row, events)
    asks = _user_asks(session)
    earlier = sum(1 for e in events if e.meta.get("signal") == "compact")
    lines = [
        f"# Checkpoint saved before context compaction{f' ({trigger})' if trigger else ''}",
        f"Session `{session_id}` ({source}), {len(asks)} user messages, {earlier} earlier compaction(s).",
        "",
    ]
    if asks:
        shown = asks[-_ASK_COUNT:]
        lines.append(f"## User messages, last {len(shown)} of {len(asks)} (long ones clipped; [mN] = message number)")
        lines.extend(f"- [m{i}] {_clip(text, _ASK_CHARS)}" for i, text in shown)
        lines.append("")
    lines.extend(_format_digest(digest, compact=True, root=session.get("project") or ""))
    lines.append(
        "The summary above may have dropped exact wording, code, errors and decisions. They are still on disk: "
        f'call session_recall(session_id="{session_id}", source="{source}", query="...") to search them, '
        "or pass message=N to read message [mN] in full."
    )
    return redact("\n".join(lines))


def checkpoint_path(source: str, session_id: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", session_id)
    return agent_home() / "compactions" / f"{source}-{safe}.md"


def write_checkpoint(transcript: Path, session_id: str, source: str, trigger: str = "") -> Path | None:
    session = load_session(transcript, source)
    if not session:
        return None
    path = checkpoint_path(source, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_checkpoint(session, session_id, source, trigger), encoding="utf-8")
    return path


def read_checkpoint(source: str, session_id: str) -> str:
    path = checkpoint_path(source, session_id)
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _snippet(text: str, terms: list[str]) -> str:
    lowered = text.lower()
    at = min((lowered.find(t) for t in terms if t in lowered), default=0)
    start = max(0, at - _WINDOW)
    body = " ".join(text[start: at + _WINDOW].split())
    return ("…" if start else "") + body + ("…" if at + _WINDOW < len(text) else "")


def recall(session: dict, query: str = "", message: int | None = None, max_hits: int = 5) -> str:
    """Search one session's full transcript, or return one message in full."""
    messages = session["messages"]
    if message is not None:
        if not 0 <= message < len(messages):
            return f"message must be between 0 and {len(messages) - 1}."
        m = messages[message]
        text = m["text"]
        more = f"\n…[{len(text) - _MESSAGE_CHARS} more chars]" if len(text) > _MESSAGE_CHARS else ""
        return f"[m{message}] {m['role']} {m['ts']}\n{text[:_MESSAGE_CHARS]}{more}"

    terms = [t for t in re.findall(r"\w[\w.-]*", (query or "").lower()) if len(t) > 1]
    if not terms:
        return "Pass a query, or message=N to read one message in full."
    # Messages hold full text; tool events hold commands, paths and the first 500 chars of output.
    candidates = [(f"m{i}", f"{m['role']} {m['ts']}", m["text"]) for i, m in enumerate(messages)]
    candidates += [
        (f"e{e.idx}", f"{e.kind} {e.tool} {e.ts[:16]}", " ".join(filter(None, (e.command, " ".join(e.paths), e.text))))
        for e in session["events"] if e.kind in ("tool_call", "tool_result")
    ]
    scored = []
    for order, (ref, label, text) in enumerate(candidates):
        lowered = text.lower()
        score = sum(1 for t in terms if t in lowered)
        if score:
            scored.append((score, order, ref, label, text))
    if not scored:
        return f"No match for {query!r} in session {session['session_id']}."
    # ponytail: substring term counting, no ranking model; add BM25 if recall quality is poor.
    scored.sort(key=lambda s: (-s[0], -s[1]))
    lines = [f"{len(scored)} match(es) for {query!r}; showing {min(max_hits, len(scored))}. Read one message in full with message=N."]
    for score, _order, ref, label, text in scored[:max(1, max_hits)]:
        lines.append(f"\n[{ref}] {label} ({score}/{len(terms)} terms)\n{_snippet(text, terms)}")
    return "\n".join(lines)
