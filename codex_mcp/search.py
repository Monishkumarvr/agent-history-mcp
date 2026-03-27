"""
Keyword search across unified session message lists.
No embeddings — simple case-insensitive substring match with context window.
Fast enough for local JSONL files (typically < 500MB total).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class SearchHit:
    session_id: str
    session_title: str
    session_date: str
    source: str          # "codex" | "claude"
    matched_role: str
    matched_text: str    # the matching message (truncated)
    context: list[dict]  # up to 2 messages before + after for context


def search_sessions(
    sessions: list[dict],
    query: str,
    max_results: int = 5,
    context_window: int = 2,
) -> list[SearchHit]:
    """
    Search all sessions for messages containing `query`.
    Returns at most one hit per session (the best/first match),
    up to max_results total hits.
    """
    query_lower = query.lower()
    hits: list[SearchHit] = []
    seen_sessions: set[str] = set()

    for session in sessions:
        if session["session_id"] in seen_sessions:
            continue

        msgs = session["messages"]
        for i, msg in enumerate(msgs):
            if query_lower not in msg["text"].lower():
                continue

            # found a match — grab surrounding context
            start = max(0, i - context_window)
            end   = min(len(msgs), i + context_window + 1)
            ctx   = msgs[start:end]

            hits.append(SearchHit(
                session_id    = session["session_id"],
                session_title = session["session_title"],
                session_date  = session["session_date"],
                source        = session["source"],
                matched_role  = msg["role"],
                matched_text  = msg["text"][:1500],
                context       = [
                    {"role": m["role"], "text": m["text"][:400], "ts": m["ts"]}
                    for m in ctx
                ],
            ))
            seen_sessions.add(session["session_id"])
            break   # one hit per session

        if len(hits) >= max_results:
            break

    return hits


def format_hits(hits: list[SearchHit], query: str) -> str:
    """Format search hits as readable text for Claude's context."""
    if not hits:
        return f'No results found for "{query}".'

    lines = [f'Found {len(hits)} session(s) matching "{query}":\n']

    for i, hit in enumerate(hits, 1):
        lines.append(
            f"── Result {i} ──────────────────────────────────────────\n"
            f"Source : {hit.source.upper()}\n"
            f"Session: {hit.session_title}\n"
            f"Date   : {hit.session_date}\n"
            f"ID     : {hit.session_id[:36]}\n"
        )
        lines.append("Context:")
        for msg in hit.context:
            role_label = "YOU  " if msg["role"] == "user" else "AGENT"
            marker = " ◄ match" if query.lower() in msg["text"].lower() else ""
            lines.append(f"  [{role_label}] {msg['ts']}  {msg['text'][:300]}{marker}")
        lines.append("")

    return "\n".join(lines)


def format_session(session: dict, max_messages: int = 30) -> str:
    """Format a full session as readable text."""
    msgs = session["messages"][:max_messages]
    truncated = len(session["messages"]) > max_messages

    lines = [
        f"Source : {session['source'].upper()}",
        f"Session: {session['session_title']}",
        f"Date   : {session['session_date']}",
        f"ID     : {session['session_id'][:36]}",
        f"Messages: {len(session['messages'])}"
        + (f" (showing first {max_messages})" if truncated else ""),
        "─" * 60,
    ]

    for msg in msgs:
        role_label = "YOU  " if msg["role"] == "user" else "AGENT"
        lines.append(f"\n[{role_label}] {msg['ts']}")
        lines.append(msg["text"][:2000])

    return "\n".join(lines)
