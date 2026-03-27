"""
MCP server — exposes 3 tools to Claude Code (and any MCP-compatible client):
  • search_history   — keyword search across Codex + Claude sessions
  • list_sessions    — list all sessions with titles and dates
  • get_session      — retrieve full content of one session
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from .parsers import parse_codex_sessions, parse_claude_sessions
from .search  import search_sessions, format_hits, format_session

# ── Path resolution ───────────────────────────────────────────────────────────

def _resolve_path(env_var: str, default: Path) -> Path | None:
    raw = os.environ.get(env_var, "")
    p = Path(raw).expanduser() if raw else default
    return p if p.is_dir() else None


CODEX_PATH  = _resolve_path("CODEX_PATH",  Path.home() / ".codex")
CLAUDE_PATH = _resolve_path("CLAUDE_PATH", Path.home() / ".claude")


# ── Session loader (lazy, cached) ─────────────────────────────────────────────

_cache: dict | None = None


def _load_all(sources: list[str]) -> list[dict]:
    """Load and cache all requested sessions."""
    global _cache
    if _cache is None:
        _cache = {}
        if CODEX_PATH:
            _cache["codex"] = parse_codex_sessions(CODEX_PATH)
        if CLAUDE_PATH:
            _cache["claude"] = parse_claude_sessions(CLAUDE_PATH)

    sessions: list[dict] = []
    for src in sources:
        sessions.extend(_cache.get(src, []))
    return sessions


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "agent-history",
    instructions=(
        "Search and retrieve past conversations from Codex CLI and Claude Code. "
        "Use search_history to find relevant context from previous sessions. "
        "Always prefer search_history over get_session to stay token-efficient."
    ),
)


@mcp.tool()
def search_history(
    query: str,
    sources: list[str] = ["codex", "claude"],
    max_results: int = 5,
) -> str:
    """
    Search past AI coding agent conversations for a keyword or topic.

    Returns matching message excerpts with surrounding context.
    sources can include "codex" (OpenAI Codex CLI) and/or "claude" (Claude Code).
    Prefer this over get_session to avoid token bloat.

    Example: search_history("CUDA illegal address", sources=["codex", "claude"])
    """
    valid = [s for s in sources if s in ("codex", "claude")]
    if not valid:
        return "Invalid sources. Use 'codex', 'claude', or both."

    sessions = _load_all(valid)
    if not sessions:
        return "No sessions found. Check that CODEX_PATH / CLAUDE_PATH are correct."

    hits = search_sessions(sessions, query, max_results=max_results)
    return format_hits(hits, query)


@mcp.tool()
def list_sessions(
    source: str = "all",
    limit: int = 50,
) -> str:
    """
    List available past sessions with their titles, dates, and sources.

    source: "all" | "codex" | "claude"
    Returns session IDs needed for get_session.
    """
    sources = ["codex", "claude"] if source == "all" else [source]
    sessions = _load_all(sources)

    if not sessions:
        return "No sessions found."

    lines = [f"{'SOURCE':<8} {'DATE':<12} {'MSGS':>5}  TITLE"]
    lines.append("─" * 70)
    for s in sessions[:limit]:
        n_msgs = len(s["messages"])
        title  = s["session_title"][:48]
        lines.append(f"{s['source']:<8} {s['session_date']:<12} {n_msgs:>5}  {title}")
        lines.append(f"         id: {s['session_id'][:36]}")

    if len(sessions) > limit:
        lines.append(f"\n... and {len(sessions) - limit} more (increase limit to see all)")

    return "\n".join(lines)


@mcp.tool()
def get_session(
    session_id: str,
    source: str,
    max_messages: int = 30,
) -> str:
    """
    Retrieve the full conversation for a specific session.

    session_id: from list_sessions output
    source: "codex" | "claude"
    max_messages: limit messages returned (default 30) to control token usage.

    Prefer search_history for finding context — use this only when you need
    the full conversation flow.
    """
    if source not in ("codex", "claude"):
        return "source must be 'codex' or 'claude'"

    sessions = _load_all([source])
    session  = next((s for s in sessions if s["session_id"] == session_id), None)

    if not session:
        return f"Session '{session_id}' not found in {source} history."

    return format_session(session, max_messages=max_messages)
