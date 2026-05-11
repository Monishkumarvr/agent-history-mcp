"""
MCP server — exposes 3 tools to Claude Code (and any MCP-compatible client):
  • search_history   — keyword search across Codex + Claude sessions
  • list_sessions    — list all sessions with titles and dates
  • get_session      — retrieve full content of one session
"""

import os
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from .parsers import parse_codex_sessions, parse_claude_sessions
from .search  import SearchHit, search_smart, build_fts_index, format_hits, format_session
from .graph import DEFAULT_DB_PATH, GraphHit, HistoryGraphIndex, format_graph_hits
from .skills import format_skill_candidates, suggest_skill_candidates

# ── Path resolution ───────────────────────────────────────────────────────────

def _resolve_path(env_var: str, default: Path) -> Path | None:
    raw = os.environ.get(env_var, "")
    p = Path(raw).expanduser() if raw else default
    return p if p.is_dir() else None


CODEX_PATH  = _resolve_path("CODEX_PATH",  Path.home() / ".codex")
CLAUDE_PATH = _resolve_path("CLAUDE_PATH", Path.home() / ".claude")


# ── Session loader (lazy, cached) ─────────────────────────────────────────────

_cache: dict | None = None
_graph_index: HistoryGraphIndex | None = None


def _get_graph_index() -> HistoryGraphIndex:
    global _graph_index
    if _graph_index is None:
        _graph_index = HistoryGraphIndex()
    return _graph_index


def _refresh_history_index() -> dict[str, int]:
    """
    Refresh the persistent graph index from changed history files.
    If files changed, invalidate the in-memory FTS cache so keyword search sees
    new chats on the next load.
    """
    global _cache
    stats = _get_graph_index().refresh(CODEX_PATH, CLAUDE_PATH)
    stats_dict = stats.as_dict()
    if stats.files_added or stats.files_changed or stats.files_removed:
        _cache = None
    return stats_dict


def _load_all(sources: list[str]) -> tuple[list[dict], sqlite3.Connection | None, dict]:
    """
    Load and cache all sessions, building the FTS index on first call.
    Returns (filtered_sessions, fts_conn, session_map).
    """
    global _cache
    if _cache is None:
        codex_sessions  = parse_codex_sessions(CODEX_PATH)  if CODEX_PATH  else []
        claude_sessions = parse_claude_sessions(CLAUDE_PATH) if CLAUDE_PATH else []
        all_sessions    = codex_sessions + claude_sessions

        _cache = {
            "codex":       codex_sessions,
            "claude":      claude_sessions,
            "conn":        build_fts_index(all_sessions),
            "session_map": {s["session_id"]: s for s in all_sessions},
        }

    sessions: list[dict] = []
    for src in sources:
        sessions.extend(_cache.get(src, []))

    return sessions, _cache["conn"], _cache["session_map"]


def _graph_to_search_hit(hit: GraphHit) -> SearchHit:
    return SearchHit(
        session_id=hit.session_id,
        session_title=hit.session_title,
        session_date=hit.session_date,
        source=hit.source,
        matched_role="graph",
        question_text=hit.question_text,
        question_ts=hit.question_ts,
        answer_text=hit.answer_text,
        answer_ts=hit.answer_ts,
        density_score=0.0,
        graph_score=hit.score,
        matched_topics=hit.matched_topics,
        related_topics=hit.related_topics,
        why=hit.why,
    )


def _merge_keyword_and_graph_hits(
    keyword_hits: list[SearchHit],
    graph_hits: list[GraphHit],
    max_results: int,
) -> list[SearchHit]:
    graph_by_key = {(h.source, h.session_id): h for h in graph_hits}
    scored: dict[tuple[str, str], tuple[float, SearchHit]] = {}

    keyword_count = max(1, len(keyword_hits))
    for idx, hit in enumerate(keyword_hits):
        key = (hit.source, hit.session_id)
        score = (keyword_count - idx) * 10.0 + hit.density_score
        graph_hit = graph_by_key.get(key)
        if graph_hit:
            hit.graph_score = graph_hit.score
            hit.matched_topics = graph_hit.matched_topics
            hit.related_topics = graph_hit.related_topics
            hit.why = graph_hit.why
            score += min(graph_hit.score, 25.0)
        scored[key] = (score, hit)

    max_graph_score = max((h.score for h in graph_hits), default=1.0)
    for graph_hit in graph_hits:
        key = (graph_hit.source, graph_hit.session_id)
        if key in scored:
            continue
        normalized = (graph_hit.score / max_graph_score) * 20.0
        scored[key] = (normalized, _graph_to_search_hit(graph_hit))

    return [
        hit
        for _score, hit in sorted(scored.values(), key=lambda item: item[0], reverse=True)[:max_results]
    ]


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

    _refresh_history_index()
    sessions, conn, session_map = _load_all(valid)
    if not sessions:
        return "No sessions found. Check that CODEX_PATH / CLAUDE_PATH are correct."

    keyword_hits = search_smart(conn, query, sessions, session_map, valid, max_results * 2)
    graph_hits = _get_graph_index().search(query, valid, max_results * 2)
    hits = _merge_keyword_and_graph_hits(keyword_hits, graph_hits, max_results)
    return format_hits(hits, query)


@mcp.tool()
def search_graph(
    query: str,
    sources: list[str] = ["codex", "claude"],
    max_results: int = 5,
) -> str:
    """
    Search the persistent local knowledge graph built from past conversations.

    Use this for relationship-heavy questions where exact keywords may differ
    across sessions, such as "where did I solve a similar auth migration bug?".
    sources can include "codex" and/or "claude".
    """
    valid = [s for s in sources if s in ("codex", "claude")]
    if not valid:
        return "Invalid sources. Use 'codex', 'claude', or both."

    _refresh_history_index()
    hits = _get_graph_index().search(query, valid, max_results)
    return format_graph_hits(hits, query)


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
    _refresh_history_index()
    sessions, _conn, _session_map = _load_all(sources)

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

    _refresh_history_index()
    sessions, _conn, _session_map = _load_all([source])
    session  = next((s for s in sessions if s["session_id"] == session_id), None)

    if not session:
        return f"Session '{session_id}' not found in {source} history."

    return format_session(session, max_messages=max_messages)


@mcp.tool()
def refresh_history_index(rebuild: bool = False) -> str:
    """
    Manually refresh the persistent graph index and in-memory keyword cache.

    rebuild=true deletes and recreates the derived graph database. It never
    modifies Codex or Claude history files.
    """
    global _cache, _graph_index
    if rebuild:
        if _graph_index is not None:
            db_path = _graph_index.db_path
            _graph_index.close()
            _graph_index = None
        else:
            db_path = Path(os.environ.get("AGENT_HISTORY_GRAPH_DB") or DEFAULT_DB_PATH).expanduser()
        if db_path.exists():
            db_path.unlink()
        _cache = None

    stats = _refresh_history_index()
    db_path = _get_graph_index().db_path
    return (
        f"History graph index refreshed at {db_path}\n"
        f"files_seen     : {stats['files_seen']}\n"
        f"files_added    : {stats['files_added']}\n"
        f"files_changed  : {stats['files_changed']}\n"
        f"files_removed  : {stats['files_removed']}\n"
        f"files_unchanged: {stats['files_unchanged']}\n"
        f"sessions_indexed: {stats['sessions_indexed']}"
    )


@mcp.tool()
def suggest_skills(
    sources: list[str] = ["codex", "claude"],
    max_candidates: int = 5,
    min_sessions: int = 2,
    days_back: int | None = None,
) -> str:
    """
    Suggest reusable skills that could be created from repeated chat patterns.

    Returns ranked skill ideas with trigger phrases, supporting sessions, and
    boundaries. This never writes SKILL.md files and never calls a model.
    """
    valid = [s for s in sources if s in ("codex", "claude")]
    if not valid:
        return "Invalid sources. Use 'codex', 'claude', or both."

    _refresh_history_index()
    candidates = suggest_skill_candidates(
        _get_graph_index(),
        sources=valid,
        max_candidates=max_candidates,
        min_sessions=min_sessions,
        days_back=days_back,
    )
    return format_skill_candidates(candidates)
