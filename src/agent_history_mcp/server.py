"""
MCP server for searching, bridging, and analyzing coding-agent history across
Claude Code, Codex CLI, Gemini CLI, OpenCode, and Cursor.
"""

import functools
import os
import sqlite3
from pathlib import Path

try:  # MCP Python SDK 2.x renamed FastMCP to MCPServer; keep 1.x working too.
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from . import handoff, memory_sync
from .graph import DEFAULT_DB_PATH, GraphHit, HistoryGraphIndex, format_graph_hits
from .redact import redact
from .search import SearchHit, build_fts_index, format_hits, format_session, search_smart
from .skills import format_skill_candidates, suggest_skill_candidates
from .sources import SOURCE_NAMES, SourceAdapter, default_adapters, resolve_sources

SOURCES_HELP = f"sources: 'all' or any of {', '.join(SOURCE_NAMES)}"


# Index and caches (lazy)

_cache: dict | None = None
_graph_index: HistoryGraphIndex | None = None
_adapters: list[SourceAdapter] | None = None


def _get_adapters() -> list[SourceAdapter]:
    global _adapters
    if _adapters is None:
        _adapters = default_adapters()
    return _adapters


def _get_graph_index() -> HistoryGraphIndex:
    global _graph_index
    if _graph_index is None:
        _graph_index = HistoryGraphIndex()
    return _graph_index


def _refresh_history_index() -> dict:
    """
    Refresh the persistent graph index from changed history units.
    If anything changed, invalidate the in-memory FTS cache so keyword search
    sees new chats on the next load.
    """
    global _cache
    stats = _get_graph_index().refresh(adapters=_get_adapters())
    if stats.files_added or stats.files_changed or stats.files_removed:
        _cache = None
    return {**stats.as_dict(), "per_source": stats.per_source, "diagnostics": stats.diagnostics}


def _load_all(sources: list[str]) -> tuple[list[dict], sqlite3.Connection | None, dict]:
    """
    Load sessions from the graph index and build the FTS index on first call.
    Returns (filtered_sessions, fts_conn, session_map).
    """
    global _cache
    if _cache is None:
        all_sessions = _get_graph_index().load_sessions()
        by_source: dict[str, list[dict]] = {}
        for session in all_sessions:
            by_source.setdefault(session["source"], []).append(session)
        _cache = {
            "by_source": by_source,
            "conn": build_fts_index(all_sessions),
            "session_map": {s["session_id"]: s for s in all_sessions},
        }

    sessions: list[dict] = []
    for src in sources:
        sessions.extend(_cache["by_source"].get(src, []))
    return sessions, _cache["conn"], _cache["session_map"]


def _project_keys(sources: list[str], project: str | None) -> set[tuple[str, str]] | None:
    if not project:
        return None
    rows = _get_graph_index().session_rows(sources, handoff.resolve_project(project))
    return {(row["source"], row["session_id"]) for row in rows}


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


def _redacted(fn):
    """Every tool output passes through secret redaction."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return redact(fn(*args, **kwargs))
    return wrapper


# MCP server

mcp = FastMCP(
    "agent-history",
    instructions=(
        "Local memory and context bridge across Claude Code, Codex CLI, Gemini CLI, OpenCode and Cursor. "
        "Use search_history to find relevant context from previous sessions in any agent. "
        "Use handoff_brief at the start of work in a repository to see where the last agent stopped. "
        "Prefer search_history over get_session to stay token-efficient. "
        "Tools that write files (install_skill, sync_memory, save_checkpoint with apply) only write when apply=true."
    ),
)


@mcp.tool()
@_redacted
def search_history(
    query: str,
    sources: list[str] | str = "all",
    max_results: int = 5,
    project: str | None = None,
) -> str:
    """
    Search past AI coding agent conversations for a keyword or topic.

    Returns matching message excerpts with surrounding context.
    sources: "all" or any of claude, codex, gemini, opencode, cursor.
    project: optional repository path to limit results to sessions run there.
    Prefer this over get_session to avoid token bloat.

    Example: search_history("CUDA illegal address", sources=["codex", "claude"])
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."

    _refresh_history_index()
    sessions, conn, session_map = _load_all(valid)
    if not sessions:
        return "No sessions found. Check the history paths (CLAUDE_PATH, CODEX_PATH, GEMINI_PATH, OPENCODE_DB, CURSOR_PATH)."

    allowed = _project_keys(valid, project)
    fetch = max_results * (6 if allowed is not None else 2)
    keyword_hits = search_smart(conn, query, sessions, session_map, valid, fetch)
    graph_hits = _get_graph_index().search(query, valid, fetch)
    if allowed is not None:
        keyword_hits = [h for h in keyword_hits if (h.source, h.session_id) in allowed]
        graph_hits = [h for h in graph_hits if (h.source, h.session_id) in allowed]
    hits = _merge_keyword_and_graph_hits(keyword_hits, graph_hits, max_results)
    return format_hits(hits, query)


@mcp.tool()
@_redacted
def search_graph(
    query: str,
    sources: list[str] | str = "all",
    max_results: int = 5,
) -> str:
    """
    Search the persistent local knowledge graph built from past conversations.

    Use this for relationship-heavy questions where exact keywords may differ
    across sessions, such as "where did I solve a similar auth migration bug?".
    sources: "all" or any of claude, codex, gemini, opencode, cursor.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."

    _refresh_history_index()
    hits = _get_graph_index().search(query, valid, max_results)
    return format_graph_hits(hits, query)


@mcp.tool()
@_redacted
def list_sessions(
    source: str = "all",
    limit: int = 50,
    project: str | None = None,
) -> str:
    """
    List available past sessions with their titles, dates, sources and projects.

    source: "all" or one of claude, codex, gemini, opencode, cursor.
    project: optional repository path filter.
    Returns session IDs needed for get_session.
    """
    sources = resolve_sources(source)
    if not sources:
        return f"Invalid source. Use {SOURCES_HELP}."
    _refresh_history_index()
    root = handoff.resolve_project(project) if project else None
    rows = _get_graph_index().session_rows(sources, root, order="session_date DESC, ended_at DESC")

    if not rows:
        return "No sessions found."

    lines = [f"{'SOURCE':<9} {'DATE':<12} {'MSGS':>5} {'TOOLS':>5}  TITLE"]
    lines.append("-" * 78)
    for row in rows[:limit]:
        title = row["title"][:46]
        lines.append(f"{row['source']:<9} {row['session_date']:<12} {row['n_messages']:>5} {row['n_tool_calls']:>5}  {title}")
        where = f"  project: {row['project']}" if row["project"] else ""
        lines.append(f"          id: {row['session_id'][:36]}{where}")

    if len(rows) > limit:
        lines.append(f"\n... and {len(rows) - limit} more (increase limit to see all)")

    return "\n".join(lines)


@mcp.tool()
@_redacted
def get_session(
    session_id: str,
    source: str,
    max_messages: int = 30,
    include_tools: bool = False,
) -> str:
    """
    Retrieve the conversation for a specific session.

    session_id: from list_sessions output (a unique prefix works)
    source: claude | codex | gemini | opencode | cursor
    max_messages: limit messages returned (default 30) to control token usage.
    include_tools: interleave tool calls/results (commands, edits, errors).

    Prefer search_history for finding context; use this only when you need
    the full conversation flow.
    """
    if source not in SOURCE_NAMES:
        return f"source must be one of {', '.join(SOURCE_NAMES)}"

    _refresh_history_index()
    index = _get_graph_index()
    row = index.find_session(session_id, source)
    if row is None:
        return f"Session '{session_id}' not found in {source} history."

    if include_tools:
        return format_session(dict(row), max_messages=max_messages, events=index.events_for(row["session_key"]))
    sessions, _conn, _session_map = _load_all([source])
    session = next((s for s in sessions if s["session_id"] == row["session_id"]), None)
    if not session:
        return f"Session '{session_id}' has no messages in {source} history."
    return format_session(session, max_messages=max_messages)


@mcp.tool()
@_redacted
def refresh_history_index(rebuild: bool = False) -> str:
    """
    Manually refresh the persistent graph index and in-memory keyword cache.

    rebuild=true deletes and recreates the derived graph database. It never
    modifies any agent's history files.
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
    counts = _get_graph_index().conn.execute(
        "SELECT source, COUNT(*) AS n FROM sessions GROUP BY source ORDER BY source"
    ).fetchall()
    lines = [
        f"History graph index refreshed at {db_path}",
        f"files_seen     : {stats['files_seen']}",
        f"files_added    : {stats['files_added']}",
        f"files_changed  : {stats['files_changed']}",
        f"files_removed  : {stats['files_removed']}",
        f"files_unchanged: {stats['files_unchanged']}",
        f"sessions_indexed: {stats['sessions_indexed']}",
        "indexed by source: " + (", ".join(f"{r['source']}={r['n']}" for r in counts) or "none"),
    ]
    if stats["diagnostics"]:
        lines.append("notes:")
        lines.extend(f"  - {d}" for d in stats["diagnostics"])
    return "\n".join(lines)


@mcp.tool()
@_redacted
def suggest_skills(
    sources: list[str] | str = "all",
    max_candidates: int = 5,
    min_sessions: int = 2,
    days_back: int | None = None,
) -> str:
    """
    Suggest reusable skills that could be created from repeated chat patterns.

    Returns ranked skill ideas with trigger phrases, supporting sessions, and
    boundaries. This never writes SKILL.md files and never calls a model.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."

    _refresh_history_index()
    candidates = suggest_skill_candidates(
        _get_graph_index(),
        sources=valid,
        max_candidates=max_candidates,
        min_sessions=min_sessions,
        days_back=days_back,
    )
    return format_skill_candidates(candidates)


# Context bridge

@mcp.tool()
@_redacted
def handoff_brief(
    project: str | None = None,
    sources: list[str] | str = "all",
    sessions: int = 2,
    exclude_source: str | None = None,
) -> str:
    """
    Summarize where the latest session(s) in a repository stopped, across agents.

    Includes the goal, open todos, files touched, recent commands with outcomes,
    unresolved errors, dead ends (commands that failed repeatedly), the latest
    checkpoint note, and live git status. Call this when starting work in a repo
    that another agent (or an earlier session) worked on.

    project: repository path (defaults to the server's working directory).
    exclude_source: skip one agent, e.g. "claude" to see only what others did.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    _refresh_history_index()
    return handoff.handoff_brief(_get_graph_index(), project, valid, sessions, exclude_source)


@mcp.tool()
@_redacted
def save_checkpoint(note: str, project: str | None = None, apply: bool = False) -> str:
    """
    Save a handoff checkpoint (your note + the current brief) for this repository.

    The checkpoint is stored under ~/.agent-history-mcp/checkpoints and shown by
    handoff_brief to any agent. apply=true also writes .agent-context/HANDOFF.md
    inside the repository.
    """
    _refresh_history_index()
    return handoff.save_checkpoint(_get_graph_index(), note, project, apply=apply)


@mcp.tool()
@_redacted
def list_agent_memory(project: str | None = None) -> str:
    """
    Show which instruction and memory files each agent reads for a repository,
    and the drift between them (rules one agent sees but another doesn't).

    Covers CLAUDE.md, Claude auto-memory, AGENTS.md (Codex/Cursor/OpenCode),
    GEMINI.md, .cursor/rules, and Codex memories at project and user level.
    """
    _refresh_history_index()
    return memory_sync.list_agent_memory(_get_graph_index(), handoff.resolve_project(project))


@mcp.tool()
@_redacted
def suggest_rules(
    project: str | None = None,
    sources: list[str] | str = "all",
    days_back: int | None = 90,
    min_sessions: int = 2,
) -> str:
    """
    Mine your repeated corrections ("no, use pnpm"), stated preferences
    ("always run ruff first") and rejected tool calls into candidate rules for
    CLAUDE.md / AGENTS.md / GEMINI.md, with evidence and whether each rule is
    already documented. Pass project="" to search all projects.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    _refresh_history_index()
    root = handoff.resolve_project(project) if project != "" else None
    candidates = memory_sync.suggest_rules(_get_graph_index(), root, valid, days_back, min_sessions)
    return memory_sync.format_rule_candidates(candidates)


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=True))
@_redacted
def sync_memory(
    rules: list[str] | None = None,
    targets: list[str] | None = None,
    project: str | None = None,
    scope: str = "project",
    apply: bool = False,
) -> str:
    """
    Add shared rules to every agent's instruction file via a managed block.

    rules: rules to add; omit to sync the drift (rules some agents are missing).
    targets: agents to update (claude -> CLAUDE.md, codex/cursor/opencode ->
    AGENTS.md, gemini -> GEMINI.md); defaults to agents that already have one.
    scope: "project" (repo files) or "user" (~/.claude, ~/.codex, ~/.gemini).
    Returns a unified diff. Writes only when apply=true, and only edits the
    <!-- agent-history:begin/end --> block.
    """
    return memory_sync.sync_memory(handoff.resolve_project(project), rules, targets, scope, apply)


def run() -> None:
    mcp.run()
