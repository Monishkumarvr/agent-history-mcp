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

from . import analysis, analytics, compaction, handoff, memory_sync, skill_store, skills
from .model import agent_home
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
        "After context compaction, use session_recall to get back exact text the summary dropped. "
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
    include_workflows: bool = True,
    project: str | None = None,
) -> str:
    """
    Suggest reusable skills from repeated work across all your agents.

    Combines repeated multi-step tool workflows (the same commands/edits in the
    same order across sessions) with recurring topic clusters, then checks each
    against your installed skills: create, update <existing>, or skip.
    Returns ranked ideas with steps, triggers and evidence sessions. Never
    writes files or calls a model; use draft_skill next.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."

    _refresh_history_index()
    root = handoff.resolve_project(project) if project else None
    candidates = suggest_skill_candidates(
        _get_graph_index(),
        sources=valid,
        max_candidates=max_candidates,
        min_sessions=min_sessions,
        days_back=days_back,
        include_workflows=include_workflows,
        project=root,
        installed=skill_store.discover_installed_skills(handoff.resolve_project(project)),
    )
    return format_skill_candidates(candidates)


@mcp.tool()
@_redacted
def draft_skill(candidate_id: str, sources: list[str] | str = "all") -> str:
    """
    Draft an Agent Skills SKILL.md (works in Claude Code, Codex, Gemini CLI,
    Cursor, OpenCode...) for a candidate from suggest_skills.

    The draft is deterministic: frontmatter (name, description with triggers),
    steps with real example commands, known failure modes and their fixes,
    verification, and evidence. Review and edit it, then pass it to
    install_skill. Use get_session on the evidence sessions to refine it.
    """
    valid = resolve_sources(sources)
    _refresh_history_index()
    candidate = skills.find_candidate(_get_graph_index(), candidate_id, valid or None)
    if candidate is None:
        return f"Unknown candidate '{candidate_id}'. Run suggest_skills first and use one of its IDs."
    note = ""
    if candidate.action in ("skip", "update") and candidate.overlap_with:
        note = f"Note: {candidate.overlap_reason} (`{candidate.overlap_with}`).\n\n"
    return f"{note}{skills.draft_skill_md(candidate)}"


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False))
@_redacted
def install_skill(
    skill_md: str | None = None,
    candidate_id: str | None = None,
    targets: list[str] | None = None,
    scope: str = "user",
    project: str | None = None,
    apply: bool = False,
    overwrite: bool = False,
) -> str:
    """
    Install a SKILL.md for several agents at once (dry run unless apply=true).

    skill_md: the reviewed SKILL.md text (or pass candidate_id to install the
    unedited draft). targets: "claude" (~/.claude/skills) and/or "agents"
    (~/.agents/skills, read by Codex and other Agent Skills clients), or
    "codex" (~/.codex/skills, older Codex). scope: "user" or "project".
    Existing skills are never replaced unless overwrite=true.
    """
    _refresh_history_index()
    index = _get_graph_index()
    steps = None
    if candidate_id:
        candidate = skills.find_candidate(index, candidate_id)
        if candidate is None:
            return f"Unknown candidate '{candidate_id}'. Run suggest_skills first."
        skill_md = skill_md or skills.draft_skill_md(candidate)
        steps = candidate.steps or None
    if not skill_md:
        return "Pass skill_md (the SKILL.md text) or candidate_id."
    root = handoff.resolve_project(project) if (project or scope == "project") else None
    return skill_store.install_skill(index, skill_md, targets, scope, root, apply, overwrite, candidate_id, steps)


@mcp.tool()
@_redacted
def skill_health(days_back: int | None = 60, project: str | None = None) -> str:
    """
    Check how your installed skills are actually used across agents: invocations
    per agent, last use, missed triggers (matching work where the skill never
    fired, meaning its description needs work), outcome proxies, references to
    commands that no longer appear, and a keep / revise / update / retire call.
    """
    _refresh_history_index()
    root = handoff.resolve_project(project) if project else None
    installed = skill_store.discover_installed_skills(root)
    report = skills.skill_health(_get_graph_index(), installed, days_back, root)
    return skills.format_skill_health(report, days_back)


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
def session_recall(
    session_id: str,
    query: str = "",
    source: str = "claude",
    message: int | None = None,
    max_hits: int = 5,
) -> str:
    """
    Get back exact text from one session, including what context compaction summarized away.

    session_id: the session to search (the post-compaction checkpoint names it).
    query: words to find, e.g. an error string, file name or decision.
    source: claude | codex
    message: read message [mN] in full instead of searching.

    Reads the transcript on disk directly, so it sees the current, still-running session.
    """
    if source not in compaction.RECALL_SOURCES:
        return f"source must be one of {', '.join(compaction.RECALL_SOURCES)}"
    root = next((a.root for a in _get_adapters() if a.name == source and a.root), None)
    path = compaction.find_transcript(session_id, source, root) if root else None
    if path is None:
        return f"No {source} transcript found for session '{session_id}'."
    # ponytail: re-parses the transcript on every call; cache by (path, mtime) if large sessions feel slow.
    session = compaction.load_session(path, source)
    if not session:
        return f"Session '{session_id}' has no messages."
    return compaction.recall(session, query, message, max_hits)


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


# Analytics

@mcp.tool()
@_redacted
def analyze_sessions(
    project: str | None = None,
    days_back: int | None = 30,
    sources: list[str] | str = "all",
    group_by: str = "project",
) -> str:
    """
    Analyze your coding-agent sessions: active time, tokens, tool mix, top and
    failing commands, most edited files, friction (retry loops, corrections,
    interrupts, rejected tool calls, compactions) and a directional comparison
    of agents on shared topics.

    project: limit to one repository (omit for all projects).
    group_by: project | source | day | model.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    _refresh_history_index()
    root = handoff.resolve_project(project) if project else None
    result = analytics.analyze(_get_graph_index(), valid, root, days_back)
    return analytics.format_analysis(result, group_by)


def _report_sections(valid: list[str], root: str | None, days_back: int | None) -> list[tuple[str, list[tuple[str, str]]]]:
    index = _get_graph_index()
    candidates = suggest_skill_candidates(
        index, sources=valid, max_candidates=6, min_sessions=2, days_back=days_back,
        include_workflows=True, project=root, installed=skill_store.discover_installed_skills(root),
    )
    rules = memory_sync.suggest_rules(index, root, valid, days_back, 2)
    health = skills.skill_health(index, skill_store.discover_installed_skills(root), days_back, root)
    return [
        ("Skill candidates", [
            (f"{c.name} ({c.action})", (" -> ".join(c.steps) + " · " if c.steps else "") + c.why + f" · id {c.candidate_id}")
            for c in candidates
        ]),
        ("Installed skill health", [
            (f"{h.name}: {h.recommendation}", f"used {sum(h.by_source.values())}× · {len(h.missed)} missed trigger(s) · {h.outcome}")
            for h in health[:10]
        ]),
        ("Rule candidates from your corrections", [
            (r.rule, f"{r.kind} · {len(r.sessions)} sessions" + (f" · already in {', '.join(r.documented_in)}" if r.documented_in else ""))
            for r in rules[:8]
        ]),
    ]


@mcp.tool()
@_redacted
def history_report(
    project: str | None = None,
    days_back: int | None = 30,
    sources: list[str] | str = "all",
) -> str:
    """
    Write a self-contained HTML report (analytics, friction, cross-agent
    comparison, skill and rule candidates) to ~/.agent-history-mcp/reports and
    return its path. Local only; no model calls.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    _refresh_history_index()
    root = handoff.resolve_project(project) if project else None
    result = analytics.analyze(_get_graph_index(), valid, root, days_back)
    page = analytics.render_html(result, _report_sections(valid, root, days_back))
    out_dir = agent_home() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import date
    path = out_dir / f"report-{date.today().isoformat()}.html"
    path.write_text(redact(page), encoding="utf-8")
    return f"Report written to {path} ({len(result.sessions)} sessions, {result.scope}). Open it in a browser."


# Scorecards

def _scorecards(sources: list[str], days_back: int | None, project: str | None = None) -> list:
    _refresh_history_index()
    index = _get_graph_index()
    root = handoff.resolve_project(project) if project else None
    sessions = analysis.load_sessions(index, sources, root, days_back)
    cards = analysis.analyze_sessions(sessions, days_back=None, sources=sources)
    analysis.persist_analysis(index.conn, sessions, cards)
    return cards


@mcp.tool()
@_redacted
def score_sessions(
    project: str | None = None,
    days_back: int | None = 30,
    sources: list[str] | str = "all",
    report_path: str | None = None,
) -> str:
    """
    Score recent sessions on prompt clarity, context efficiency, rework,
    consistency and completion, and list the lowest and highest scorers.
    If report_path is given, also writes the report as local Markdown.
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    cards = _scorecards(valid, days_back, project)
    report = analysis.format_aggregate_report(cards, days_back, valid)
    if report_path:
        path = Path(report_path).expanduser()
        analysis.write_markdown_report(path, "Session Scorecards", redact(report))
        report += f"\n\nMarkdown report written to: {path}"
    return report


@mcp.tool()
@_redacted
def get_session_scorecard(session_id: str, source: str) -> str:
    """
    Explain one session's scorecard. session_id may be a full ID or a unique
    prefix from list_sessions.
    """
    valid = resolve_sources(source)
    if len(valid) != 1:
        return f"source must be one of {', '.join(SOURCE_NAMES)}"
    card = analysis.find_scorecard(_scorecards(valid, None), session_id, valid[0])
    if not card:
        return f"Session '{session_id}' not found in {valid[0]} history."
    return analysis.format_scorecard(card)


@mcp.tool()
@_redacted
def compare_sessions(
    before_session_id: str,
    after_session_id: str,
    before_source: str = "claude",
    after_source: str = "claude",
) -> str:
    """
    Compare two session scorecards to see whether a workflow improved.
    Session IDs may be full IDs or unique prefixes.
    """
    before_valid, after_valid = resolve_sources(before_source), resolve_sources(after_source)
    if len(before_valid) != 1 or len(after_valid) != 1:
        return f"sources must be one of {', '.join(SOURCE_NAMES)}"
    cards = _scorecards(sorted({before_valid[0], after_valid[0]}), None)
    before = analysis.find_scorecard(cards, before_session_id, before_valid[0])
    after = analysis.find_scorecard(cards, after_session_id, after_valid[0])
    if not before:
        return f"Before session '{before_session_id}' not found in {before_valid[0]} history."
    if not after:
        return f"After session '{after_session_id}' not found in {after_valid[0]} history."
    return analysis.format_comparison(before, after)


@mcp.tool()
@_redacted
def find_friction_patterns(
    project: str | None = None,
    days_back: int | None = 30,
    sources: list[str] | str = "all",
    min_sessions: int = 2,
) -> str:
    """
    List friction patterns that recur across sessions (repeated file reads,
    failing commands, correction loops, vague openers, tool-call explosions).
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    return analysis.format_friction_patterns(_scorecards(valid, days_back, project), min_sessions=max(1, min_sessions))


@mcp.tool()
@_redacted
def suggest_instruction_updates(
    project: str | None = None,
    targets: list[str] = ["CLAUDE.md", "AGENTS.md"],
    days_back: int | None = 30,
    sources: list[str] | str = "all",
    max_suggestions: int = 8,
) -> str:
    """
    Suggest CLAUDE.md / AGENTS.md rules from friction that recurs across
    sessions, with the sessions as evidence. Read-only: never writes files
    (use sync_memory to apply a rule).
    """
    valid = resolve_sources(sources)
    if not valid:
        return f"Invalid sources. Use {SOURCES_HELP}."
    cards = _scorecards(valid, days_back, project)
    suggestions = analysis.suggest_instruction_updates(cards, targets=targets, max_suggestions=max_suggestions)
    analysis.persist_instruction_suggestions(_get_graph_index().conn, suggestions)
    return analysis.format_instruction_suggestions(suggestions)


# Prompts (surface as slash commands in clients that support MCP prompts)

@mcp.prompt(name="handoff", description="Continue work from where the last agent stopped in this repository")
def handoff_prompt(project: str = "") -> str:
    target = f'project="{project}"' if project else "the current repository"
    return (
        f"Call the agent-history handoff_brief tool for {target}. Read the brief, then: "
        "1) restate the goal and where the previous session stopped in two sentences, "
        "2) avoid the listed dead ends, 3) verify the repository state matches the brief, "
        "and 4) continue with the suggested next step, asking me first if the goal is unclear."
    )


@mcp.prompt(name="draft-skill", description="Turn a repeated workflow from your history into a reviewed SKILL.md")
def draft_skill_prompt(candidate_id: str = "") -> str:
    pick = f"candidate {candidate_id}" if candidate_id else "the best candidate whose action is create or update"
    return (
        "Use the agent-history tools to turn my repeated work into a skill. "
        f"Run suggest_skills, pick {pick}, and call draft_skill. Open one or two evidence sessions with "
        "get_session(include_tools=true) to check the steps. Improve the draft: precise description with "
        "trigger phrases, generic placeholders instead of project values, and real failure modes. Show me the "
        "final SKILL.md, then call install_skill with apply=false so I can review where it will be written."
    )


@mcp.prompt(name="weekly-review", description="Review the last week of agent sessions: friction, skills, rules")
def weekly_review_prompt(days: str = "7") -> str:
    return (
        f"Using the agent-history tools for the last {days} days: run analyze_sessions, suggest_skills, "
        "suggest_rules and skill_health, then history_report. Summarize for me in under 15 lines: where time "
        "went, the top friction patterns, up to three skills worth creating or fixing, and up to three rules "
        "worth adding to every agent's instructions (via sync_memory). Include the report path."
    )


def run() -> None:
    mcp.run()
