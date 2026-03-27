"""
Search across unified session message lists.

Three-layer approach:
  1. FTS5 (stdlib sqlite3) — BM25 ranking, multi-word AND, phrase search, prefix search
  2. Session density scoring — match_count / total_messages ranks sessions by topic relevance
  3. Answer-aware context — returns Q→A pairs (problem + solution) instead of ±N neighbours
  4. rapidfuzz fallback — handles typos and 0-result FTS5 queries
"""

from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    session_id:    str
    session_title: str
    session_date:  str
    source:        str           # "codex" | "claude"
    matched_role:  str           # "user" | "assistant" — which message triggered the match
    question_text: str | None    # user-side message text (may be None at session boundary)
    question_ts:   str | None
    answer_text:   str | None    # assistant-side message text (may be None at session boundary)
    answer_ts:     str | None
    density_score: float = 0.0


# ── Private helper — answer-aware Q&A pair ────────────────────────────────────

def _build_qa_pair(
    session: dict,
    match_idx: int,
) -> tuple[dict | None, dict | None]:
    """
    Given a session and the index of the matched message, return (question, answer)
    where question is the user turn and answer is the assistant turn.

    Scans up to MAX_SCAN adjacent messages to find the right pairing.
    Returns None for either side if no suitable message is found.
    """
    MAX_SCAN = 3
    msgs = session["messages"]
    matched = msgs[match_idx]

    if matched["role"] == "user":
        question = matched
        answer = None
        for j in range(match_idx + 1, min(len(msgs), match_idx + MAX_SCAN + 1)):
            if msgs[j]["role"] == "assistant":
                answer = msgs[j]
                break
    else:  # "assistant"
        answer = matched
        question = None
        for j in range(match_idx - 1, max(-1, match_idx - MAX_SCAN - 1), -1):
            if msgs[j]["role"] == "user":
                question = msgs[j]
                break

    return question, answer


# ── FTS5 index ────────────────────────────────────────────────────────────────

def build_fts_index(sessions: list[dict]) -> sqlite3.Connection | None:
    """
    Build an in-memory SQLite FTS5 index over all messages.
    Returns None if FTS5 is unavailable in this Python build (extremely rare).
    """
    try:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("""
            CREATE VIRTUAL TABLE msg_fts USING fts5(
                text,
                session_id  UNINDEXED,
                source      UNINDEXED,
                role        UNINDEXED,
                msg_idx     UNINDEXED
            )
        """)
        conn.executemany(
            "INSERT INTO msg_fts(text, session_id, source, role, msg_idx) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (msg["text"], session["session_id"], session["source"],
                 msg["role"], idx)
                for session in sessions
                for idx, msg in enumerate(session["messages"])
            ),
        )
        conn.commit()
        return conn
    except sqlite3.OperationalError:
        return None


# ── FTS5 search ───────────────────────────────────────────────────────────────

def search_fts(
    conn: sqlite3.Connection,
    query: str,
    session_map: dict,
    sources: list[str],
    max_results: int,
) -> list[SearchHit]:
    """
    BM25-ranked FTS5 search → density scoring → answer-aware Q&A pairs.
    Raises sqlite3.OperationalError on malformed query (caller handles it).
    """
    sources_set = set(sources)

    rows = conn.execute(
        "SELECT session_id, source, role, msg_idx, rank "
        "FROM msg_fts WHERE msg_fts MATCH ? ORDER BY rank",
        (query,),
    ).fetchall()

    # Filter to requested sources
    rows = [r for r in rows if r[1] in sources_set]
    if not rows:
        return []

    # Group by session: track match count and best (lowest rank) row per session
    session_matches: dict[str, dict] = {}
    for session_id, source, role, msg_idx, rank in rows:
        if session_id not in session_map:
            continue
        if session_id not in session_matches:
            session_matches[session_id] = {
                "count": 0,
                "best_rank": rank,
                "best_msg_idx": msg_idx,
                "best_role": role,
                "source": source,
            }
        entry = session_matches[session_id]
        entry["count"] += 1
        if rank < entry["best_rank"]:  # more negative = better BM25 score
            entry["best_rank"] = rank
            entry["best_msg_idx"] = msg_idx
            entry["best_role"] = role

    # Density score = match_count / total_messages; sort sessions descending
    scored = []
    for session_id, entry in session_matches.items():
        session = session_map[session_id]
        total = len(session["messages"])
        density = entry["count"] / total if total else 0.0
        scored.append((density, entry["best_rank"], session_id, entry))

    scored.sort(key=lambda x: (-x[0], x[1]))  # density desc, then BM25 asc

    hits: list[SearchHit] = []
    for density, _rank, session_id, entry in scored[:max_results]:
        session = session_map[session_id]
        q, a = _build_qa_pair(session, entry["best_msg_idx"])
        hits.append(SearchHit(
            session_id    = session_id,
            session_title = session["session_title"],
            session_date  = session["session_date"],
            source        = session["source"],
            matched_role  = entry["best_role"],
            question_text = q["text"][:1500] if q else None,
            question_ts   = q["ts"]          if q else None,
            answer_text   = a["text"][:1500] if a else None,
            answer_ts     = a["ts"]          if a else None,
            density_score = density,
        ))

    return hits


# ── rapidfuzz fallback ────────────────────────────────────────────────────────

def search_fuzzy(
    sessions: list[dict],
    query: str,
    max_results: int,
) -> list[SearchHit]:
    """
    partial_ratio fuzzy search as fallback when FTS5 finds nothing or fails.
    Returns [] silently if rapidfuzz is not installed.
    """
    try:
        from rapidfuzz.fuzz import partial_ratio
    except ImportError:
        return []

    THRESHOLD = 65
    scored: list[tuple[float, dict, int]] = []

    for session in sessions:
        for idx, msg in enumerate(session["messages"]):
            score = partial_ratio(query, msg["text"])
            if score >= THRESHOLD:
                scored.append((score, session, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    # One hit per session (highest-scoring message wins)
    seen: set[str] = set()
    hits: list[SearchHit] = []
    for score, session, idx in scored:
        sid = session["session_id"]
        if sid in seen:
            continue
        seen.add(sid)
        q, a = _build_qa_pair(session, idx)
        hits.append(SearchHit(
            session_id    = sid,
            session_title = session["session_title"],
            session_date  = session["session_date"],
            source        = session["source"],
            matched_role  = session["messages"][idx]["role"],
            question_text = q["text"][:1500] if q else None,
            question_ts   = q["ts"]          if q else None,
            answer_text   = a["text"][:1500] if a else None,
            answer_ts     = a["ts"]          if a else None,
            density_score = score / 100.0,
        ))
        if len(hits) >= max_results:
            break

    return hits


# ── Orchestrator ──────────────────────────────────────────────────────────────

def search_smart(
    conn: sqlite3.Connection | None,
    query: str,
    sessions: list[dict],
    session_map: dict,
    sources: list[str],
    max_results: int = 5,
) -> list[SearchHit]:
    """
    FTS5 → density ranking → answer-aware pairs → fuzzy fallback.
    """
    if conn is not None:
        try:
            hits = search_fts(conn, query, session_map, sources, max_results)
            if hits:
                return hits
        except sqlite3.OperationalError:
            pass  # malformed query → fall through to fuzzy

    return search_fuzzy(sessions, query, max_results)


# ── Formatters ────────────────────────────────────────────────────────────────

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

        if hit.question_text is not None:
            ts = f" {hit.question_ts}" if hit.question_ts else ""
            lines.append(f"[YOU ASKED]{ts}")
            lines.append(hit.question_text)
            lines.append("")

        if hit.answer_text is not None:
            ts = f" {hit.answer_ts}" if hit.answer_ts else ""
            lines.append(f"[ANSWER]{ts}")
            lines.append(hit.answer_text)
        elif hit.question_text is not None:
            lines.append("[ANSWER] (no response recorded)")

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
