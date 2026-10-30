"""
Persistent local graph index for Codex and Claude chat history.

This is intentionally local-first and deterministic: it extracts paths, commands,
errors, APIs, and technical topics with regex/token rules, then stores an
explainable SQLite graph that can be refreshed incrementally from JSONL history.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path

from .model import Event, agent_home, project_root
from .sources import SOURCE_NAMES, SourceAdapter
from .sources.claude import ClaudeAdapter
from .sources.codex import CodexAdapter


DEFAULT_DB_PATH = agent_home() / "history_graph.sqlite"
SCHEMA_VERSION = 4

# Graph size knobs. Co-occurrence links grow with the square of the term count
# per Q&A pair, and related_to with the square of each term's session count.
COOCCUR_TERMS = 12           # strongest terms per Q&A pair linked to each other (66 pairs)
RELATED_MAX_DF_SHARE = 0.02  # terms in more than 2% of sessions don't link sessions...
RELATED_MIN_DF_CAP = 25      # ...unless the corpus is small enough that 2% is under 25


@dataclass(frozen=True)
class GraphTerm:
    kind: str
    value: str
    norm: str
    weight: float

    @property
    def key(self) -> str:
        return f"term:{self.kind}:{self.norm}"


@dataclass
class GraphHit:
    session_id: str
    session_key: str
    session_title: str
    session_date: str
    source: str
    question_text: str | None
    question_ts: str | None
    answer_text: str | None
    answer_ts: str | None
    matched_topics: list[str]
    related_topics: list[str]
    why: str
    score: float


@dataclass
class RefreshStats:
    files_seen: int = 0
    files_added: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    sessions_indexed: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "files_added": self.files_added,
            "files_changed": self.files_changed,
            "files_removed": self.files_removed,
            "files_unchanged": self.files_unchanged,
            "sessions_indexed": self.sessions_indexed,
        }


_STOP_WORDS = {
    "about", "above", "after", "again", "agent", "also", "always", "because",
    "before", "being", "below", "between", "build", "called", "cannot",
    "change", "chat", "chats", "check", "claude", "codex", "could", "current",
    "default", "doing", "during", "each", "every", "file", "files", "first",
    "from", "get", "gets", "given", "have", "having", "history", "into",
    "just", "keep", "last", "like", "make", "message", "messages", "more",
    "need", "needs", "only", "other", "output", "past", "please", "project",
    "query", "read", "really", "request", "return", "right", "same", "search",
    "session", "sessions", "should", "some", "than", "that", "their", "them",
    "then", "there", "these", "thing", "this", "those", "through", "tool",
    "tools", "update", "using", "want", "when", "where", "which", "while",
    "with", "work", "works", "would", "your",
}

_COMMAND_PREFIXES = (
    "az", "cargo", "docker", "dotnet", "git", "gh", "go", "kubectl", "make",
    "mvn", "node", "npm", "pnpm", "powershell", "pwsh", "pytest", "python",
    "python3", "ruff", "uv", "yarn",
)

_PATH_RE = re.compile(
    r"(?<![\w.-])("
    r"[A-Za-z]:[\\/][^\s`'\"<>|]{3,}|"
    r"(?:\.{1,2}[\\/]|[/\\])?[A-Za-z0-9_.@()-]+(?:[\\/][A-Za-z0-9_.@() -]+)+\.[A-Za-z0-9]{1,8}|"
    r"[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|rs|go|java|cs|cpp|cxx|c|h|hpp|cu|cuh|sql|sh|ps1)"
    r")(?![A-Za-z0-9])"
)
_BACKTICK_RE = re.compile(r"`([^`\n]{2,180})`")
_COMMAND_LINE_RE = re.compile(
    rf"(?im)^\s*((?:{'|'.join(re.escape(c) for c in _COMMAND_PREFIXES)})\b[^\n]{{0,180}})"
)
_ERROR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b(?::?\s+[^\n.]{1,120})?")
_COMMON_ERROR_RE = re.compile(
    r"\b(?:access denied|failed|failure|illegal address|not found|out of memory|"
    r"permission denied|segmentation fault|timed out|timeout|traceback)\b[^\n.]{0,100}",
    re.IGNORECASE,
)
_API_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,4}\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]{2,}")
_SIGNAL_CHAR_RE = re.compile(r"[0-9_+#.-]")


def _clean_value(value: str) -> str:
    return value.strip().strip(".,:;!?)]}\"'")


def _norm(value: str) -> str:
    return " ".join(_clean_value(value).replace("\\", "/").lower().split())


def _add_term(terms: dict[tuple[str, str], GraphTerm], kind: str, value: str, weight: float) -> None:
    value = _clean_value(value)
    norm = _norm(value)
    if len(norm) < 3:
        return
    _put_term(terms, kind, value, norm, weight)


def _put_term(terms: dict[tuple[str, str], GraphTerm], kind: str, value: str, norm: str, weight: float) -> None:
    key = (kind, norm)
    existing = terms.get(key)
    if existing is None or weight > existing.weight:
        terms[key] = GraphTerm(kind=kind, value=value, norm=norm, weight=weight)


def _looks_like_command(value: str) -> bool:
    stripped = value.strip()
    first = stripped.split(maxsplit=1)[0].lower() if stripped else ""
    return first in _COMMAND_PREFIXES


def extract_graph_terms(text: str, max_terms: int = 80) -> list[GraphTerm]:
    terms: dict[tuple[str, str], GraphTerm] = {}

    for match in _PATH_RE.finditer(text):
        _add_term(terms, "path", match.group(1), 4.0)

    for match in _BACKTICK_RE.finditer(text):
        value = match.group(1)
        if _looks_like_command(value):
            _add_term(terms, "command", value, 3.5)

    for match in _COMMAND_LINE_RE.finditer(text):
        _add_term(terms, "command", match.group(1), 3.5)

    for match in _ERROR_RE.finditer(text):
        _add_term(terms, "error", match.group(0), 5.0)
    for match in _COMMON_ERROR_RE.finditer(text):
        _add_term(terms, "error", match.group(0), 4.5)

    for match in _API_RE.finditer(text):
        value = match.group(0)
        if not value.lower().startswith(("http.", "https.")):
            _add_term(terms, "api", value, 3.0)

    word_items: list[tuple[str, str]] = []
    for match in _WORD_RE.finditer(text):
        raw = match.group(0).strip(".")
        norm = raw.lower()
        if norm in _STOP_WORDS or len(norm) < 3:
            continue
        if norm.startswith(("http", "www")):
            continue
        word_items.append((raw, norm))
        signal = len(norm) >= 4 or raw.isupper() or _SIGNAL_CHAR_RE.search(raw)
        if signal:
            _put_term(terms, "topic", raw, norm, 1.0 if raw.islower() else 1.4)

    compact_words = [(raw, norm) for raw, norm in word_items if norm not in _STOP_WORDS]
    for size, weight in ((2, 2.0), (3, 2.4)):
        for i in range(0, max(0, len(compact_words) - size + 1)):
            chunk = compact_words[i:i + size]
            norms = [n for _r, n in chunk]
            if len(set(norms)) != len(norms):
                continue
            _put_term(terms, "topic", " ".join(raw for raw, _n in chunk), " ".join(norms), weight)

    return sorted(terms.values(), key=lambda t: (-t.weight, t.kind, t.norm))[:max_terms]


def session_filters(
    sources: list[str] | None,
    project: str | None,
    days_back: int | None,
    alias: str = "s",
) -> tuple[list[str], list]:
    """SQL WHERE clauses shared by every query that scopes sessions."""
    clauses: list[str] = []
    params: list = []
    if sources is not None:
        valid = [s for s in sources if s in SOURCE_NAMES]
        clauses.append(f"{alias}.source IN ({','.join('?' for _ in valid) or 'NULL'})")
        params.extend(valid)
    if project:
        root = project_root(project)
        clauses.append(f"({alias}.project = ? OR {alias}.cwd = ? OR {alias}.cwd LIKE ?)")
        params.extend([root, root, root.rstrip("/") + "/%"])
    if days_back is not None and days_back > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        clauses.append(f"({alias}.ended_at >= ? OR ({alias}.ended_at = '' AND {alias}.session_date >= ?))")
        params.extend([cutoff, cutoff[:10]])
    return clauses, params


def event_from_row(row: sqlite3.Row) -> Event:
    return Event(
        idx=row["idx"],
        ts=row["ts"] or "",
        kind=row["kind"],
        text=row["text"] or "",
        tool=row["tool"] or "",
        canon_tool=row["canon_tool"] or "",
        command=row["command"] or "",
        cmd_sig=row["cmd_sig"] or "",
        paths=json.loads(row["paths"]) if row["paths"] else [],
        is_error=bool(row["is_error"]),
        exit_code=row["exit_code"],
        call_id=row["call_id"] or "",
        meta=json.loads(row["meta"]) if row["meta"] else {},
    )


def _session_key(source: str, session_id: str) -> str:
    return f"session:{source}:{session_id}"


def _message_key(source: str, session_id: str, idx: int) -> str:
    return f"message:{source}:{session_id}:{idx:06d}"


class HistoryGraphIndex:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or os.environ.get("AGENT_HISTORY_GRAPH_DB") or DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or row["value"] != str(SCHEMA_VERSION):
            # The database is derived from history files, so an old schema is simply rebuilt.
            # installed_skills is our own record of writes and survives the rebuild.
            tables = [
                r["name"]
                for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
                if r["name"] not in ("meta", "installed_skills") and not r["name"].startswith("sqlite_")
            ]
            for table in tables:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                session_key TEXT,
                change_token TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                session_date TEXT NOT NULL,
                path TEXT NOT NULL,
                cwd TEXT NOT NULL DEFAULT '',
                project TEXT NOT NULL DEFAULT '',
                git_branch TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL DEFAULT '',
                ended_at TEXT NOT NULL DEFAULT '',
                n_messages INTEGER NOT NULL DEFAULT 0,
                n_user INTEGER NOT NULL DEFAULT 0,
                n_tool_calls INTEGER NOT NULL DEFAULT 0,
                n_errors INTEGER NOT NULL DEFAULT 0,
                tools_available INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS events (
                session_key TEXT NOT NULL,
                idx INTEGER NOT NULL,
                ts TEXT,
                kind TEXT NOT NULL,
                text TEXT,
                tool TEXT,
                canon_tool TEXT,
                command TEXT,
                cmd_sig TEXT,
                paths TEXT,
                is_error INTEGER NOT NULL DEFAULT 0,
                exit_code INTEGER,
                call_id TEXT,
                meta TEXT,
                PRIMARY KEY(session_key, idx)
            );

            CREATE TABLE IF NOT EXISTS api_calls (
                session_key TEXT NOT NULL,
                idx INTEGER NOT NULL,
                ts TEXT,
                model TEXT,
                sidechain INTEGER NOT NULL DEFAULT 0,
                input INTEGER NOT NULL DEFAULT 0,
                cache_read INTEGER NOT NULL DEFAULT 0,
                cache_write_5m INTEGER NOT NULL DEFAULT 0,
                cache_write_1h INTEGER NOT NULL DEFAULT 0,
                output INTEGER NOT NULL DEFAULT 0,
                reasoning INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(session_key, idx)
            );

            CREATE TABLE IF NOT EXISTS installed_skills (
                name TEXT NOT NULL,
                path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                candidate_id TEXT,
                steps TEXT,
                installed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_key TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                msg_idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                ts TEXT,
                FOREIGN KEY(session_key) REFERENCES sessions(session_key) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                src_key TEXT NOT NULL,
                dst_key TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                provenance TEXT NOT NULL DEFAULT 'EXTRACTED',
                confidence REAL NOT NULL DEFAULT 1.0,
                evidence TEXT,
                PRIMARY KEY(src_key, dst_key, relation)
            );

            CREATE TABLE IF NOT EXISTS message_terms (
                message_key TEXT NOT NULL,
                term_key TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(message_key, term_key)
            );

            CREATE TABLE IF NOT EXISTS session_terms (
                session_key TEXT NOT NULL,
                term_key TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(session_key, term_key)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project, ended_at);
            CREATE INDEX IF NOT EXISTS idx_events_tool ON events(canon_tool, cmd_sig);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_key, msg_idx);
            CREATE INDEX IF NOT EXISTS idx_message_terms_term ON message_terms(term_key);
            CREATE INDEX IF NOT EXISTS idx_session_terms_term ON session_terms(term_key);
            CREATE INDEX IF NOT EXISTS idx_edges_relation_src ON edges(relation, src_key);
            CREATE INDEX IF NOT EXISTS idx_edges_relation_dst ON edges(relation, dst_key);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # Read helpers shared by search, handoff, analytics, and skills

    def session_rows(
        self,
        sources: list[str] | None = None,
        project: str | None = None,
        days_back: int | None = None,
        limit: int | None = None,
        order: str = "ended_at DESC",
    ) -> list[sqlite3.Row]:
        clauses, params = session_filters(sources, project, days_back)
        sql = f"SELECT * FROM sessions s WHERE {' AND '.join(clauses) or '1=1'} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql, params).fetchall()

    def events_for(self, session_key: str, kinds: tuple[str, ...] | None = None) -> list[Event]:
        sql = "SELECT * FROM events WHERE session_key = ?"
        params: list = [session_key]
        if kinds:
            sql += f" AND kind IN ({','.join('?' for _ in kinds)})"
            params.extend(kinds)
        sql += " ORDER BY idx"
        return [event_from_row(row) for row in self.conn.execute(sql, params).fetchall()]

    def calls_for(self, session_key: str) -> list[sqlite3.Row]:
        """Per-API-call token usage, in time order."""
        return self.conn.execute("SELECT * FROM api_calls WHERE session_key = ? ORDER BY idx", (session_key,)).fetchall()

    def load_sessions(self, sources: list[str] | None = None, project: str | None = None) -> list[dict]:
        """Rebuild session/message dicts from the index for keyword search (no JSONL re-parse)."""
        rows = self.session_rows(sources, project, order="session_date DESC, ended_at DESC")
        sessions: dict[str, dict] = {}
        for row in rows:
            sessions[row["session_key"]] = {
                "session_id": row["session_id"],
                "session_title": row["title"],
                "session_date": row["session_date"],
                "source": row["source"],
                "session_key": row["session_key"],
                "project": row["project"],
                "messages": [],
            }
        if not sessions:
            return []
        for msg in self.conn.execute("SELECT session_key, role, text, ts FROM messages ORDER BY session_key, msg_idx"):
            session = sessions.get(msg["session_key"])
            if session is None:
                continue
            session["messages"].append({
                "role": msg["role"],
                "text": msg["text"],
                "ts": msg["ts"] or "",
                "session_id": session["session_id"],
                "session_title": session["session_title"],
                "session_date": session["session_date"],
                "source": session["source"],
            })
        return [s for s in sessions.values() if s["messages"]]

    def find_session(self, session_id: str, source: str | None = None) -> sqlite3.Row | None:
        sql = "SELECT * FROM sessions WHERE (session_id = ? OR session_id LIKE ?)"
        params: list = [session_id, f"{session_id}%"]
        if source:
            sql += " AND source = ?"
            params.append(source)
        return self.conn.execute(sql + " ORDER BY session_id = ? DESC LIMIT 1", (*params, session_id)).fetchone()

    def refresh(
        self,
        codex_path: Path | None = None,
        claude_path: Path | None = None,
        adapters: list[SourceAdapter] | None = None,
    ) -> RefreshStats:
        """
        Incrementally sync the index with every adapter's units. Passing
        codex_path/claude_path keeps the original two-source API.
        """
        stats = RefreshStats()
        if adapters is None:
            adapters = []
            if codex_path:
                adapters.append(CodexAdapter(codex_path))
            if claude_path:
                adapters.append(ClaudeAdapter(claude_path))

        discovered: dict[str, tuple[SourceAdapter, object]] = {}
        scanned_sources: set[str] = set()
        for adapter in adapters:
            adapter.diagnostics = []
            try:
                units = adapter.discover() if adapter.available() else []
            except Exception as exc:  # an unreadable store must not break the other sources
                stats.diagnostics.append(f"{adapter.name}: discovery failed ({type(exc).__name__}: {exc})")
                continue
            scanned_sources.add(adapter.name)
            stats.diagnostics.extend(f"{adapter.name}: {msg}" for msg in adapter.diagnostics)
            for unit in units:
                discovered[unit.key] = (adapter, unit)
        stats.files_seen = len(discovered)

        known_paths = {
            row["path"]: row
            for row in self.conn.execute("SELECT path, source, session_key, change_token FROM files").fetchall()
        }

        for path, row in known_paths.items():
            # Only drop units of sources that were scanned; a missing source is not a deletion.
            if path not in discovered and row["source"] in scanned_sources:
                if row["session_key"]:
                    self._delete_session(row["session_key"])
                self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
                stats.files_removed += 1

        for path_key, (adapter, unit) in discovered.items():
            known = known_paths.get(path_key)
            if known and known["change_token"] == unit.change_token:
                stats.files_unchanged += 1
                continue

            if known and known["session_key"]:
                self._delete_session(known["session_key"])

            try:
                session = adapter.parse(unit)
            except Exception as exc:
                stats.diagnostics.append(f"{adapter.name}: could not parse {path_key} ({type(exc).__name__})")
                session = None
            session_key = None
            if session:
                session_key = self._index_session(session, path_key)
                stats.sessions_indexed += 1
                stats.per_source[adapter.name] = stats.per_source.get(adapter.name, 0) + 1

            self.conn.execute(
                """
                INSERT OR REPLACE INTO files(path, source, session_key, change_token, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (path_key, adapter.name, session_key, unit.change_token, time.time()),
            )
            if known:
                stats.files_changed += 1
            else:
                stats.files_added += 1

        if stats.files_added or stats.files_changed or stats.files_removed:
            self._rebuild_related_sessions()

        self.conn.commit()
        return stats

    def _insert_nodes(self, nodes: dict[str, tuple[str, str]]) -> None:
        """nodes maps node_key -> (kind, value); an existing node keeps its first value."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO nodes(node_key, kind, value) VALUES (?, ?, ?)",
            [(key, kind, value) for key, (kind, value) in nodes.items()],
        )

    def _insert_edges(self, edges: list[tuple[str, str, str, float]]) -> None:
        """edges are (src_key, dst_key, relation, weight); a repeated edge takes the later weight."""
        self.conn.executemany(
            """
            INSERT INTO edges(src_key, dst_key, relation, weight, provenance, confidence)
            VALUES (?, ?, ?, ?, 'EXTRACTED', 1.0)
            ON CONFLICT(src_key, dst_key, relation) DO UPDATE SET weight = excluded.weight
            """,
            edges,
        )

    def _delete_session(self, session_key: str) -> None:
        message_keys = [
            row["message_key"]
            for row in self.conn.execute(
                "SELECT message_key FROM messages WHERE session_key = ?", (session_key,)
            ).fetchall()
        ]
        keys = [session_key, *message_keys]
        if keys:
            placeholders = ",".join("?" for _ in keys)
            self.conn.execute(f"DELETE FROM edges WHERE src_key IN ({placeholders}) OR dst_key IN ({placeholders})", keys * 2)
            self.conn.execute(f"DELETE FROM message_terms WHERE message_key IN ({placeholders})", keys)
        self.conn.execute("DELETE FROM session_terms WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM events WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM api_calls WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM messages WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM sessions WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM nodes WHERE node_key = ?", (session_key,))
        for message_key in message_keys:
            self.conn.execute("DELETE FROM nodes WHERE node_key = ?", (message_key,))

    def _index_session(self, session: dict, path: str) -> str:
        source = session["source"]
        session_id = session["session_id"]
        session_key = _session_key(source, session_id)
        # Rows are collected per session and written with executemany: one execute per row was
        # about a fifth of the whole build.
        nodes: dict[str, tuple[str, str]] = {session_key: ("session", session["session_title"])}
        edges: list[tuple[str, str, str, float]] = []
        message_rows: list[tuple] = []
        message_term_rows: list[tuple[str, str, float]] = []
        events = session.get("events") or []
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sessions(
                session_key, session_id, source, title, session_date, path,
                cwd, project, git_branch, model, input_tokens, output_tokens,
                started_at, ended_at, n_messages, n_user, n_tool_calls, n_errors, tools_available
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_key, session_id, source, session["session_title"], session["session_date"], path,
                session.get("cwd", ""), session.get("project", ""), session.get("git_branch", ""),
                session.get("model", ""), session.get("input_tokens", 0), session.get("output_tokens", 0),
                session.get("started_at", ""), session.get("ended_at", ""), len(session["messages"]),
                sum(1 for m in session["messages"] if m["role"] == "user"),
                sum(1 for e in events if e.kind == "tool_call"),
                sum(1 for e in events if e.kind == "tool_result" and e.is_error),
                1 if session.get("tools_available", True) else 0,
            ),
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO events(
                session_key, idx, ts, kind, text, tool, canon_tool, command, cmd_sig,
                paths, is_error, exit_code, call_id, meta
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(session_key, *event.as_row()) for event in events],
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO api_calls(
                session_key, idx, ts, model, sidechain, input, cache_read,
                cache_write_5m, cache_write_1h, output, reasoning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (session_key, i, c["ts"], c["model"], int(c["sidechain"]), c["input"], c["cache_read"],
                 c["cache_write_5m"], c["cache_write_1h"], c["output"], c["reasoning"])
                for i, c in enumerate(session.get("calls") or [])
            ],
        )

        session_term_weights: defaultdict[str, float] = defaultdict(float)
        message_terms_by_idx: dict[int, list[GraphTerm]] = {}

        # Structured tool calls are stronger evidence than regex hits in prose.
        for event in events:
            if event.kind != "tool_call":
                continue
            structured: list[tuple[str, str, float]] = []
            if event.command:
                structured.append(("command", event.command.splitlines()[0][:180], 4.5))
            if event.canon_tool in ("edit", "write"):
                structured.extend(("path", p, 4.5) for p in event.paths[:5])
            for kind, value, weight in structured:
                norm = _norm(value)
                if len(norm) < 3:
                    continue
                term = GraphTerm(kind=kind, value=_clean_value(value), norm=norm, weight=weight)
                nodes.setdefault(term.key, (term.kind, term.value))
                session_term_weights[term.key] += weight

        for idx, msg in enumerate(session["messages"]):
            message_key = _message_key(source, session_id, idx)
            nodes.setdefault(message_key, ("message", f"{source}:{session_id}:{idx}"))
            message_rows.append((message_key, session_key, idx, msg["role"], msg["text"], msg.get("ts", "")))
            edges.append((session_key, message_key, "contains_message", 1.0))

            terms = extract_graph_terms(msg["text"], max_terms=60)
            message_terms_by_idx[idx] = terms
            for term in terms:
                nodes.setdefault(term.key, (term.kind, term.value))
                message_term_rows.append((message_key, term.key, term.weight))
                session_term_weights[term.key] += term.weight

        edges.extend(self._qa_and_cooccurrence_edges(session, message_terms_by_idx))
        self._insert_nodes(nodes)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO messages(message_key, session_key, msg_idx, role, text, ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            message_rows,
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO message_terms(message_key, term_key, weight) VALUES (?, ?, ?)",
            message_term_rows,
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO session_terms(session_key, term_key, weight) VALUES (?, ?, ?)",
            [(session_key, term_key, weight) for term_key, weight in session_term_weights.items()],
        )
        self._insert_edges(edges)
        return session_key

    @staticmethod
    def _qa_and_cooccurrence_edges(
        session: dict, message_terms_by_idx: dict[int, list[GraphTerm]]
    ) -> list[tuple[str, str, str, float]]:
        edges: list[tuple[str, str, str, float]] = []
        source = session["source"]
        session_id = session["session_id"]
        messages = session["messages"]

        for idx, msg in enumerate(messages):
            if msg["role"] != "user":
                continue
            answer_idx = None
            for j in range(idx + 1, min(len(messages), idx + 4)):
                if messages[j]["role"] == "assistant":
                    answer_idx = j
                    break
            if answer_idx is None:
                continue

            question_key = _message_key(source, session_id, idx)
            answer_key = _message_key(source, session_id, answer_idx)
            edges.append((question_key, answer_key, "answered_by", 1.0))

            combined = {
                term.key: term
                for term in [*message_terms_by_idx.get(idx, []), *message_terms_by_idx.get(answer_idx, [])]
            }
            qa_terms = sorted(combined.values(), key=lambda t: (-t.weight, t.norm))[:COOCCUR_TERMS]
            for left_index, left in enumerate(qa_terms):
                for right in qa_terms[left_index + 1:]:
                    src, dst = sorted((left.key, right.key))
                    weight = min(left.weight, right.weight)
                    edges.append((src, dst, "co_occurs_with", weight))
        return edges

    def _rebuild_related_sessions(self) -> None:
        self.conn.execute("DELETE FROM edges WHERE relation = 'related_to'")
        # Terms shared by many sessions (git status, a common word) relate nothing and
        # make the self-join quadratic, so only fairly specific terms link sessions.
        n_sessions = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        max_df = max(RELATED_MIN_DF_CAP, int(n_sessions * RELATED_MAX_DF_SHARE))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges(src_key, dst_key, relation, weight, provenance, confidence, evidence)
            WITH specific AS (
                SELECT term_key FROM session_terms GROUP BY term_key HAVING COUNT(*) BETWEEN 2 AND ?
            )
            SELECT
                a.session_key,
                b.session_key,
                'related_to',
                SUM(min(a.weight, b.weight)) AS weight,
                'EXTRACTED',
                1.0,
                'shared extracted topics'
            FROM specific t
            JOIN session_terms a ON a.term_key = t.term_key
            JOIN session_terms b ON b.term_key = t.term_key AND a.session_key < b.session_key
            GROUP BY a.session_key, b.session_key
            HAVING COUNT(*) >= 2 AND SUM(min(a.weight, b.weight)) >= 4.0
            """,
            (max_df,),
        )

    def search(self, query: str, sources: list[str], max_results: int = 5) -> list[GraphHit]:
        query_terms = extract_graph_terms(query, max_terms=25)
        if not query_terms:
            return []

        source_set = {s for s in sources if s in SOURCE_NAMES}
        if not source_set:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        matched: defaultdict[str, dict[str, GraphTerm]] = defaultdict(dict)
        session_rows: dict[str, sqlite3.Row] = {}

        for term in query_terms:
            rows = self.conn.execute(
                """
                SELECT s.session_key, s.session_id, s.source, s.title, s.session_date, st.weight
                FROM session_terms st
                JOIN sessions s ON s.session_key = st.session_key
                WHERE st.term_key = ? AND s.source IN ({})
                """.format(",".join("?" for _ in source_set)),
                (term.key, *source_set),
            ).fetchall()
            for row in rows:
                session_key = row["session_key"]
                scores[session_key] += row["weight"] * term.weight
                matched[session_key][term.key] = term
                session_rows[session_key] = row

        direct_scores = dict(scores)
        for session_key, base_score in sorted(direct_scores.items(), key=lambda item: item[1], reverse=True)[:10]:
            rows = self.conn.execute(
                """
                SELECT e.src_key, e.dst_key, e.weight, s.session_key, s.session_id, s.source, s.title, s.session_date
                FROM edges e
                JOIN sessions s ON s.session_key = CASE WHEN e.src_key = ? THEN e.dst_key ELSE e.src_key END
                WHERE e.relation = 'related_to'
                  AND (e.src_key = ? OR e.dst_key = ?)
                  AND s.source IN ({})
                """.format(",".join("?" for _ in source_set)),
                (session_key, session_key, session_key, *source_set),
            ).fetchall()
            for row in rows:
                related_key = row["session_key"]
                if related_key == session_key:
                    continue
                scores[related_key] += min(base_score * 0.25, row["weight"])
                session_rows[related_key] = row

        hits: list[GraphHit] = []
        for session_key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_results]:
            row = session_rows.get(session_key)
            if row is None:
                continue
            qa = self._best_qa_for_session(session_key, [t.key for t in query_terms])
            matched_terms = list(matched.get(session_key, {}).values())
            matched_labels = [f"{term.kind}:{term.value}" for term in matched_terms[:8]]
            related_labels = self._related_topic_labels([term.key for term in matched_terms], limit=8)
            why = self._why(matched_labels, related_labels, session_key in direct_scores)
            hits.append(GraphHit(
                session_id=row["session_id"],
                session_key=session_key,
                session_title=row["title"],
                session_date=row["session_date"],
                source=row["source"],
                question_text=qa.get("question_text"),
                question_ts=qa.get("question_ts"),
                answer_text=qa.get("answer_text"),
                answer_ts=qa.get("answer_ts"),
                matched_topics=matched_labels,
                related_topics=related_labels,
                why=why,
                score=score,
            ))
        return hits

    def _best_qa_for_session(self, session_key: str, term_keys: list[str]) -> dict[str, str | None]:
        if term_keys:
            placeholders = ",".join("?" for _ in term_keys)
            row = self.conn.execute(
                f"""
                SELECT m.msg_idx, SUM(mt.weight) AS score
                FROM message_terms mt
                JOIN messages m ON m.message_key = mt.message_key
                WHERE m.session_key = ? AND mt.term_key IN ({placeholders})
                GROUP BY m.message_key, m.msg_idx
                ORDER BY score DESC, m.msg_idx ASC
                LIMIT 1
                """,
                (session_key, *term_keys),
            ).fetchone()
            best_idx = row["msg_idx"] if row else 0
        else:
            best_idx = 0

        messages = self.conn.execute(
            """
            SELECT msg_idx, role, text, ts
            FROM messages
            WHERE session_key = ?
            ORDER BY msg_idx
            """,
            (session_key,),
        ).fetchall()
        if not messages:
            return {}

        by_idx = {row["msg_idx"]: row for row in messages}
        matched = by_idx.get(best_idx, messages[0])
        if matched["role"] == "user":
            question = matched
            answer = next((row for row in messages if row["msg_idx"] > matched["msg_idx"] and row["role"] == "assistant"), None)
        else:
            answer = matched
            question = next(
                (row for row in reversed(messages) if row["msg_idx"] < matched["msg_idx"] and row["role"] == "user"),
                None,
            )

        return {
            "question_text": question["text"][:1500] if question else None,
            "question_ts": question["ts"] if question else None,
            "answer_text": answer["text"][:1500] if answer else None,
            "answer_ts": answer["ts"] if answer else None,
        }

    def _related_topic_labels(self, term_keys: list[str], limit: int = 8) -> list[str]:
        if not term_keys:
            return []
        placeholders = ",".join("?" for _ in term_keys)
        rows = self.conn.execute(
            f"""
            SELECT n.kind, n.value, MAX(e.weight) AS weight
            FROM (
                SELECT dst_key AS term_key, weight
                FROM edges
                WHERE relation = 'co_occurs_with'
                  AND src_key IN ({placeholders})
                UNION ALL
                SELECT src_key AS term_key, weight
                FROM edges
                WHERE relation = 'co_occurs_with'
                  AND dst_key IN ({placeholders})
            ) e
            JOIN nodes n ON n.node_key = e.term_key
            WHERE n.kind NOT IN ('session', 'message')
            GROUP BY e.term_key, n.kind, n.value
            ORDER BY weight DESC, n.value ASC
            LIMIT ?
            """,
            (*term_keys, *term_keys, limit),
        ).fetchall()
        return [f"{row['kind']}:{row['value']}" for row in rows]

    @staticmethod
    def _why(matched_topics: list[str], related_topics: list[str], direct: bool) -> str:
        if direct and matched_topics:
            return "Matched extracted topics: " + ", ".join(matched_topics[:5])
        if related_topics:
            return "Related through shared extracted topics: " + ", ".join(related_topics[:5])
        return "Related through shared extracted graph structure."


def format_graph_hits(hits: list[GraphHit], query: str) -> str:
    if not hits:
        return f'No graph results found for "{query}".'

    lines = [f'Found {len(hits)} graph result(s) matching "{query}":\n']
    for i, hit in enumerate(hits, 1):
        lines.extend([
            f"-- Graph Result {i} ----------------------------------------",
            f"Source : {hit.source.upper()}",
            f"Session: {hit.session_title}",
            f"Date   : {hit.session_date}",
            f"ID     : {hit.session_id[:36]}",
            f"Why    : {hit.why}",
        ])
        if hit.related_topics:
            lines.append("Related: " + ", ".join(hit.related_topics[:6]))
        lines.append("")
        if hit.question_text is not None:
            ts = f" {hit.question_ts}" if hit.question_ts else ""
            lines.append(f"[YOU ASKED]{ts}")
            lines.append(hit.question_text)
            lines.append("")
        if hit.answer_text is not None:
            ts = f" {hit.answer_ts}" if hit.answer_ts else ""
            lines.append(f"[ANSWER]{ts}")
            lines.append(hit.answer_text)
        lines.append("")
    return "\n".join(lines)
