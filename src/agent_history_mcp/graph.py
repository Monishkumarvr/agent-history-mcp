"""
Persistent local graph index for Codex and Claude chat history.

This is intentionally local-first and deterministic: it extracts paths, commands,
errors, APIs, and technical topics with regex/token rules, then stores an
explainable SQLite graph that can be refreshed incrementally from JSONL history.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .parsers import (
    iter_claude_session_files,
    iter_codex_session_files,
    load_codex_title_index,
    parse_claude_session_file,
    parse_codex_session_file,
)


DEFAULT_DB_PATH = Path.home() / ".agent-history-mcp" / "history_graph.sqlite"


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


def _clean_value(value: str) -> str:
    return value.strip().strip(".,:;!?)]}\"'")


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", _clean_value(value).replace("\\", "/").lower())


def _add_term(terms: dict[tuple[str, str], GraphTerm], kind: str, value: str, weight: float) -> None:
    value = _clean_value(value)
    norm = _norm(value)
    if len(norm) < 3:
        return
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
        signal = (
            any(ch.isdigit() for ch in raw)
            or any(ch in raw for ch in ("_", "-", "+", "#", "."))
            or raw.isupper()
            or len(norm) >= 4
        )
        if signal:
            _add_term(terms, "topic", raw, 1.0 if raw.islower() else 1.4)

    compact_words = [(raw, norm) for raw, norm in word_items if norm not in _STOP_WORDS]
    for size, weight in ((2, 2.0), (3, 2.4)):
        for i in range(0, max(0, len(compact_words) - size + 1)):
            chunk = compact_words[i:i + size]
            norms = [n for _r, n in chunk]
            if len(set(norms)) != len(norms):
                continue
            phrase = " ".join(raw for raw, _n in chunk)
            _add_term(terms, "topic", phrase, weight)

    return sorted(terms.values(), key=lambda t: (-t.weight, t.kind, t.norm))[:max_terms]


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
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                session_key TEXT,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_key TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                session_date TEXT NOT NULL,
                path TEXT NOT NULL
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

    def refresh(self, codex_path: Path | None, claude_path: Path | None) -> RefreshStats:
        stats = RefreshStats()
        discovered: list[tuple[Path, str, dict[str, str] | None]] = []

        if codex_path and codex_path.is_dir():
            title_index = load_codex_title_index(codex_path)
            discovered.extend((path, "codex", title_index) for path in iter_codex_session_files(codex_path))
        if claude_path and claude_path.is_dir():
            discovered.extend((path, "claude", None) for path in iter_claude_session_files(claude_path))

        discovered_by_path = {str(path.resolve()): (path, source, title_index) for path, source, title_index in discovered}
        stats.files_seen = len(discovered_by_path)

        known_paths = {
            row["path"]: row
            for row in self.conn.execute("SELECT path, session_key, mtime_ns, size FROM files").fetchall()
        }

        for path, row in known_paths.items():
            if path not in discovered_by_path:
                if row["session_key"]:
                    self._delete_session(row["session_key"])
                self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
                stats.files_removed += 1

        for path_key, (path, source, title_index) in discovered_by_path.items():
            stat = path.stat()
            known = known_paths.get(path_key)
            if known and known["mtime_ns"] == stat.st_mtime_ns and known["size"] == stat.st_size:
                stats.files_unchanged += 1
                continue

            if known and known["session_key"]:
                self._delete_session(known["session_key"])

            session = (
                parse_codex_session_file(path, title_index)
                if source == "codex"
                else parse_claude_session_file(path)
            )
            session_key = None
            if session:
                session_key = self._index_session(session, path_key)
                stats.sessions_indexed += 1

            self.conn.execute(
                """
                INSERT OR REPLACE INTO files(path, source, session_key, mtime_ns, size, sha256, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (path_key, source, session_key, stat.st_mtime_ns, stat.st_size, time.time()),
            )
            if known:
                stats.files_changed += 1
            else:
                stats.files_added += 1

        if stats.files_added or stats.files_changed or stats.files_removed:
            self._rebuild_related_sessions()

        self.conn.commit()
        return stats

    def _insert_node(self, node_key: str, kind: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO nodes(node_key, kind, value) VALUES (?, ?, ?)",
            (node_key, kind, value),
        )

    def _insert_edge(
        self,
        src_key: str,
        dst_key: str,
        relation: str,
        weight: float = 1.0,
        evidence: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO edges(src_key, dst_key, relation, weight, provenance, confidence, evidence)
            VALUES (?, ?, ?, ?, 'EXTRACTED', 1.0, ?)
            ON CONFLICT(src_key, dst_key, relation)
            DO UPDATE SET weight = excluded.weight, evidence = COALESCE(excluded.evidence, edges.evidence)
            """,
            (src_key, dst_key, relation, weight, evidence),
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
        self.conn.execute("DELETE FROM messages WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM sessions WHERE session_key = ?", (session_key,))
        self.conn.execute("DELETE FROM nodes WHERE node_key = ?", (session_key,))
        for message_key in message_keys:
            self.conn.execute("DELETE FROM nodes WHERE node_key = ?", (message_key,))

    def _index_session(self, session: dict, path: str) -> str:
        source = session["source"]
        session_id = session["session_id"]
        session_key = _session_key(source, session_id)
        self._insert_node(session_key, "session", session["session_title"])
        self.conn.execute(
            """
            INSERT OR REPLACE INTO sessions(session_key, session_id, source, title, session_date, path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_key, session_id, source, session["session_title"], session["session_date"], path),
        )

        session_term_weights: defaultdict[str, float] = defaultdict(float)
        message_terms_by_idx: dict[int, list[GraphTerm]] = {}

        for idx, msg in enumerate(session["messages"]):
            message_key = _message_key(source, session_id, idx)
            self._insert_node(message_key, "message", f"{source}:{session_id}:{idx}")
            self.conn.execute(
                """
                INSERT OR REPLACE INTO messages(message_key, session_key, msg_idx, role, text, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_key, session_key, idx, msg["role"], msg["text"], msg.get("ts", "")),
            )
            self._insert_edge(session_key, message_key, "contains_message", 1.0)

            terms = extract_graph_terms(msg["text"], max_terms=60)
            message_terms_by_idx[idx] = terms
            for term in terms:
                self._insert_node(term.key, term.kind, term.value)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO message_terms(message_key, term_key, weight)
                    VALUES (?, ?, ?)
                    """,
                    (message_key, term.key, term.weight),
                )
                session_term_weights[term.key] += term.weight
                self._insert_edge(message_key, term.key, f"mentions_{term.kind}", term.weight)

        for term_key, weight in session_term_weights.items():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO session_terms(session_key, term_key, weight)
                VALUES (?, ?, ?)
                """,
                (session_key, term_key, weight),
            )
            self._insert_edge(session_key, term_key, "mentions_topic", weight)

        self._index_qa_and_cooccurrence(session, message_terms_by_idx)
        return session_key

    def _index_qa_and_cooccurrence(self, session: dict, message_terms_by_idx: dict[int, list[GraphTerm]]) -> None:
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
            self._insert_edge(question_key, answer_key, "answered_by", 1.0)

            combined = {
                term.key: term
                for term in [*message_terms_by_idx.get(idx, []), *message_terms_by_idx.get(answer_idx, [])]
            }
            qa_terms = sorted(combined.values(), key=lambda t: (-t.weight, t.norm))[:30]
            for left_index, left in enumerate(qa_terms):
                for right in qa_terms[left_index + 1:]:
                    src, dst = sorted((left.key, right.key))
                    weight = min(left.weight, right.weight)
                    self._insert_edge(src, dst, "co_occurs_with", weight)

    def _rebuild_related_sessions(self) -> None:
        self.conn.execute("DELETE FROM edges WHERE relation = 'related_to'")
        self.conn.execute(
            """
            INSERT OR REPLACE INTO edges(src_key, dst_key, relation, weight, provenance, confidence, evidence)
            SELECT
                a.session_key,
                b.session_key,
                'related_to',
                SUM(min(a.weight, b.weight)) AS weight,
                'EXTRACTED',
                1.0,
                'shared extracted topics'
            FROM session_terms a
            JOIN session_terms b ON a.term_key = b.term_key AND a.session_key < b.session_key
            GROUP BY a.session_key, b.session_key
            HAVING COUNT(*) >= 2 AND SUM(min(a.weight, b.weight)) >= 4.0
            """
        )

    def search(self, query: str, sources: list[str], max_results: int = 5) -> list[GraphHit]:
        query_terms = extract_graph_terms(query, max_terms=25)
        if not query_terms:
            return []

        source_set = {s for s in sources if s in ("codex", "claude")}
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
            FROM edges e
            JOIN nodes n ON n.node_key = CASE WHEN e.src_key IN ({placeholders}) THEN e.dst_key ELSE e.src_key END
            WHERE e.relation = 'co_occurs_with'
              AND (e.src_key IN ({placeholders}) OR e.dst_key IN ({placeholders}))
              AND n.kind NOT IN ('session', 'message')
            GROUP BY n.node_key, n.kind, n.value
            ORDER BY weight DESC, n.value ASC
            LIMIT ?
            """,
            (*term_keys, *term_keys, *term_keys, limit),
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
