"""
History source adapters.

Each adapter discovers "units" (a JSONL file, or one session inside a SQLite
store) with a change token, and parses a unit into the unified session dict
built by model.SessionBuilder. Adapters must never modify the agent's history
and must fail soft on formats they do not recognize.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SOURCE_NAMES = ("claude", "codex", "gemini", "opencode", "cursor")


@dataclass
class SourceUnit:
    key: str
    source: str
    change_token: str
    locator: Any = None


class SourceAdapter:
    name = ""

    def __init__(self, root: Path | None):
        # Resolved once here, so discover() can key units by str(path) without a per-file resolve().
        self.root = Path(root).expanduser().resolve() if root else None
        self.diagnostics: list[str] = []

    def available(self) -> bool:
        return bool(self.root and self.root.exists())

    def discover(self) -> list[SourceUnit]:
        raise NotImplementedError

    def parse(self, unit: SourceUnit) -> dict | None:
        raise NotImplementedError


def read_jsonl(path: Path) -> list[dict]:
    try:
        out = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out
    except OSError:
        return []


def file_token(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@contextmanager
def open_readonly_sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    """Open another tool's SQLite store without writing to it."""
    uri = path.resolve().as_uri()
    conn = None
    try:
        conn = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=2)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
    except sqlite3.Error:
        if conn is not None:
            conn.close()
        conn = sqlite3.connect(f"{uri}?mode=ro&immutable=1", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def resolve_sources(sources) -> list[str]:
    """Accept "all", a single name, or a list; return valid source names."""
    if sources is None:
        return list(SOURCE_NAMES)
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",")]
    names: list[str] = []
    for source in sources:
        source = str(source).strip().lower()
        if source == "all":
            return list(SOURCE_NAMES)
        if source in SOURCE_NAMES and source not in names:
            names.append(source)
    return names


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "")
    return Path(raw).expanduser() if raw else default


def _default_cursor_user_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "Library" / "Application Support" / "Cursor" / "User",
        home / ".config" / "Cursor" / "User",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.insert(0, Path(appdata) / "Cursor" / "User")
    return next((c for c in candidates if c.exists()), candidates[-1])


def default_adapters() -> list[SourceAdapter]:
    """Adapters for every enabled source, configured from environment variables."""
    from .claude import ClaudeAdapter
    from .codex import CodexAdapter
    from .cursor import CursorAdapter
    from .gemini import GeminiAdapter
    from .opencode import OpenCodeAdapter

    enabled = resolve_sources(os.environ.get("AGENT_HISTORY_SOURCES") or "all")
    home = Path.home()
    adapters: list[SourceAdapter] = []
    if "claude" in enabled:
        adapters.append(ClaudeAdapter(_env_path("CLAUDE_PATH", home / ".claude")))
    if "codex" in enabled:
        adapters.append(CodexAdapter(_env_path("CODEX_PATH", home / ".codex")))
    if "gemini" in enabled:
        adapters.append(GeminiAdapter(_env_path("GEMINI_PATH", home / ".gemini")))
    if "opencode" in enabled:
        adapters.append(OpenCodeAdapter(_env_path("OPENCODE_DB", home / ".local" / "share" / "opencode")))
    if "cursor" in enabled:
        adapters.append(CursorAdapter(_env_path("CURSOR_PATH", _default_cursor_user_dir())))
    return adapters
