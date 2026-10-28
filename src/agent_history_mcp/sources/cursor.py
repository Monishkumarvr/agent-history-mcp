"""
Cursor IDE agent chats: <Cursor User dir>/globalStorage/state.vscdb

Table cursorDiskKV holds "composerData:<composerId>" (metadata with
fullConversationHeadersOnly, or an inline legacy "conversation") and
"bubbleId:<composerId>:<bubbleId>" rows (type 1 = user, 2 = assistant, with
text, toolFormerData, tokenCount). Workspace folders come from
workspaceStorage/*/workspace.json + ItemTable "composer.composerData".

The format is undocumented and versioned (_v), so parsing is best-effort and
read-only. These databases can be tens of GB, so bubbles are capped per chat.
The cursor-agent CLI store (~/.cursor/chats/**/store.db) is detected but not
parsed yet.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..model import SessionBuilder, output_text
from . import SourceAdapter, SourceUnit, open_readonly_sqlite

MAX_BUBBLES = 2000


def _json(value):
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _workspace_map(user_dir: Path) -> dict[str, str]:
    """composerId -> workspace folder."""
    mapping: dict[str, str] = {}
    storage = user_dir / "workspaceStorage"
    if not storage.is_dir():
        return mapping
    for ws_dir in storage.iterdir():
        meta = ws_dir / "workspace.json"
        db = ws_dir / "state.vscdb"
        if not meta.is_file() or not db.is_file():
            continue
        try:
            folder = json.loads(meta.read_text(encoding="utf-8")).get("folder", "")
            with open_readonly_sqlite(db) as conn:
                row = conn.execute("SELECT value FROM ItemTable WHERE key = 'composer.composerData'").fetchone()
        except (OSError, ValueError, sqlite3.Error):
            continue
        data = _json(row["value"]) if row else None
        for composer in (data or {}).get("allComposers", []) if isinstance(data, dict) else []:
            if isinstance(composer, dict) and composer.get("composerId") and folder:
                mapping[composer["composerId"]] = folder
    return mapping


def _bubble_text(bubble: dict) -> str:
    # richText is editor state; the plain "text" field carries the message.
    text = bubble.get("text")
    return text if isinstance(text, str) else ""


def parse_cursor_composer(db_path: Path, composer_id: str, cwd: str = "") -> dict | None:
    with open_readonly_sqlite(db_path) as conn:
        row = conn.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (f"composerData:{composer_id}",)).fetchone()
        composer = _json(row["value"]) if row else None
        if not isinstance(composer, dict):
            return None

        bubbles: list[dict] = []
        headers = composer.get("fullConversationHeadersOnly")
        if isinstance(headers, list) and headers:
            for header in headers[:MAX_BUBBLES]:
                bubble_id = header.get("bubbleId") if isinstance(header, dict) else None
                if not bubble_id:
                    continue
                brow = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?", (f"bubbleId:{composer_id}:{bubble_id}",)
                ).fetchone()
                bubble = _json(brow["value"]) if brow else None
                if isinstance(bubble, dict):
                    bubble.setdefault("type", header.get("type"))
                    bubbles.append(bubble)
        elif isinstance(composer.get("conversation"), list):
            bubbles = [b for b in composer["conversation"][:MAX_BUBBLES] if isinstance(b, dict)]

    if not bubbles:
        return None
    builder = SessionBuilder("cursor", composer_id, f"{db_path}#{composer_id}")
    builder.title = str(composer.get("name") or "")
    builder.set_cwd(cwd)
    default_ts = composer.get("createdAt")

    for bubble in bubbles:
        ts = bubble.get("createdAt") or (bubble.get("timingInfo") or {}).get("clientStartTime") or default_ts
        text = _bubble_text(bubble)
        if bubble.get("type") == 1:
            builder.add_message("user", text, ts)
            if not builder.title and text:
                builder.title = text.strip().splitlines()[0][:60]
            continue
        model_info = bubble.get("modelInfo")
        if isinstance(model_info, dict):
            builder.add_model(model_info.get("modelName"))
        tokens = bubble.get("tokenCount")
        if isinstance(tokens, dict):
            builder.add_tokens(tokens.get("inputTokens"), tokens.get("outputTokens"))
        builder.add_message("assistant", text, ts)
        tool = bubble.get("toolFormerData")
        if isinstance(tool, dict) and tool.get("name"):
            call_id = str(tool.get("toolCallId") or bubble.get("bubbleId") or "")
            args = tool.get("rawArgs") or tool.get("params") or {}
            builder.add_tool_call(tool["name"], args, ts, call_id=call_id)
            status = str(tool.get("status", "")).lower()
            if tool.get("result") is not None or status:
                result = _json(tool.get("result")) if isinstance(tool.get("result"), str) else tool.get("result")
                builder.add_tool_result(
                    call_id, output_text(result if result is not None else tool.get("result")), ts,
                    is_error=True if status in ("error", "failed", "cancelled") else None,
                )
    return builder.build()


class CursorAdapter(SourceAdapter):
    name = "cursor"

    def __init__(self, root):
        super().__init__(root)
        self._workspaces: dict[str, str] | None = None

    def _global_db(self) -> Path | None:
        if not self.root:
            return None
        if self.root.is_file():
            return self.root
        db = self.root / "globalStorage" / "state.vscdb"
        return db if db.is_file() else None

    def _count_cli_stores(self) -> None:
        chats = Path.home() / ".cursor" / "chats"
        if chats.is_dir():
            stores = list(chats.glob("*/*/store.db"))
            if stores:
                self.diagnostics.append(
                    f"found {len(stores)} cursor-agent CLI chat store(s); that format is not parsed yet"
                )

    def available(self) -> bool:
        return True

    def discover(self) -> list[SourceUnit]:
        self._count_cli_stores()
        db = self._global_db()
        if db is None:
            return []
        self._workspaces = _workspace_map(self.root) if self.root.is_dir() else {}
        units = []
        try:
            with open_readonly_sqlite(db) as conn:
                try:
                    rows = conn.execute(
                        """
                        SELECT key,
                               json_extract(CAST(value AS TEXT), '$.lastUpdatedAt') AS updated,
                               json_extract(CAST(value AS TEXT), '$.createdAt') AS created
                        FROM cursorDiskKV WHERE key LIKE 'composerData:%'
                        """
                    ).fetchall()
                    stamps = [(r["key"], r["updated"] or r["created"] or "") for r in rows]
                except sqlite3.Error:
                    stamps = []
                    for r in conn.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"):
                        data = _json(r["value"]) or {}
                        stamps.append((r["key"], data.get("lastUpdatedAt") or data.get("createdAt") or ""))
        except sqlite3.Error as exc:
            self.diagnostics.append(f"state.vscdb unreadable ({exc})")
            return []
        for key, stamp in stamps:
            composer_id = key.split(":", 1)[1]
            units.append(SourceUnit(f"cursor://{db.resolve()}#{composer_id}", self.name, str(stamp), (db, composer_id)))
        return units

    def parse(self, unit: SourceUnit) -> dict | None:
        db, composer_id = unit.locator
        if self._workspaces is None:
            self._workspaces = _workspace_map(self.root) if self.root and self.root.is_dir() else {}
        return parse_cursor_composer(db, composer_id, self._workspaces.get(composer_id, ""))
