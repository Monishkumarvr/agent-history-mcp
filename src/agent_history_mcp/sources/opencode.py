"""
OpenCode: SQLite store at ~/.local/share/opencode/opencode*.db

Tables session(id, title, directory, parent_id, time_created, time_updated),
message(id, session_id, time_created, data JSON) and part(id, message_id,
session_id, time_created, data JSON). Message data carries role, model and
tokens; part data is {type: "text" | "tool" | ..., text, tool, callID, state}.
The schema is introspected so missing columns degrade instead of failing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..model import SessionBuilder, output_text
from . import SourceAdapter, SourceUnit, open_readonly_sqlite, table_columns


def _json(value) -> dict:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _time(row: sqlite3.Row, data: dict, columns: set[str]):
    if "time_created" in columns and row["time_created"]:
        return row["time_created"]
    t = data.get("time")
    return t.get("created") if isinstance(t, dict) else None


def find_opencode_dbs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("opencode*.db") if p.is_file())


def parse_opencode_session(db_path: Path, session_id: str) -> dict | None:
    with open_readonly_sqlite(db_path) as conn:
        s_cols = table_columns(conn, "session")
        m_cols = table_columns(conn, "message")
        p_cols = table_columns(conn, "part")
        if not {"id", "session_id", "data"} <= m_cols:
            return None
        srow = conn.execute("SELECT * FROM session WHERE id = ?", (session_id,)).fetchone()
        if srow is None:
            return None

        builder = SessionBuilder("opencode", session_id, f"{db_path}#{session_id}")
        builder.title = srow["title"] if "title" in s_cols and srow["title"] else ""
        if "directory" in s_cols:
            builder.set_cwd(srow["directory"])

        order = "time_created, id" if "time_created" in m_cols else "id"
        messages = conn.execute(f"SELECT * FROM message WHERE session_id = ? ORDER BY {order}", (session_id,)).fetchall()
        parts_by_message: dict[str, list[tuple[sqlite3.Row, dict]]] = {}
        if {"message_id", "data"} <= p_cols:
            p_order = "time_created, id" if "time_created" in p_cols else "id"
            p_filter = "session_id = ?" if "session_id" in p_cols else "message_id IN (SELECT id FROM message WHERE session_id = ?)"
            for prow in conn.execute(f"SELECT * FROM part WHERE {p_filter} ORDER BY {p_order}", (session_id,)):
                parts_by_message.setdefault(prow["message_id"], []).append((prow, _json(prow["data"])))

        for mrow in messages:
            data = _json(mrow["data"])
            role = data.get("role", "")
            ts = _time(mrow, data, m_cols)
            if role == "assistant":
                model = data.get("modelID")
                if not model and isinstance(data.get("model"), dict):
                    model = data["model"].get("modelID")
                builder.add_model(model)
                tokens = data.get("tokens") or {}
                if isinstance(tokens, dict):
                    cache = tokens.get("cache")
                    cache_read = cache.get("read") or 0 if isinstance(cache, dict) else 0
                    builder.add_tokens((tokens.get("input") or 0) + cache_read, tokens.get("output") or 0)
                path_info = data.get("path") or {}
                if isinstance(path_info, dict):
                    builder.set_cwd(path_info.get("cwd"))

            texts: list[str] = []
            tool_parts: list[tuple[sqlite3.Row, dict]] = []
            for prow, pdata in parts_by_message.get(mrow["id"], []):
                ptype = pdata.get("type")
                if ptype == "text" and pdata.get("text") and not pdata.get("synthetic"):
                    texts.append(pdata["text"])
                elif ptype == "tool":
                    tool_parts.append((prow, pdata))
            if role in ("user", "assistant"):
                builder.add_message(role, "\n".join(texts), ts)
            for prow, pdata in tool_parts:
                state = pdata.get("state") or {}
                if not isinstance(state, dict):
                    state = {}
                call_id = str(pdata.get("callID") or prow["id"])
                part_ts = _time(prow, pdata, p_cols) or ts
                builder.add_tool_call(pdata.get("tool", ""), state.get("input") or {}, part_ts, call_id=call_id)
                status = str(state.get("status", ""))
                if status in ("completed", "error"):
                    output = state.get("output") if status == "completed" else state.get("error")
                    builder.add_tool_result(call_id, output_text(output), part_ts, is_error=True if status == "error" else None)
        return builder.build()


class OpenCodeAdapter(SourceAdapter):
    name = "opencode"

    def discover(self) -> list[SourceUnit]:
        if not self.root:
            return []
        units = []
        for db_path in find_opencode_dbs(self.root):
            try:
                with open_readonly_sqlite(db_path) as conn:
                    cols = table_columns(conn, "session")
                    if "id" not in cols:
                        self.diagnostics.append(f"{db_path.name}: no session table, skipped")
                        continue
                    token_col = "time_updated" if "time_updated" in cols else ("time_created" if "time_created" in cols else "id")
                    where = "WHERE parent_id IS NULL" if "parent_id" in cols else ""
                    for row in conn.execute(f"SELECT id, {token_col} AS token FROM session {where}"):
                        units.append(SourceUnit(
                            f"opencode://{db_path.resolve()}#{row['id']}", self.name, str(row["token"]), (db_path, row["id"]),
                        ))
            except sqlite3.Error as exc:
                self.diagnostics.append(f"{db_path.name}: unreadable ({exc})")
        return units

    def parse(self, unit: SourceUnit) -> dict | None:
        db_path, session_id = unit.locator
        return parse_opencode_session(db_path, session_id)
