"""
Synthetic history writers for every supported agent.

Each writer takes a list of steps:
  ("user", text) | ("assistant", text)
  ("bash", command, output, exit_code)
  ("edit", path) | ("write", path) | ("read", path)
  ("todo", [(text, status), ...]) | ("skill", name)
and writes that agent's native on-disk format.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _ts(start: str, i: int) -> str:
    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return (base + timedelta(minutes=i)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# Claude Code

def write_claude_session(root: Path, session_id: str, cwd: str, steps: list[tuple],
                         start: str = "2026-05-07T10:00:00Z", branch: str = "main",
                         title: str | None = None) -> Path:
    encoded = "".join(ch if ch.isalnum() else "-" for ch in cwd)
    path = root / ".claude" / "projects" / encoded / f"{session_id}.jsonl"
    common = {"cwd": cwd, "gitBranch": branch, "sessionId": session_id}
    for i, step in enumerate(steps):
        ts = _ts(start, i)
        kind = step[0]
        if kind == "user":
            _append(path, {**common, "type": "user", "timestamp": ts,
                           "message": {"role": "user", "content": step[1]}})
        elif kind == "assistant":
            _append(path, {**common, "type": "assistant", "timestamp": ts,
                           "message": {"id": f"msg{i}", "role": "assistant", "model": "claude-test",
                                       "content": [{"type": "text", "text": step[1]}],
                                       "usage": {"input_tokens": 100, "output_tokens": 10}}})
        else:
            name, tool_input, output, is_error = _claude_tool(step)
            call_id = f"toolu_{i}"
            _append(path, {**common, "type": "assistant", "timestamp": ts,
                           "message": {"id": f"msg{i}", "role": "assistant", "model": "claude-test",
                                       "content": [{"type": "tool_use", "id": call_id, "name": name, "input": tool_input}],
                                       "usage": {"input_tokens": 50, "output_tokens": 5}}})
            _append(path, {**common, "type": "user", "timestamp": ts,
                           "message": {"role": "user", "content": [
                               {"type": "tool_result", "tool_use_id": call_id, "content": output, "is_error": is_error}]}})
    if title:
        _append(path, {"type": "ai-title", "aiTitle": title, "sessionId": session_id})
    return path


def _claude_tool(step: tuple) -> tuple[str, dict, str, bool]:
    kind = step[0]
    if kind == "bash":
        _k, command, output, code = step
        text = output if code == 0 else f"Exit code {code}\n{output}"
        return "Bash", {"command": command}, text, code != 0
    if kind == "edit":
        return "Edit", {"file_path": step[1], "old_string": "a", "new_string": "b"}, "ok", False
    if kind == "write":
        return "Write", {"file_path": step[1], "content": "x"}, "ok", False
    if kind == "read":
        return "Read", {"file_path": step[1]}, "contents", False
    if kind == "todo":
        return "TodoWrite", {"todos": [{"content": t, "status": s, "activeForm": t} for t, s in step[1]]}, "ok", False
    if kind == "skill":
        return "Skill", {"skill": step[1]}, "loaded", False
    if kind == "rejected":
        return "Bash", {"command": step[1]}, "The user doesn't want to proceed with this tool use. The tool use was rejected.", True
    raise ValueError(kind)


# Codex CLI

def write_codex_session(root: Path, name: str, cwd: str, steps: list[tuple],
                        start: str = "2026-05-07T10:00:00Z", branch: str = "main") -> Path:
    day = start[8:10]
    path = root / ".codex" / "sessions" / start[:4] / start[5:7] / day / f"{name}.jsonl"
    _append(path, {"timestamp": _ts(start, 0), "type": "session_meta",
                   "payload": {"id": name, "cwd": cwd, "git": {"branch": branch}}})
    _append(path, {"timestamp": _ts(start, 0), "type": "turn_context", "payload": {"cwd": cwd, "model": "gpt-test"}})
    for i, step in enumerate(steps):
        ts = _ts(start, i)
        kind = step[0]
        call_id = f"call_{i}"
        if kind in ("user", "assistant"):
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "message", "role": kind, "content": [{"type": "input_text", "text": step[1]}]}})
        elif kind == "bash":
            _k, command, output, code = step
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "function_call", "name": "shell", "call_id": call_id,
                                       "arguments": json.dumps({"command": ["bash", "-lc", command]})}})
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "function_call_output", "call_id": call_id,
                                       "output": json.dumps({"output": output, "metadata": {"exit_code": code}})}})
        elif kind in ("edit", "write"):
            verb = "Update" if kind == "edit" else "Add"
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": call_id,
                                       "input": f"*** Begin Patch\n*** {verb} File: {step[1]}\n+x\n*** End Patch"}})
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "custom_tool_call_output", "call_id": call_id,
                                       "output": json.dumps({"output": "Success", "metadata": {"exit_code": 0}})}})
        elif kind == "todo":
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "function_call", "name": "update_plan", "call_id": call_id,
                                       "arguments": json.dumps({"plan": [{"step": t, "status": s} for t, s in step[1]]})}})
        elif kind == "read":
            _append(path, {"timestamp": ts, "type": "response_item",
                           "payload": {"type": "function_call", "name": "shell", "call_id": call_id,
                                       "arguments": json.dumps({"command": ["bash", "-lc", f"cat {step[1]}"]})}})
    _append(path, {"timestamp": _ts(start, len(steps)), "type": "event_msg",
                   "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1000, "output_tokens": 100}}}})
    return path


# Gemini CLI

def write_gemini_session(root: Path, session_id: str, cwd: str, steps: list[tuple],
                         start: str = "2026-05-07T10:00:00Z") -> Path:
    project_dir = root / ".gemini" / "tmp" / f"proj-{abs(hash(cwd)) % 10000}"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".project_root").write_text(cwd, encoding="utf-8")
    path = project_dir / "chats" / f"session-{session_id}.jsonl"
    _append(path, {"sessionId": session_id, "projectHash": "abc", "startTime": _ts(start, 0), "lastUpdated": _ts(start, 0), "kind": "main"})
    last_gemini: dict | None = None
    for i, step in enumerate(steps):
        ts = _ts(start, i)
        kind = step[0]
        if kind == "user":
            _append(path, {"id": f"m{i}", "timestamp": ts, "type": "user", "content": [{"text": step[1]}]})
            last_gemini = None
        elif kind == "assistant":
            last_gemini = {"id": f"m{i}", "timestamp": ts, "type": "gemini", "content": step[1], "model": "gemini-test",
                           "tokens": {"input": 10, "output": 2}, "toolCalls": []}
            _append(path, last_gemini)
        else:
            if last_gemini is None:
                last_gemini = {"id": f"m{i}", "timestamp": ts, "type": "gemini", "content": "", "model": "gemini-test", "toolCalls": []}
            name, args, output, status = _gemini_tool(step)
            last_gemini["toolCalls"].append({"id": f"tc{i}", "name": name, "args": args, "status": status, "timestamp": ts,
                                             "result": [{"functionResponse": {"id": f"tc{i}", "name": name, "response": {"output": output}}}]})
            _append(path, last_gemini)
    _append(path, {"$set": {"summary": f"Gemini {session_id}"}})
    return path


def _gemini_tool(step: tuple) -> tuple[str, dict, str, str]:
    kind = step[0]
    if kind == "bash":
        _k, command, output, code = step
        return "run_shell_command", {"command": command}, f"{output}\nExit Code: {code}", "success" if code == 0 else "error"
    if kind == "edit":
        return "replace", {"file_path": step[1]}, "ok", "success"
    if kind == "write":
        return "write_file", {"file_path": step[1]}, "ok", "success"
    if kind == "read":
        return "read_file", {"absolute_path": step[1]}, "contents", "success"
    if kind == "todo":
        return "write_todos", {"todos": [{"description": t, "status": s} for t, s in step[1]]}, "ok", "success"
    raise ValueError(kind)


# OpenCode

def write_opencode_db(root: Path, sessions: list[tuple[str, str, list[tuple]]], start: str = "2026-05-07T10:00:00Z") -> Path:
    db_path = root / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session (id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, directory TEXT,
            title TEXT, time_created INTEGER, time_updated INTEGER);
        CREATE TABLE IF NOT EXISTS message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
        CREATE TABLE IF NOT EXISTS part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
        """
    )
    base = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() * 1000)
    for session_id, cwd, steps in sessions:
        conn.execute("INSERT INTO session VALUES (?, 'p1', NULL, ?, ?, ?, ?)",
                     (session_id, cwd, f"OpenCode {session_id}", base, base + len(steps) * 60000))
        for i, step in enumerate(steps):
            t = base + i * 60000
            kind = step[0]
            mid = f"{session_id}-m{i}"
            role = "user" if kind == "user" else "assistant"
            data = {"role": role}
            if role == "assistant":
                data.update({"modelID": "oc-test", "tokens": {"input": 20, "output": 3, "cache": {"read": 5}}})
            conn.execute("INSERT INTO message VALUES (?, ?, ?, ?)", (mid, session_id, t, json.dumps(data)))
            if kind in ("user", "assistant"):
                part = {"type": "text", "text": step[1]}
            else:
                tool, tool_input, output, status = _opencode_tool(step)
                part = {"type": "tool", "tool": tool, "callID": f"c{i}",
                        "state": {"status": status, "input": tool_input,
                                  "output" if status == "completed" else "error": output}}
            conn.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?)", (f"{mid}-p", mid, session_id, t, json.dumps(part)))
    conn.commit()
    conn.close()
    return db_path


def _opencode_tool(step: tuple) -> tuple[str, dict, str, str]:
    kind = step[0]
    if kind == "bash":
        _k, command, output, code = step
        return "bash", {"command": command}, output, "completed" if code == 0 else "error"
    if kind == "edit":
        return "edit", {"filePath": step[1]}, "ok", "completed"
    if kind == "write":
        return "write", {"filePath": step[1]}, "ok", "completed"
    if kind == "read":
        return "read", {"filePath": step[1]}, "contents", "completed"
    if kind == "todo":
        return "todowrite", {"todos": [{"content": t, "status": s} for t, s in step[1]]}, "ok", "completed"
    raise ValueError(kind)


# Cursor

def write_cursor_store(root: Path, chats: list[tuple[str, str, list[tuple]]], start: str = "2026-05-07T10:00:00Z") -> Path:
    user_dir = root / "Cursor" / "User"
    gdb = user_dir / "globalStorage" / "state.vscdb"
    gdb.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(gdb)
    conn.execute("CREATE TABLE IF NOT EXISTS cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    base = int(datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp() * 1000)
    for n, (composer_id, cwd, steps) in enumerate(chats):
        headers = []
        for i, step in enumerate(steps):
            bubble_id = f"b{i}"
            kind = step[0]
            bubble = {"bubbleId": bubble_id, "type": 1 if kind == "user" else 2, "createdAt": _ts(start, i)}
            if kind in ("user", "assistant"):
                bubble["text"] = step[1]
            else:
                name, args, result, status = _cursor_tool(step)
                bubble["text"] = ""
                bubble["toolFormerData"] = {"name": name, "rawArgs": json.dumps(args), "result": json.dumps(result),
                                            "status": status, "toolCallId": f"t{i}"}
            if kind == "assistant":
                bubble["tokenCount"] = {"inputTokens": 30, "outputTokens": 4}
            headers.append({"bubbleId": bubble_id, "type": bubble["type"]})
            conn.execute("INSERT INTO cursorDiskKV VALUES (?, ?)", (f"bubbleId:{composer_id}:{bubble_id}", json.dumps(bubble).encode()))
        composer = {"_v": 14, "composerId": composer_id, "name": f"Cursor {composer_id}", "createdAt": base,
                    "lastUpdatedAt": base + len(steps) * 60000, "fullConversationHeadersOnly": headers}
        conn.execute("INSERT INTO cursorDiskKV VALUES (?, ?)", (f"composerData:{composer_id}", json.dumps(composer).encode()))

        ws = user_dir / "workspaceStorage" / f"ws{n}"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "workspace.json").write_text(json.dumps({"folder": f"file://{cwd}"}), encoding="utf-8")
        wconn = sqlite3.connect(ws / "state.vscdb")
        wconn.execute("CREATE TABLE IF NOT EXISTS ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
        wconn.execute("INSERT INTO ItemTable VALUES ('composer.composerData', ?)",
                      (json.dumps({"allComposers": [{"composerId": composer_id}]}),))
        wconn.commit()
        wconn.close()
    conn.commit()
    conn.close()
    return user_dir


def _cursor_tool(step: tuple) -> tuple[str, dict, dict, str]:
    kind = step[0]
    if kind == "bash":
        _k, command, output, code = step
        return "run_terminal_cmd", {"command": command}, {"output": output, "exitCode": code}, "completed" if code == 0 else "error"
    if kind == "edit":
        return "edit_file", {"target_file": step[1]}, {"ok": True}, "completed"
    if kind == "write":
        return "write", {"file_path": step[1]}, {"ok": True}, "completed"
    if kind == "read":
        return "read_file", {"target_file": step[1]}, {"contents": "x"}, "completed"
    if kind == "todo":
        return "todo_write", {"todos": [{"content": t, "status": s} for t, s in step[1]]}, {"ok": True}, "completed"
    raise ValueError(kind)


def all_adapters(root: Path):
    """Adapters pointed at fixture locations under root."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agent_history_mcp.sources.claude import ClaudeAdapter
    from agent_history_mcp.sources.codex import CodexAdapter
    from agent_history_mcp.sources.cursor import CursorAdapter
    from agent_history_mcp.sources.gemini import GeminiAdapter
    from agent_history_mcp.sources.opencode import OpenCodeAdapter

    return [
        ClaudeAdapter(root / ".claude"),
        CodexAdapter(root / ".codex"),
        GeminiAdapter(root / ".gemini"),
        OpenCodeAdapter(root / "opencode"),
        CursorAdapter(root / "Cursor" / "User"),
    ]
