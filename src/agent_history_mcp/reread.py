"""
Stop Claude re-reading a file it already has in context.

About half of Claude's Read calls fetch a path the session read before, with
no edit or compaction in between. The PreToolUse hook remembers each Read
(path, range, file mtime and size). When the same range of an unchanged file
is read again, it denies the call and says when it was read. Repeating the
Read lets it through, for when the earlier output really is gone from context.
Compaction clears the memory (PreCompact calls reset).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .model import agent_home


def _off_flag() -> Path:
    return agent_home() / "reread-guard-off"


def set_enabled(on: bool) -> str:
    """The switch: a flag file, so it takes effect on the next Read without touching agent settings."""
    flag = _off_flag()
    if on:
        flag.unlink(missing_ok=True)
    else:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
    return status()


def status() -> str:
    return "re-read guard: " + ("off" if _off_flag().exists() else "on")


def _state_file(session_id: str, agent_id: str = "") -> Path:
    # A subagent starts with an empty context, so it gets its own memory.
    name = re.sub(r"[^\w.-]", "_", f"{session_id}-{agent_id}" if agent_id else session_id)
    return agent_home() / "reads" / f"{name}.jsonl"


def reset(session_id: str) -> None:
    folder = agent_home() / "reads"
    if not session_id or not folder.is_dir():
        return
    stem = re.sub(r"[^\w.-]", "_", session_id)
    for path in folder.glob(f"{stem}*.jsonl"):  # the session and its subagents
        path.unlink(missing_ok=True)
    # ponytail: files of sessions that never compact stay behind (a few KB each); prune by age if they pile up.


def check(payload: dict) -> str:
    """Reason to deny this Read, or "" to let it run."""
    if payload.get("tool_name") != "Read" or not payload.get("session_id") or _off_flag().exists():
        return ""
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path")
    if not raw:
        return ""
    path = os.path.normcase(os.path.abspath(os.path.join(payload.get("cwd") or "", raw)))
    try:
        st = os.stat(path)
    except OSError:
        return ""  # let Read report the error
    key = [path, tool_input.get("offset"), tool_input.get("limit"), tool_input.get("pages")]
    state = _state_file(str(payload["session_id"]), str(payload.get("agent_id") or ""))
    last = None
    if state.is_file():
        for line in state.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("k") == key:
                last = entry
    unchanged = last and last["m"] == st.st_mtime_ns and last["s"] == st.st_size
    deny = bool(unchanged and not last.get("denied"))
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    entry = {"k": key, "m": st.st_mtime_ns, "s": st.st_size, "ts": last["ts"] if deny else now, "denied": deny}
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a", encoding="utf-8") as f:  # one short append per call, so parallel reads don't clobber
        f.write(json.dumps(entry) + "\n")
    if not deny:
        return ""
    return (
        f"You already read {raw} (same range) at {last['ts']} in this conversation, and the file has not changed "
        "since. Use that earlier output instead of reading it again. If it is no longer in your context, "
        "repeat the same Read and it will go through."
    )
