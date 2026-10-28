"""
Unified session/event model shared by every history source.

Adapters build sessions with SessionBuilder. A built session is a plain dict so
the existing search code keeps working:

  {session_id, session_title, session_date, source, messages: [...],
   cwd, project, git_branch, model, input_tokens, output_tokens,
   started_at, ended_at, tools_available, path, events: [Event, ...]}

`messages` keeps the original shape ({role, text, ts, session_id, ...});
`events` is the full ordered stream including tool calls and results.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .redact import redact


# Paths

def agent_home() -> Path:
    """Directory for derived data: graph DB, checkpoints, reports."""
    raw = os.environ.get("AGENT_HISTORY_HOME", "")
    return Path(raw).expanduser() if raw else Path.home() / ".agent-history-mcp"


def norm_path(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().replace("\\", "/")
    if value.startswith("file://"):
        value = value[len("file://"):]
        if re.match(r"^/[A-Za-z]:/", value):
            value = value[1:]
    if len(value) > 1:
        value = value.rstrip("/")
    if re.match(r"^[A-Za-z]:/", value):
        value = value[0].lower() + value[1:]
    return value


@lru_cache(maxsize=512)
def project_root(cwd: str | None) -> str:
    """Git root for cwd when it exists locally, otherwise the normalized cwd."""
    cwd = norm_path(cwd)
    if not cwd:
        return ""
    try:
        path = Path(cwd)
        if path.is_dir():
            for candidate in (path, *path.parents):
                if (candidate / ".git").exists():
                    return norm_path(str(candidate))
    except OSError:
        pass
    return cwd


# Time helpers

def parse_ts(ts_raw) -> datetime | None:
    if ts_raw is None or ts_raw == "":
        return None
    if isinstance(ts_raw, (int, float)):
        seconds = ts_raw / 1000 if ts_raw > 1e11 else ts_raw
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_ts(ts_raw) -> str:
    dt = parse_ts(ts_raw)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else ""


def fmt_ts(ts_raw) -> str:
    """Local display timestamp used in message dicts (matches the original parser)."""
    dt = parse_ts(ts_raw)
    if dt:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    return str(ts_raw)[:16] if ts_raw else ""


def local_date(ts_raw) -> str:
    dt = parse_ts(ts_raw)
    return dt.astimezone().strftime("%Y-%m-%d") if dt else ""


# Canonical tools

_CANON_TOOLS = {
    # shell
    "bash": "shell", "shell": "shell", "exec_command": "shell", "shell_command": "shell",
    "local_shell": "shell", "local_shell_call": "shell", "run_shell_command": "shell",
    "run_terminal_cmd": "shell", "run_terminal_command": "shell", "terminal": "shell",
    "powershell": "shell", "container.exec": "shell",
    # edit / write
    "edit": "edit", "multiedit": "edit", "apply_patch": "edit", "replace": "edit",
    "str_replace": "edit", "str_replace_editor": "edit", "edit_file": "edit",
    "search_replace": "edit", "notebookedit": "edit", "patch": "edit",
    "write": "write", "write_file": "write", "create_file": "write",
    # read / search
    "read": "read", "read_file": "read", "read_many_files": "read", "view": "read",
    "notebookread": "read", "cat": "read",
    "grep": "search", "glob": "search", "ls": "search", "list": "search",
    "list_directory": "search", "search_file_content": "search", "codebase_search": "search",
    "grep_search": "search", "file_search": "search", "list_dir": "search", "find": "search",
    "toolsearch": "search",
    # planning
    "todowrite": "plan", "todoread": "plan", "update_plan": "plan", "write_todos": "plan",
    "todo_write": "plan", "exitplanmode": "plan", "enterplanmode": "plan",
    # sub-agents and skills
    "task": "subagent", "agent": "subagent", "spawn_agent": "subagent",
    "skill": "skill",
    # web
    "webfetch": "web", "websearch": "web", "web_fetch": "web", "web_search": "web",
    "web_search_call": "web", "google_web_search": "web", "fetch": "web",
}


def canonical_tool(name: str | None, namespace: str | None = None) -> str:
    if not name:
        return "other"
    raw = name.strip()
    if raw.startswith("mcp__"):
        parts = raw.split("__")
        return f"mcp:{parts[1]}" if len(parts) > 2 else "mcp"
    lowered = raw.lower()
    if lowered in _CANON_TOOLS:
        return _CANON_TOOLS[lowered]
    if namespace:
        return f"mcp:{namespace}"
    if "__" in raw:
        return f"mcp:{raw.split('__', 1)[0]}"
    return "other"


# Commands

_MULTIPLEX_COMMANDS = {
    "apt", "apt-get", "az", "brew", "bun", "bundle", "cargo", "composer", "conda",
    "deno", "docker", "docker-compose", "dotnet", "flutter", "gcloud", "gh", "git",
    "go", "gradle", "helm", "kubectl", "make", "mvn", "npm", "npx", "pip", "pip3",
    "pnpm", "poetry", "rails", "rake", "systemctl", "terraform", "uv", "uvx", "yarn",
    "aws", "firebase", "fly", "flyctl", "heroku", "prisma", "alembic", "supabase", "twine",
    "vercel", "wrangler", "nx", "turbo", "bazel", "pulumi", "skaffold", "minikube", "kind",
}
_SHELL_WRAPPERS = {"bash", "sh", "zsh", "pwsh", "powershell", "powershell.exe", "cmd", "cmd.exe"}
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def unwrap_shell_command(command) -> str:
    """Turn ["bash", "-lc", "x"] or "bash -lc 'x'" into "x"."""
    if isinstance(command, list):
        parts = [str(p) for p in command]
        if len(parts) >= 3 and Path(parts[0]).name.lower() in _SHELL_WRAPPERS and parts[1] in ("-lc", "-c", "-Command", "/c"):
            return parts[2]
        return " ".join(parts)
    return str(command or "")


def command_signature(command: str | None) -> str:
    """Normalize a shell command to executable (+ subcommand) for pattern mining."""
    if not command:
        return ""
    text = command.strip()
    # Prefer the last segment of `cd x && real-command` chains.
    segments = [s.strip() for s in re.split(r"&&|;|\|\|", text) if s.strip()]
    meaningful = [s for s in segments if not re.match(r"^(cd|pushd|source|export|set)\b", s)]
    text = (meaningful or segments or [text])[0]
    text = text.split("|", 1)[0]
    tokens = text.replace("\n", " ").split()
    while tokens and (_ENV_ASSIGN_RE.match(tokens[0]) or tokens[0] in ("sudo", "time", "env", "nohup", "exec")):
        tokens.pop(0)
    if not tokens:
        return ""
    exe = Path(tokens[0].strip("'\"")).name.lower()
    if exe.endswith(".exe"):
        exe = exe[:-4]
    rest = [t for t in tokens[1:] if not t.startswith("-")]
    if exe in ("python", "python3", "py") and len(tokens) > 2 and tokens[1] == "-m":
        return f"python -m {tokens[2]}"
    if exe in _MULTIPLEX_COMMANDS and rest:
        sub = rest[0].strip("'\"")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9:_-]*", sub):
            if exe in ("npm", "pnpm", "yarn", "bun") and sub == "run" and len(rest) > 1:
                return f"{exe} run {rest[1]}"
            return f"{exe} {sub}"
    return exe


# User signals

_CORRECTION_START_RE = re.compile(
    r"^\s*(?:no\b|nope\b|don'?t\b|do not\b|stop\b|wrong\b|that'?s not\b|that is not\b|"
    r"not what\b|i said\b|i told you\b|instead\b|actually,|please don'?t\b|undo\b|revert\b|"
    r"why did you\b|you should(?:n'?t| not)? have\b)",
    re.IGNORECASE,
)
_CORRECTION_ANY_RE = re.compile(
    r"\b(?:i asked you to|you were supposed to|i already told you|you forgot to|"
    r"you shouldn'?t have|that broke)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:always|never|from now on|going forward|remember to|make sure to|prefer)\b",
    re.IGNORECASE,
)
_INTERRUPT_MARKERS = ("[Request interrupted by user", "[Request interrupted")
_REJECT_MARKERS = (
    "The user doesn't want to proceed with this tool use",
    "User rejected",
    "user denied",
    "The user rejected",
)


def detect_user_signal(text: str, after_agent_action: bool) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith(_INTERRUPT_MARKERS):
        return "interrupt"
    if stripped.startswith("<"):
        return None
    if after_agent_action and (_CORRECTION_START_RE.search(stripped[:200]) or _CORRECTION_ANY_RE.search(stripped[:400])):
        return "correction"
    if len(stripped) < 400 and _PREFERENCE_RE.search(stripped):
        return "preference"
    return None


def is_rejection(text: str) -> bool:
    return any(marker.lower() in (text or "")[:300].lower() for marker in _REJECT_MARKERS)


_SLASH_COMMAND_RE = re.compile(r"<command-name>/?([^<\s]+)</command-name>")
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)
_EXIT_CODE_RE = re.compile(r"(?:^|\n)\s*(?:Exit code|exit_code|Process exited with code)[:\s]+(-?\d+)", re.IGNORECASE)


def exit_code_from_text(text: str) -> int | None:
    match = _EXIT_CODE_RE.search(text or "")
    return int(match.group(1)) if match else None


# Event model

@dataclass
class Event:
    idx: int
    ts: str
    kind: str                      # user | assistant | tool_call | tool_result | system
    text: str = ""
    tool: str = ""
    canon_tool: str = ""
    command: str = ""
    cmd_sig: str = ""
    paths: list[str] = field(default_factory=list)
    is_error: bool = False
    exit_code: int | None = None
    call_id: str = ""
    meta: dict = field(default_factory=dict)

    def as_row(self) -> tuple:
        return (
            self.idx, self.ts, self.kind, self.text, self.tool, self.canon_tool,
            self.command, self.cmd_sig, json.dumps(self.paths) if self.paths else "",
            1 if self.is_error else 0, self.exit_code, self.call_id,
            json.dumps(self.meta) if self.meta else "",
        )


_TEXT_PREVIEW = 400
_TOOL_TEXT = 500


def _first(d: dict, *keys):
    for key in keys:
        value = d.get(key)
        if value:
            return value
    return None


def _json_obj(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_raw": value}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": value}
    return {}


def _todos(items) -> list[dict]:
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = _first(item, "content", "step", "description", "text", "title")
        if text:
            out.append({"text": str(text)[:200], "status": str(item.get("status", "") or "")})
    return out


def describe_tool_input(name: str, canon: str, tool_input) -> tuple[str, list[str], dict, str]:
    """Return (command, paths, meta, summary) for a tool call input."""
    data = _json_obj(tool_input)
    command = ""
    paths: list[str] = []
    meta: dict = {}

    if canon == "shell":
        raw = _first(data, "command", "cmd", "commandLine", "script")
        command = unwrap_shell_command(raw) if raw else str(data.get("_raw", ""))
    elif canon in ("edit", "write", "read"):
        path = _first(data, "file_path", "path", "filePath", "absolute_path", "notebook_path", "target_file", "filename")
        if path:
            paths.append(str(path))
        for item in data.get("paths", []) or data.get("files", []) or []:
            if isinstance(item, str):
                paths.append(item)
        raw = data.get("_raw") or data.get("input") or data.get("patch")
        if isinstance(raw, str) and "*** " in raw:
            paths.extend(m.strip() for m in _PATCH_FILE_RE.findall(raw))
    elif canon == "search":
        pattern = _first(data, "pattern", "query", "glob", "path")
        if pattern:
            meta["pattern"] = str(pattern)[:120]
    elif canon == "plan":
        todos = _todos(data.get("todos") or data.get("plan") or data.get("items"))
        if todos:
            meta["todos"] = todos
    elif canon == "skill":
        skill = _first(data, "skill", "name", "command")
        if skill:
            meta["skill"] = str(skill).lstrip("/")
    elif canon == "subagent":
        meta["subagent"] = str(_first(data, "subagent_type", "description", "agent") or "")[:80]
    elif canon == "web":
        target = _first(data, "url", "query", "q")
        if target:
            meta["target"] = str(target)[:200]

    if name and isinstance(tool_input, str) and "*** Begin Patch" in tool_input and not paths:
        paths.extend(m.strip() for m in _PATCH_FILE_RE.findall(tool_input))

    if command:
        summary = command
    elif paths:
        summary = ", ".join(paths[:3])
    elif meta:
        summary = json.dumps(meta)[:_TOOL_TEXT]
    else:
        summary = json.dumps(data)[:_TOOL_TEXT] if data else ""
    return redact(command)[:2000], [norm_path(p) for p in paths], meta, redact(summary)[:_TOOL_TEXT]


def output_text(output) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict):
                value = _first(item, "text", "output", "content")
                if isinstance(value, (dict, list)):
                    value = output_text(value)
                if value:
                    parts.append(str(value))
                elif "functionResponse" in item:
                    parts.append(output_text(item["functionResponse"].get("response")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(output, dict):
        value = _first(output, "output", "content", "text", "result", "error", "stdout")
        if value is not None:
            return output_text(value)
        return json.dumps(output)
    return str(output)


class SessionBuilder:
    """Accumulates messages and events for one session in arrival order."""

    def __init__(self, source: str, session_id: str, path: str = ""):
        self.source = source
        self.session_id = session_id
        self.path = path
        self.title = ""
        self.session_date = ""
        self.cwd = ""
        self.git_branch = ""
        self.models: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.first_ts = ""
        self.last_ts = ""
        self.tools_available = True
        self.messages: list[dict] = []
        self.events: list[Event] = []
        self._calls: dict[str, Event] = {}

    # metadata

    def set_cwd(self, cwd: str | None) -> None:
        if cwd and not self.cwd:
            self.cwd = norm_path(cwd)

    def set_branch(self, branch: str | None) -> None:
        if branch and not self.git_branch:
            self.git_branch = branch

    def add_model(self, model: str | None) -> None:
        if model and model not in self.models:
            self.models.append(model)

    def add_tokens(self, input_tokens=0, output_tokens=0) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def set_tokens(self, input_tokens=None, output_tokens=None) -> None:
        if input_tokens is not None:
            self.input_tokens = int(input_tokens)
        if output_tokens is not None:
            self.output_tokens = int(output_tokens)

    def _touch(self, ts) -> str:
        iso = iso_ts(ts)
        if iso:
            if not self.first_ts or iso < self.first_ts:
                self.first_ts = iso
            if not self.last_ts or iso > self.last_ts:
                self.last_ts = iso
        return iso

    def _append(self, event: Event) -> Event:
        event.idx = len(self.events)
        self.events.append(event)
        return event

    def _last_kind(self) -> str:
        for event in reversed(self.events):
            if event.kind != "system":
                return event.kind
        return ""

    # messages

    def add_message(self, role: str, text: str, ts=None) -> None:
        if not text:
            return
        iso = self._touch(ts)
        if role == "user":
            signal = detect_user_signal(text, self._last_kind() in ("assistant", "tool_call", "tool_result"))
            meta = {"signal": signal} if signal else {}
            command = _SLASH_COMMAND_RE.search(text)
            if command:
                meta["slash_command"] = command.group(1)
            if signal == "interrupt":
                self._append(Event(0, iso, "system", text[:200], meta={"signal": "interrupt"}))
                return
            self._append(Event(0, iso, "user", redact(text[:_TEXT_PREVIEW]), meta=meta))
        else:
            self._append(Event(0, iso, "assistant", redact(text[:_TEXT_PREVIEW])))
        self.messages.append({
            "role": role,
            "text": text,
            "ts": fmt_ts(ts),
            "session_id": self.session_id,
            "session_title": self.title,
            "session_date": self.session_date,
            "source": self.source,
        })

    def add_signal(self, signal: str, text: str = "", ts=None) -> None:
        iso = self._touch(ts)
        self._append(Event(0, iso, "system", text[:200], meta={"signal": signal}))

    # tools

    def add_tool_call(self, name: str, tool_input=None, ts=None, call_id: str = "", namespace: str | None = None, sidechain: bool = False) -> Event:
        iso = self._touch(ts)
        canon = canonical_tool(name, namespace)
        command, paths, meta, summary = describe_tool_input(name, canon, tool_input)
        if sidechain:
            meta["sidechain"] = True
        event = self._append(Event(
            0, iso, "tool_call", summary, tool=name or "", canon_tool=canon,
            command=command, cmd_sig=command_signature(command) if command else "",
            paths=paths, call_id=call_id or "", meta=meta,
        ))
        if call_id:
            self._calls[call_id] = event
        return event

    def add_tool_result(self, call_id: str = "", output=None, ts=None, is_error: bool | None = None, exit_code: int | None = None, name: str = "") -> Event:
        iso = self._touch(ts)
        text = output_text(output)
        if exit_code is None:
            exit_code = exit_code_from_text(text)
        meta: dict = {}
        if is_rejection(text):
            meta["signal"] = "rejected"
        if is_error is None:
            is_error = (exit_code is not None and exit_code != 0) or bool(meta)
        call = self._calls.get(call_id) if call_id else None
        tool = call.tool if call else name
        canon = call.canon_tool if call else canonical_tool(name)
        if call and call.meta.get("sidechain"):
            meta["sidechain"] = True
        return self._append(Event(
            0, iso, "tool_result", redact(text[:_TOOL_TEXT]), tool=tool, canon_tool=canon,
            command=call.command if call else "", cmd_sig=call.cmd_sig if call else "",
            paths=list(call.paths) if call else [], is_error=bool(is_error),
            exit_code=exit_code, call_id=call_id or "", meta=meta,
        ))

    def set_exit_code(self, call_id: str, exit_code: int | None) -> None:
        if not call_id or exit_code is None:
            return
        for event in reversed(self.events):
            if event.kind == "tool_result" and event.call_id == call_id:
                event.exit_code = exit_code
                event.is_error = exit_code != 0
                return

    # build

    def build(self) -> dict | None:
        if not self.messages and not any(e.kind == "tool_call" for e in self.events):
            return None
        title = self.title or self.session_id[:60]
        date = self.session_date or local_date(self.first_ts) or "unknown"
        for msg in self.messages:
            msg["session_title"] = title
            msg["session_date"] = date
        return {
            "session_id": self.session_id,
            "session_title": title,
            "session_date": date,
            "source": self.source,
            "messages": self.messages,
            "events": self.events,
            "cwd": self.cwd,
            "project": project_root(self.cwd) if self.cwd else "",
            "git_branch": self.git_branch,
            "model": ",".join(self.models),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "started_at": self.first_ts,
            "ended_at": self.last_ts,
            "tools_available": self.tools_available,
            "path": self.path,
        }
