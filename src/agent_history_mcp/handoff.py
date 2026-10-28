"""
Cross-agent handoff briefs and checkpoints.

A brief summarizes where the latest session(s) in a repository stopped: the
goal, open todos, files touched, commands and their outcomes, unresolved errors,
and dead ends worth not retrying. It is built deterministically from indexed
events, so any agent can pick up work another agent left behind.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .graph import HistoryGraphIndex
from .model import Event, agent_home, project_root
from .redact import redact

_DONE_STATUSES = {"completed", "complete", "done", "cancelled", "canceled"}
_CHECKPOINT_SPLIT = "\n---\n"


@dataclass
class SessionDigest:
    source: str
    session_id: str
    title: str
    started_at: str
    ended_at: str
    branch: str
    goal: str = ""
    latest_ask: str = ""
    status: str = ""
    files: list[tuple[str, int]] = field(default_factory=list)
    commands: list[tuple[str, bool, int | None]] = field(default_factory=list)
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    dead_ends: list[tuple[str, int, bool]] = field(default_factory=list)
    open_todos: list[str] = field(default_factory=list)
    done_todos: int = 0
    signals: Counter = field(default_factory=Counter)


def resolve_project(project: str | None) -> str:
    return project_root(project or os.getcwd())


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_prompt_text(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and not stripped.startswith("<")


def digest_session(row, events: list[Event]) -> SessionDigest:
    digest = SessionDigest(
        source=row["source"], session_id=row["session_id"], title=row["title"],
        started_at=row["started_at"], ended_at=row["ended_at"], branch=row["git_branch"],
    )
    asks = [e.text for e in events if e.kind == "user" and _is_prompt_text(e.text)]
    if asks:
        digest.goal = asks[0]
        digest.latest_ask = asks[-1] if len(asks) > 1 else ""
    replies = [e.text for e in events if e.kind == "assistant" and e.text.strip()]
    digest.status = replies[-1] if replies else ""

    file_last: dict[str, int] = {}
    file_count: Counter = Counter()
    for event in events:
        if event.kind == "tool_call" and event.canon_tool in ("edit", "write"):
            for path in event.paths:
                file_last[path] = event.idx
                file_count[path] += 1
    digest.files = [(p, file_count[p]) for p in sorted(file_last, key=lambda p: -file_last[p])]

    shell_results = [e for e in events if e.kind == "tool_result" and e.canon_tool == "shell" and e.command]
    latest_by_command: dict[str, Event] = {}
    for event in shell_results:
        latest_by_command[" ".join(event.command.split())] = event
    recent = sorted(latest_by_command.items(), key=lambda item: -item[1].idx)
    digest.commands = [(cmd, not ev.is_error, ev.exit_code) for cmd, ev in recent]

    last_by_sig: dict[str, Event] = {}
    for event in shell_results:
        last_by_sig[event.cmd_sig or event.command] = event
    digest.unresolved = [
        (ev.command, ev.text.strip().splitlines()[0] if ev.text.strip() else "")
        for ev in sorted(last_by_sig.values(), key=lambda e: -e.idx)
        if ev.is_error
    ]

    failures: Counter = Counter()
    last_failure_idx: dict[str, int] = {}
    for event in shell_results:
        if event.is_error:
            key = " ".join(event.command.split())
            failures[key] += 1
            last_failure_idx[key] = event.idx
    for cmd, count in failures.most_common():
        if count < 2:
            break
        sig = next((e.cmd_sig for e in shell_results if " ".join(e.command.split()) == cmd), "")
        recovered = any(
            not e.is_error and e.idx > last_failure_idx[cmd] and (e.cmd_sig == sig or " ".join(e.command.split()) == cmd)
            for e in shell_results
        )
        digest.dead_ends.append((cmd, count, recovered))

    plans = [e for e in events if e.kind == "tool_call" and e.meta.get("todos")]
    if plans:
        todos = plans[-1].meta["todos"]
        digest.open_todos = [t["text"] for t in todos if t.get("status", "").lower() not in _DONE_STATUSES]
        digest.done_todos = len(todos) - len(digest.open_todos)

    for event in events:
        signal = event.meta.get("signal")
        if signal:
            digest.signals[signal] += 1
    return digest


def _git_snapshot(project: str) -> list[str]:
    if not project or not Path(project).is_dir():
        return []
    lines: list[str] = []
    for args, label, limit in (
        (["status", "--short", "--branch"], "status", 15),
        (["log", "--oneline", "-5"], "recent commits", 5),
    ):
        try:
            out = subprocess.run(
                ["git", "-C", project, *args], capture_output=True, text=True, timeout=3, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return lines
        if out.returncode != 0:
            return lines
        body = [line for line in out.stdout.splitlines() if line.strip()][:limit]
        if body:
            lines.append(f"{label}:")
            lines.extend(f"    {line}" for line in body)
    return lines


# Checkpoints

def checkpoint_dir(project: str) -> Path:
    digest = hashlib.sha1(project.encode("utf-8")).hexdigest()[:12]
    return agent_home() / "checkpoints" / digest


def latest_checkpoint(project: str) -> tuple[Path, str] | None:
    directory = checkpoint_dir(project)
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.md"))
    if not files:
        return None
    return files[-1], files[-1].read_text(encoding="utf-8", errors="replace")


def _checkpoint_note(text: str) -> tuple[str, str]:
    header = text.split(_CHECKPOINT_SPLIT, 1)[0]
    saved = next((line.split(":", 1)[1].strip() for line in header.splitlines() if line.startswith("Saved:")), "")
    note_lines = header.split("Note:", 1)[1].strip() if "Note:" in header else ""
    return saved, note_lines


def save_checkpoint(index: HistoryGraphIndex, note: str, project: str | None = None,
                    sources: list[str] | None = None, apply: bool = False) -> str:
    root = resolve_project(project)
    now = datetime.now(timezone.utc)
    brief = handoff_brief(index, root, sources=sources, sessions=1, include_checkpoint=False)
    content = (
        f"# Checkpoint for {root}\n"
        f"Saved: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"Note: {note.strip()}\n"
        f"{_CHECKPOINT_SPLIT}{brief}\n"
    )
    directory = checkpoint_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.md"
    path.write_text(content, encoding="utf-8")
    lines = [f"Checkpoint saved to {path}"]
    repo_file = Path(root) / ".agent-context" / "HANDOFF.md"
    if apply:
        if Path(root).is_dir():
            repo_file.parent.mkdir(parents=True, exist_ok=True)
            repo_file.write_text(content, encoding="utf-8")
            lines.append(f"Also wrote {repo_file} (commit or .gitignore it as you prefer).")
        else:
            lines.append(f"Skipped {repo_file}: project directory does not exist on this machine.")
    else:
        lines.append(f"Pass apply=true to also write {repo_file} inside the repository.")
    lines.append("Any agent can load it with handoff_brief for this project.")
    return "\n".join(lines)


# Brief

def _display_path(path: str, root: str = "") -> str:
    if root and path.startswith(root.rstrip("/") + "/"):
        return path[len(root.rstrip("/")) + 1:]
    home = str(Path.home()).replace("\\", "/")
    return "~" + path[len(home):] if home and path.startswith(home) else path


def _format_digest(d: SessionDigest, compact: bool, root: str = "") -> list[str]:
    files_limit, cmd_limit = (6, 5) if compact else (12, 8)
    when = f"{d.started_at[:16].replace('T', ' ')} → {d.ended_at[11:16]}" if d.started_at else "unknown time"
    branch = f" · branch `{d.branch}`" if d.branch else ""
    lines = [f"## {d.source} · {d.title}", f"_{when} UTC{branch} · id `{d.session_id[:36]}`_", ""]
    if d.goal:
        lines.append(f"**Goal:** {_clip(d.goal, 200 if compact else 300)}")
    if d.latest_ask:
        lines.append(f"**Latest ask:** {_clip(d.latest_ask, 200 if compact else 300)}")
    if d.status:
        lines.append(f"**Where it stopped:** {_clip(d.status, 300 if compact else 500)}")
    if d.open_todos:
        lines.append(f"**Open todos** ({d.done_todos} done):")
        lines.extend(f"- [ ] {_clip(t, 160)}" for t in d.open_todos[:8])
    if d.files:
        shown = ", ".join(f"`{_display_path(p, root)}`" + (f" ×{n}" if n > 1 else "") for p, n in d.files[:files_limit])
        more = f" (+{len(d.files) - files_limit} more)" if len(d.files) > files_limit else ""
        lines.append(f"**Files touched:** {shown}{more}")
    if d.commands:
        lines.append("**Recent commands:**")
        for cmd, ok, code in d.commands[:cmd_limit]:
            mark = "ok" if ok else f"FAILED{f' (exit {code})' if code not in (None, 0) else ''}"
            lines.append(f"- {mark}: `{_clip(cmd, 140)}`")
    if d.unresolved:
        lines.append("**Unresolved errors:**")
        lines.extend(f"- `{_clip(cmd, 120)}` → {_clip(err, 160)}" for cmd, err in d.unresolved[:5])
    if d.dead_ends:
        lines.append("**Dead ends (failed repeatedly; don't retry blindly):**")
        lines.extend(
            f"- `{_clip(cmd, 120)}` failed {n}×" + (" (a later variant succeeded)" if recovered else "")
            for cmd, n, recovered in d.dead_ends[:5]
        )
    friction = [f"{n} {name}{'s' if n > 1 else ''}" for name, n in d.signals.items()
                if name in ("interrupt", "rejected", "correction", "compact")]
    if friction:
        lines.append(f"**Friction:** {', '.join(friction)}")
    lines.append("")
    return lines


def _next_step(digests: list[SessionDigest]) -> str:
    latest = digests[0]
    if latest.open_todos:
        return f"Continue with the first open todo: {_clip(latest.open_todos[0], 160)}"
    if latest.unresolved:
        return f"Resolve the failing command first: `{_clip(latest.unresolved[0][0], 120)}`"
    if latest.commands:
        return f"Re-run the last verification to confirm state: `{_clip(latest.commands[0][0], 120)}`"
    return "Confirm the goal with the user, then inspect the files touched above."


def handoff_brief(
    index: HistoryGraphIndex,
    project: str | None = None,
    sources: list[str] | None = None,
    sessions: int = 2,
    exclude_source: str | None = None,
    compact: bool = False,
    include_git: bool = True,
    include_checkpoint: bool = True,
) -> str:
    root = resolve_project(project)
    rows = [
        row for row in index.session_rows(sources, root, limit=max(1, sessions) * 4)
        if not exclude_source or row["source"] != exclude_source
    ][: max(1, sessions)]

    checkpoint = latest_checkpoint(root) if include_checkpoint else None
    if not rows and not checkpoint:
        scope = f" from sources other than {exclude_source}" if exclude_source else ""
        return f"No indexed sessions{scope} found for {_display_path(root)}."

    digests = [digest_session(row, index.events_for(row["session_key"])) for row in rows]
    lines = [f"# Handoff: {_display_path(root)}"]
    if digests:
        lines.append(
            f"From {len(digests)} recent session(s): "
            + ", ".join(f"{d.source} ({d.ended_at[:10] or 'unknown'})" for d in digests)
        )
    lines.append("")
    for digest in digests:
        lines.extend(_format_digest(digest, compact, root))

    if checkpoint:
        saved, note = _checkpoint_note(checkpoint[1])
        lines.extend([f"## Latest checkpoint ({saved or checkpoint[0].stem})", _clip(note, 600) or "(no note)", ""])

    if include_git:
        snapshot = _git_snapshot(root)
        if snapshot:
            lines.append("## Repository now")
            lines.extend(snapshot)
            lines.append("")

    if digests:
        lines.extend(["## Suggested next step", _next_step(digests)])
    return redact("\n".join(lines).rstrip())
