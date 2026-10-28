"""
Installed Agent Skills (SKILL.md) across agents: discovery, usage detection,
and opt-in installation.

Skills follow the open Agent Skills format (agentskills.io): a folder with a
SKILL.md whose YAML frontmatter has `name` and `description`. Claude Code reads
~/.claude/skills and <repo>/.claude/skills; Codex and other Agent Skills
clients read ~/.agents/skills and <repo>/.agents/skills (older Codex builds
use ~/.codex/skills).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .graph import HistoryGraphIndex, session_filters

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_PATH_RE = re.compile(r"skills/([A-Za-z0-9_.:-]+)/SKILL\.md")


@dataclass
class InstalledSkill:
    name: str
    description: str
    path: Path
    scope: str            # user | project | plugin
    agents: list[str]
    body: str


def _skill_roots(project: str | None, home: Path) -> list[tuple[Path, str, list[str]]]:
    roots = [
        (home / ".claude" / "skills", "user", ["claude"]),
        (home / ".agents" / "skills", "user", ["codex", "agent-skills clients"]),
        (home / ".codex" / "skills", "user", ["codex"]),
        (home / ".gemini" / "skills", "user", ["gemini"]),
        (home / ".config" / "opencode" / "skills", "user", ["opencode"]),
        (home / ".cursor" / "skills", "user", ["cursor"]),
    ]
    if project:
        p = Path(project)
        roots = [
            (p / ".claude" / "skills", "project", ["claude"]),
            (p / ".agents" / "skills", "project", ["codex", "agent-skills clients"]),
            (p / ".gemini" / "skills", "project", ["gemini"]),
            (p / ".opencode" / "skills", "project", ["opencode"]),
            (p / ".cursor" / "skills", "project", ["cursor"]),
            *roots,
        ]
    return roots


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        return {}, text
    data: dict[str, str] = {}
    key = None
    for line in match.group(1).splitlines():
        kv = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if kv and not line.startswith((" ", "\t")):
            key, value = kv.group(1), kv.group(2).strip()
            if value in (">", "|", ">-", "|-"):
                value = ""
            data[key] = value.strip("'\"")
        elif key and line.strip():
            data[key] = (data[key] + " " + line.strip()).strip()
    return data, match.group(2)


def discover_installed_skills(project: str | None = None, home: Path | None = None) -> list[InstalledSkill]:
    home = home or Path.home()
    found: list[InstalledSkill] = []
    seen: set[Path] = set()

    def add(path: Path, scope: str, agents: list[str]) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        meta, body = parse_frontmatter(text)
        found.append(InstalledSkill(
            name=meta.get("name") or path.parent.name,
            description=meta.get("description", ""),
            path=path, scope=scope, agents=agents, body=body,
        ))

    for root, scope, agents in _skill_roots(project, home):
        if root.is_dir():
            for skill_md in sorted(root.glob("*/SKILL.md")):
                add(skill_md, scope, agents)
    plugins = home / ".claude" / "plugins"
    if plugins.is_dir():
        for skill_md in sorted(plugins.glob("**/skills/*/SKILL.md")):
            add(skill_md, "plugin", ["claude"])
    return found


def skill_invocations(
    index: HistoryGraphIndex,
    days_back: int | None = None,
    project: str | None = None,
) -> dict[str, list[dict]]:
    """skill name -> sessions that used it (Skill tool, /slash command, or reading its SKILL.md)."""
    clauses, params = session_filters(None, project, days_back)
    where = " AND ".join(clauses) or "1=1"
    rows = index.conn.execute(
        f"""
        SELECT s.session_key, s.source, s.session_id, s.session_date, s.ended_at, e.kind, e.canon_tool,
               e.meta, e.paths, e.command
        FROM events e JOIN sessions s ON s.session_key = e.session_key
        WHERE {where}
          AND ((e.kind = 'tool_call' AND (e.canon_tool = 'skill' OR e.paths LIKE '%SKILL.md%' OR e.command LIKE '%SKILL.md%'))
               OR (e.kind = 'user' AND e.meta LIKE '%slash_command%'))
        """,
        params,
    ).fetchall()
    usage: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        meta = json.loads(row["meta"]) if row["meta"] else {}
        names: set[str] = set()
        if meta.get("skill"):
            names.add(meta["skill"])
        if meta.get("slash_command"):
            names.add(meta["slash_command"])
        for text in (row["paths"] or "", row["command"] or ""):
            names.update(_SKILL_PATH_RE.findall(text))
        for name in names:
            for variant in {name, name.split(":")[-1]}:
                usage[variant.lower()][row["session_key"]] = {
                    "session_key": row["session_key"], "source": row["source"], "session_id": row["session_id"],
                    "date": row["session_date"], "ended_at": row["ended_at"],
                }
    return {name: sorted(sessions.values(), key=lambda s: s["ended_at"]) for name, sessions in usage.items()}


def validate_skill_md(content: str) -> tuple[dict, list[str]]:
    meta, body = parse_frontmatter(content)
    errors: list[str] = []
    name = meta.get("name", "")
    if not name:
        errors.append("frontmatter is missing `name`")
    elif not NAME_RE.match(name) or len(name) > 64:
        errors.append("`name` must be lowercase letters, digits and single hyphens, at most 64 characters")
    description = meta.get("description", "")
    if not description:
        errors.append("frontmatter is missing `description`")
    elif len(description) > 1024:
        errors.append("`description` must be at most 1024 characters")
    if not body.strip():
        errors.append("SKILL.md has no instructions after the frontmatter")
    return meta, errors


def install_targets(name: str, targets: list[str], scope: str, project: str | None, home: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    notes: list[str] = []
    for target in targets:
        if scope == "project":
            if not project:
                notes.append("scope=project needs a project path.")
                break
            base = {"claude": Path(project) / ".claude" / "skills", "agents": Path(project) / ".agents" / "skills"}.get(target)
        else:
            base = {
                "claude": home / ".claude" / "skills",
                "agents": home / ".agents" / "skills",
                "codex": home / ".codex" / "skills",
            }.get(target)
        if base is None:
            notes.append(f"Unknown target '{target}' for scope={scope}; use claude, agents" + (", codex" if scope == "user" else "") + ".")
            continue
        paths.append(base / name / "SKILL.md")
    return paths, notes


def install_skill(
    index: HistoryGraphIndex | None,
    content: str,
    targets: list[str] | None = None,
    scope: str = "user",
    project: str | None = None,
    apply: bool = False,
    overwrite: bool = False,
    candidate_id: str | None = None,
    steps: list[str] | None = None,
    home: Path | None = None,
) -> str:
    home = home or Path.home()
    meta, errors = validate_skill_md(content)
    if errors:
        return "SKILL.md is not valid:\n" + "\n".join(f"- {e}" for e in errors)
    name = meta["name"]
    paths, notes = install_targets(name, targets or ["claude", "agents"], scope, project, home)
    if not content.endswith("\n"):
        content += "\n"

    lines = [f"install_skill '{name}' ({'APPLY' if apply else 'dry run'}, scope={scope})", ""]
    written = 0
    for path in paths:
        existing = path.read_text(encoding="utf-8") if path.is_file() else None
        if existing == content:
            lines.append(f"= {path} already has this exact SKILL.md")
            continue
        if existing is not None and not overwrite:
            diff = "".join(difflib.unified_diff(
                existing.splitlines(keepends=True), content.splitlines(keepends=True), str(path), str(path) + " (new)",
            ))
            lines.append(f"! {path} exists and differs; pass overwrite=true to replace it:\n{diff.rstrip()}")
            continue
        lines.append(f"{'+' if existing is None else '~'} {path}")
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written += 1
            if index is not None:
                index.conn.execute(
                    "INSERT OR REPLACE INTO installed_skills(name, path, sha256, candidate_id, steps, installed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (name, str(path), hashlib.sha256(content.encode("utf-8")).hexdigest(), candidate_id,
                     json.dumps(steps) if steps else None, time.time()),
                )
                index.conn.commit()
    lines.extend(notes)
    if not apply:
        preview = "\n".join(content.splitlines()[:40])
        lines.extend(["", "Content:", preview, "", "Dry run only. Re-run with apply=true to write the file(s)."])
    else:
        lines.append(f"\nWrote {written} file(s). Restart or reload the agent if it doesn't pick the skill up immediately.")
    return "\n".join(lines)


def installed_records(index: HistoryGraphIndex) -> dict[str, dict]:
    rows = index.conn.execute("SELECT * FROM installed_skills ORDER BY installed_at").fetchall()
    return {row["name"]: dict(row) for row in rows}
