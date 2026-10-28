"""
Instruction and memory bridge across agents.

Every agent reads different files: CLAUDE.md and Claude auto-memory, AGENTS.md
(Codex, Cursor, OpenCode), GEMINI.md, .cursor/rules/*.mdc, Codex memories. This
module shows who reads what, finds rules one agent sees but another doesn't
(drift), mines repeated user corrections into rule candidates, and syncs rules
into a managed block that is the only part of a file it ever edits.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .graph import HistoryGraphIndex, session_filters
from .model import norm_path

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:  # pragma: no cover - rapidfuzz is a declared dependency
    def token_set_ratio(a: str, b: str) -> float:
        return 100.0 if a == b else 0.0


AGENTS = ("claude", "codex", "gemini", "cursor", "opencode")
BLOCK_BEGIN = "<!-- agent-history:begin -->"
BLOCK_END = "<!-- agent-history:end -->"
BLOCK_HEADING = "## Shared agent rules (synced by agent-history-mcp)"
MATCH_THRESHOLD = 85


@dataclass
class MemoryFile:
    path: Path
    scope: str          # project | user
    kind: str           # instructions | memory
    readers: list[str]
    rules: list[str] = field(default_factory=list)


def claude_project_dir_name(project: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", project)


def _specs(project: Path, home: Path) -> list[tuple[str, str, str, list[str]]]:
    p, h = str(project), str(home)
    enc = claude_project_dir_name(norm_path(str(project)))
    return [
        (f"{p}/CLAUDE.md", "project", "instructions", ["claude"]),
        (f"{p}/.claude/CLAUDE.md", "project", "instructions", ["claude"]),
        (f"{p}/CLAUDE.local.md", "project", "instructions", ["claude"]),
        (f"{h}/.claude/CLAUDE.md", "user", "instructions", ["claude"]),
        (f"{h}/.claude/projects/{enc}/memory/*.md", "project", "memory", ["claude"]),
        (f"{p}/AGENTS.md", "project", "instructions", ["codex", "cursor", "opencode"]),
        (f"{p}/AGENTS.override.md", "project", "instructions", ["codex"]),
        (f"{h}/.codex/AGENTS.md", "user", "instructions", ["codex"]),
        (f"{h}/.codex/AGENTS.override.md", "user", "instructions", ["codex"]),
        (f"{h}/.codex/memories/*.md", "user", "memory", ["codex"]),
        (f"{p}/GEMINI.md", "project", "instructions", ["gemini"]),
        (f"{h}/.gemini/GEMINI.md", "user", "instructions", ["gemini"]),
        (f"{p}/.cursor/rules/*.mdc", "project", "instructions", ["cursor"]),
        (f"{p}/.cursorrules", "project", "instructions", ["cursor"]),
        (f"{h}/.config/opencode/AGENTS.md", "user", "instructions", ["opencode"]),
    ]


def _expand(pattern: str) -> list[Path]:
    if "*" not in pattern:
        path = Path(pattern)
        return [path] if path.is_file() else []
    base, _, glob = pattern.rpartition("/")
    base_path = Path(base)
    return sorted(p for p in base_path.glob(glob) if p.is_file()) if base_path.is_dir() else []


def _normalize_rule(text: str) -> str:
    text = re.sub(r"[*_`~]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip(" .;:")


def extract_rules(text: str) -> list[str]:
    """Bullet items and short standalone instruction lines from a markdown file."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    rules: list[str] = []
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith(("#", ">", "|", "<!--", "@")):
            continue
        match = re.match(r"^(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?(.*)$", line)
        rule = match.group(1).strip() if match else line
        if not match and (len(rule) > 300 or not re.search(r"[.!:]$|^(?:always|never|use|do|don'?t|prefer|avoid|run)\b", rule, re.IGNORECASE)):
            continue
        if re.match(r"^\[[^\]]+\]\([^)]+\)", rule):
            continue  # link-only index lines (e.g. MEMORY.md)
        if len(rule) >= 12 and len(rule.split()) >= 3:
            rules.append(rule)
    return rules


def discover_memory_files(project: str, home: Path | None = None) -> list[MemoryFile]:
    project_path = Path(project)
    home = home or Path.home()
    files: list[MemoryFile] = []
    seen: set[Path] = set()
    for pattern, scope, kind, readers in _specs(project_path, home):
        for path in _expand(pattern):
            resolved = path.resolve()
            if resolved in seen or (kind == "memory" and path.name == "MEMORY.md"):
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files.append(MemoryFile(path, scope, kind, list(readers), extract_rules(text)))

    # Claude Code reads AGENTS.md when CLAUDE.md imports it with @AGENTS.md.
    claude_imports = any(
        "claude" in f.readers and re.search(r"^@\.?/?AGENTS\.md\b", f.path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        for f in files if f.kind == "instructions"
    )
    if claude_imports:
        for f in files:
            if f.path.name == "AGENTS.md" and "claude" not in f.readers:
                f.readers.append("claude")
    return files


def _rule_present(rule: str, candidates: list[str], threshold: int = MATCH_THRESHOLD) -> bool:
    norm = _normalize_rule(rule)
    return any(token_set_ratio(norm, _normalize_rule(other)) >= threshold for other in candidates)


def _agents_used_in_project(index: HistoryGraphIndex | None, project: str) -> dict[str, int]:
    if index is None:
        return {}
    clauses, params = session_filters(None, project, None)
    rows = index.conn.execute(
        f"SELECT source, COUNT(*) AS n FROM sessions s WHERE {' AND '.join(clauses)} GROUP BY source", params,
    ).fetchall()
    return {row["source"]: row["n"] for row in rows}


@dataclass
class DriftItem:
    rule: str
    present_in: list[str]      # file labels
    missing_for: list[str]     # agents


def compute_drift(files: list[MemoryFile], agents: list[str], include_memory: bool = False) -> list[DriftItem]:
    visible: dict[str, list[str]] = {agent: [] for agent in agents}
    for f in files:
        for agent in f.readers:
            if agent in visible:
                visible[agent].extend(f.rules)

    items: list[DriftItem] = []
    seen_rules: list[str] = []
    for f in files:
        if f.kind == "memory" and not include_memory:
            continue
        for rule in f.rules:
            if _rule_present(rule, seen_rules, 95):
                continue
            seen_rules.append(rule)
            missing = [a for a in agents if not _rule_present(rule, visible[a])]
            if missing:
                holders = [_label(g) for g in files if _rule_present(rule, g.rules)]
                items.append(DriftItem(rule, holders, missing))
    return items


def _label(f: MemoryFile) -> str:
    home = str(Path.home())
    path = str(f.path)
    return "~" + path[len(home):] if path.startswith(home) else path


def list_agent_memory(index: HistoryGraphIndex | None, project: str, home: Path | None = None) -> str:
    files = discover_memory_files(project, home)
    used = _agents_used_in_project(index, project)
    lines = [f"# Agent instructions and memory for {project}", ""]
    if files:
        lines.append("## Files")
        for f in files:
            lines.append(f"- `{_label(f)}` ({f.scope} {f.kind}) read by: {', '.join(f.readers)} · {len(f.rules)} rule(s)")
    else:
        lines.append("No instruction or memory files found for this project or your user profile.")

    per_agent: dict[str, tuple[int, int]] = {}
    for agent in AGENTS:
        agent_files = [f for f in files if agent in f.readers]
        per_agent[agent] = (len(agent_files), sum(len(f.rules) for f in agent_files))
    lines.extend(["", "## Agents"])
    compared: list[str] = []
    for agent in AGENTS:
        n_files, n_rules = per_agent[agent]
        sessions = used.get(agent, 0)
        if n_files:
            compared.append(agent)
            usage = f", {sessions} session(s) here" if sessions else ""
            lines.append(f"- {agent}: {n_files} file(s), {n_rules} rule(s){usage}")
        elif sessions:
            compared.append(agent)
            lines.append(f"- {agent}: used here ({sessions} session(s)) but reads **no** instruction file")

    drift = compute_drift(files, compared) if len(compared) > 1 else []
    lines.extend(["", "## Drift (rules some agents see and others don't)"])
    if not drift:
        lines.append("None found." if len(compared) > 1 else "Need at least two agents in use to compare.")
    for item in drift[:30]:
        lines.append(f"- \"{item.rule[:160]}\" — in {', '.join(item.present_in) or 'n/a'} · missing for: {', '.join(item.missing_for)}")
    if len(drift) > 30:
        lines.append(f"... and {len(drift) - 30} more")
    if drift:
        lines.append("")
        lines.append("Preview a fix with sync_memory() (dry run; it only edits a marked agent-history block).")
    return "\n".join(lines)


# Correction mining

@dataclass
class RuleCandidate:
    rule: str
    kind: str                 # correction | preference | rejected
    sessions: list[tuple[str, str, str]]   # (source, session_id, date)
    examples: list[str]
    documented_in: list[str]


_LEADING_NOISE = re.compile(r"^(?:no+|nope|actually|wait|hmm|ok(?:ay)?|please|stop)[,.!:;\s-]+", re.IGNORECASE)


def _rule_from_text(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s|\n", text.strip(), maxsplit=1)[0]
    cleaned = _LEADING_NOISE.sub("", first).strip()
    cleaned = _LEADING_NOISE.sub("", cleaned).strip()
    return (cleaned[:1].upper() + cleaned[1:])[:200] if cleaned else first[:200]


def suggest_rules(
    index: HistoryGraphIndex,
    project: str | None,
    sources: list[str] | None = None,
    days_back: int | None = 90,
    min_sessions: int = 2,
    home: Path | None = None,
) -> list[RuleCandidate]:
    clauses, params = session_filters(sources, project, days_back)
    where = " AND ".join(clauses) or "1=1"
    rows = index.conn.execute(
        f"""
        SELECT s.source, s.session_id, s.session_date, e.kind, e.text, e.meta, e.cmd_sig, e.command
        FROM events e JOIN sessions s ON s.session_key = e.session_key
        WHERE {where}
          AND ((e.kind = 'user' AND (e.meta LIKE '%"correction"%' OR e.meta LIKE '%"preference"%'))
               OR (e.kind = 'tool_result' AND e.meta LIKE '%"rejected"%'))
        ORDER BY s.ended_at DESC, e.idx
        """,
        params,
    ).fetchall()

    clusters: list[dict] = []
    for row in rows:
        if row["kind"] == "tool_result":
            if not row["cmd_sig"]:
                continue
            kind, key_text = "rejected", f"Ask before running `{row['cmd_sig']}`"
        else:
            kind = "correction" if '"correction"' in (row["meta"] or "") else "preference"
            key_text = _rule_from_text(row["text"])
        norm = _normalize_rule(key_text)
        if len(norm) < 8:
            continue
        target = next((c for c in clusters if c["kind"] == kind and token_set_ratio(norm, c["norm"]) >= 72), None)
        if target is None:
            target = {"kind": kind, "norm": norm, "rule": key_text, "sessions": {}, "examples": [], "texts": []}
            clusters.append(target)
        target["texts"].append(key_text)
        target["sessions"][(row["source"], row["session_id"])] = row["session_date"]
        example = (row["text"] or row["command"] or "").strip()
        if example and len(target["examples"]) < 3 and example[:160] not in target["examples"]:
            target["examples"].append(example[:160])
        if len(key_text) < len(target["rule"]):
            target["rule"] = key_text

    documented_files = discover_memory_files(project, home) if project else []
    candidates: list[RuleCandidate] = []
    for cluster in clusters:
        if len(cluster["sessions"]) < max(1, min_sessions):
            continue
        documented = [
            _label(f) for f in documented_files
            if any(_rule_present(text, f.rules, 80) for text in cluster["texts"])
        ]
        candidates.append(RuleCandidate(
            rule=cluster["rule"],
            kind=cluster["kind"],
            sessions=[(src, sid, date) for (src, sid), date in cluster["sessions"].items()],
            examples=cluster["examples"],
            documented_in=documented,
        ))
    order = {"correction": 0, "rejected": 1, "preference": 2}
    candidates.sort(key=lambda c: (bool(c.documented_in), order[c.kind], -len(c.sessions)))
    return candidates


def format_rule_candidates(candidates: list[RuleCandidate]) -> str:
    if not candidates:
        return "No repeated corrections or stated preferences found. Try min_sessions=1 or a larger days_back."
    lines = [f"Found {len(candidates)} rule candidate(s) from your own corrections and preferences:", ""]
    for i, c in enumerate(candidates, 1):
        agents = sorted({src for src, _sid, _d in c.sessions})
        lines.append(f"{i}. {c.rule}")
        lines.append(f"   Kind: {c.kind} · {len(c.sessions)} session(s) · agents: {', '.join(agents)}")
        if c.examples:
            lines.append("   You said: " + " | ".join(f'"{e}"' for e in c.examples))
        lines.append("   Status: " + (f"already in {', '.join(c.documented_in)}" if c.documented_in else "not documented for any agent"))
        lines.append("")
    lines.append("Rewrite the rules you want to keep, then call sync_memory(rules=[...]) to preview adding them for every agent.")
    return "\n".join(lines).rstrip()


# Sync

def _target_files(project: str, scope: str, targets: list[str], home: Path) -> tuple[dict[Path, list[str]], list[str]]:
    p, h = Path(project), home
    if scope == "user":
        preferred = {
            "claude": h / ".claude" / "CLAUDE.md",
            "codex": h / ".codex" / "AGENTS.md",
            "gemini": h / ".gemini" / "GEMINI.md",
            "opencode": h / ".config" / "opencode" / "AGENTS.md",
        }
    else:
        preferred = {
            "claude": p / "CLAUDE.md",
            "codex": p / "AGENTS.md",
            "opencode": p / "AGENTS.md",
            "cursor": p / "AGENTS.md",
            "gemini": p / "GEMINI.md",
        }
    notes: list[str] = []
    files: dict[Path, list[str]] = defaultdict(list)
    for agent in targets:
        path = preferred.get(agent)
        if path is None:
            notes.append(f"{agent}: no {scope}-level instruction file to write; skipped.")
            continue
        files[path].append(agent)
    return dict(files), notes


def _split_block(text: str) -> tuple[str, list[str], str]:
    if BLOCK_BEGIN in text and BLOCK_END in text:
        before, rest = text.split(BLOCK_BEGIN, 1)
        block, after = rest.split(BLOCK_END, 1)
        rules = [m.group(1).strip() for m in re.finditer(r"^- (.+)$", block, re.MULTILINE)]
        return before, rules, after
    return text, [], ""


def _render(before: str, rules: list[str], after: str) -> str:
    block = "\n".join([BLOCK_BEGIN, BLOCK_HEADING, *[f"- {r}" for r in rules], BLOCK_END])
    if before and not before.endswith("\n\n"):
        before = before.rstrip("\n") + "\n\n"
    return before + block + (after if after.strip() else "\n")


def sync_memory(
    project: str,
    rules: list[str] | None = None,
    targets: list[str] | None = None,
    scope: str = "project",
    apply: bool = False,
    include_memory: bool = False,
    home: Path | None = None,
) -> str:
    home = home or Path.home()
    if scope not in ("project", "user"):
        return "scope must be 'project' or 'user'."
    files = discover_memory_files(project, home)
    if targets is None:
        # Default to agents that already read an instruction file here.
        targets = [a for a in AGENTS if any(a in f.readers for f in files if f.kind == "instructions")] or ["claude", "codex"]
    targets = [t for t in targets if t in AGENTS]

    if rules is None:
        drift = compute_drift(files, targets, include_memory=include_memory)
        wanted_by_agent = {agent: [d.rule for d in drift if agent in d.missing_for] for agent in targets}
        source_note = "rules missing for some agents (drift)"
    else:
        cleaned = [r.strip().lstrip("-* ").strip() for r in rules if r and r.strip()]
        wanted_by_agent = {agent: cleaned for agent in targets}
        source_note = f"{len(cleaned)} provided rule(s)"

    target_files, notes = _target_files(project, scope, targets, home)
    if scope == "project" and not Path(project).is_dir():
        return f"Project directory {project} does not exist on this machine; nothing to sync."

    outputs: list[str] = [f"sync_memory ({'APPLY' if apply else 'dry run'}) for {source_note}; scope={scope}", ""]
    changed = 0
    for path, agents in target_files.items():
        wanted: list[str] = []
        for agent in agents:
            for rule in wanted_by_agent.get(agent, []):
                if not _rule_present(rule, wanted, 95):
                    wanted.append(rule)
        old = path.read_text(encoding="utf-8") if path.is_file() else ""
        before, block_rules, after = _split_block(old)
        outside_rules = extract_rules(before + "\n" + after)
        merged = list(block_rules)
        for rule in wanted:
            if not _rule_present(rule, merged + outside_rules):
                merged.append(rule)
        if merged == block_rules:
            outputs.append(f"{path}: up to date ({', '.join(agents)}).")
            continue
        new = _render(before, merged, after)
        diff = difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=str(path) if old else "/dev/null", tofile=str(path),
        )
        outputs.append(f"{path} (read by {', '.join(agents)}):")
        outputs.append("".join(diff).rstrip())
        outputs.append("")
        changed += 1
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, encoding="utf-8")

    outputs.extend(notes)
    if changed and not apply:
        outputs.append("Dry run only. Re-run with apply=true to write these files (only the agent-history block changes).")
    elif changed:
        outputs.append(f"Wrote {changed} file(s).")
        if scope == "project":
            outputs.append("These files live in the repository; review before committing.")
    elif not notes:
        outputs.append("Nothing to change.")
    return "\n".join(outputs).rstrip()
