"""
Mine repeated multi-step workflows from tool-call sequences.

Each session becomes an "action skeleton": the shell commands (as signatures),
edits (by file type), MCP calls and skills it ran, with read/search noise and
consecutive repeats removed. Contiguous n-grams shared by several sessions are
candidate workflows. Ambient patterns (present almost everywhere) and patterns
subsumed by a longer one with the same sessions are dropped. Counting is
deterministic; no model is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .graph import HistoryGraphIndex, session_filters

# Shell commands that only inspect state; they are noise in a workflow skeleton.
_INSPECT_COMMANDS = {
    "cat", "ls", "head", "tail", "grep", "rg", "find", "pwd", "echo", "wc", "which", "tree",
    "less", "more", "sed", "awk", "stat", "file", "du", "df", "env", "printenv", "type",
    "git status", "git diff", "git log", "git show", "git branch", "git rev-parse", "true", "sleep",
}
_GENERIC_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "please", "help", "make", "add",
    "fix", "update", "change", "need", "want", "can", "you", "use", "using", "new", "get", "set",
    "run", "rollout", "session", "chat", "code", "file", "files", "work", "some", "what", "how",
    "why", "does", "should", "will", "would", "could", "about", "then", "also", "just", "into",
}


@dataclass
class WorkflowPattern:
    steps: tuple[str, ...]
    sessions: list[dict]
    examples: dict[int, Counter] = field(default_factory=dict)
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def sources(self) -> set[str]:
        return {s["source"] for s in self.sessions}

    @property
    def projects(self) -> set[str]:
        return {s["project"] for s in self.sessions if s["project"]}

    @property
    def scope(self) -> str:
        return "global" if len(self.projects) > 1 else "project"

    @property
    def candidate_id(self) -> str:
        digest = hashlib.sha1("\n".join(self.steps).encode("utf-8")).hexdigest()[:12]
        return f"wf-{digest}"


def step_token(canon_tool: str, cmd_sig: str, paths: list[str], meta: dict) -> str | None:
    if meta.get("sidechain"):
        return None
    if canon_tool == "shell":
        if not cmd_sig or cmd_sig in _INSPECT_COMMANDS:
            return None
        return f"shell:{cmd_sig}"
    if canon_tool in ("edit", "write"):
        suffix = PurePosixPath(paths[0]).suffix.lower() if paths else ""
        return f"edit:{suffix}" if suffix else "edit"
    if canon_tool.startswith("mcp:"):
        return canon_tool
    if canon_tool == "skill" and meta.get("skill"):
        return f"skill:{meta['skill']}"
    if canon_tool == "subagent":
        return "subagent"
    return None


def describe_step(step: str) -> str:
    kind, _, value = step.partition(":")
    if kind == "shell":
        return f"run `{value}`"
    if kind == "edit":
        return f"edit {value} files" if value else "edit files"
    if kind == "mcp":
        return f"call the {value} MCP server"
    if kind == "skill":
        return f"use the {value} skill"
    if kind == "subagent":
        return "delegate to a subagent"
    return step


@dataclass
class _Seq:
    row: dict
    tokens: list[str]
    events: list[dict]   # per token: the tool_call event row


def load_sequences(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    project: str | None = None,
    days_back: int | None = None,
) -> dict[str, _Seq]:
    clauses, params = session_filters(sources, project, days_back)
    clauses.append("s.tools_available = 1")
    where = " AND ".join(clauses)
    sessions = {
        row["session_key"]: dict(row)
        for row in index.conn.execute(f"SELECT * FROM sessions s WHERE {where}", params).fetchall()
    }
    seqs: dict[str, _Seq] = {}
    rows = index.conn.execute(
        f"""
        SELECT e.session_key, e.idx, e.kind, e.canon_tool, e.cmd_sig, e.command, e.paths, e.meta, e.is_error, e.text
        FROM events e JOIN sessions s ON s.session_key = e.session_key
        WHERE {where} AND e.kind IN ('tool_call', 'tool_result', 'user')
        ORDER BY e.session_key, e.idx
        """,
        params,
    )
    for ev in rows:
        key = ev["session_key"]
        seq = seqs.get(key)
        if seq is None:
            seq = seqs[key] = _Seq(sessions[key], [], [])
            seq.row["first_ask"] = ""
            seq.row["results"] = []
        if ev["kind"] == "user":
            if not seq.row["first_ask"] and ev["text"] and not ev["text"].lstrip().startswith("<"):
                seq.row["first_ask"] = ev["text"][:200]
            continue
        if ev["kind"] == "tool_result":
            if ev["canon_tool"] == "shell" and ev["cmd_sig"]:
                seq.row["results"].append(dict(ev))
            continue
        meta = json.loads(ev["meta"]) if ev["meta"] else {}
        paths = json.loads(ev["paths"]) if ev["paths"] else []
        token = step_token(ev["canon_tool"] or "", ev["cmd_sig"] or "", paths, meta)
        if token is None or (seq.tokens and seq.tokens[-1] == token):
            continue
        seq.tokens.append(token)
        seq.events.append(dict(ev))
    return {k: v for k, v in seqs.items() if len(v.tokens) >= 2}


def _contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def _actionable(gram: tuple[str, ...]) -> bool:
    return any(t.startswith(("shell:", "mcp:", "skill:")) for t in gram) and len(set(gram)) >= 2


def mine_workflows(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    project: str | None = None,
    days_back: int | None = None,
    min_sessions: int = 2,
    min_len: int = 3,
    max_len: int = 6,
    ambient_ratio: float | None = None,
    limit: int = 10,
) -> list[WorkflowPattern]:
    seqs = load_sequences(index, sources, project, days_back)
    support: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key, seq in seqs.items():
        tokens = seq.tokens
        grams: set[tuple[str, ...]] = set()
        for n in range(min_len, max_len + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if _actionable(gram):
                    grams.add(gram)
        for gram in grams:
            support[gram].add(key)

    total = len(seqs)
    min_sessions = max(2, min_sessions)
    # Patterns in most sessions are ambient habits, not skills. Small corpora get a looser cut.
    if ambient_ratio is None:
        ambient_ratio = 0.16 if total >= 50 else 0.5
    frequent = {
        g: s for g, s in support.items()
        if len(s) >= min_sessions and not (total >= 8 and len(s) / total > ambient_ratio)
    }

    # Keep closed patterns: drop a pattern when a longer one covers the same sessions.
    closed: list[tuple[tuple[str, ...], set[str]]] = []
    for gram, sessions in sorted(frequent.items(), key=lambda kv: (-len(kv[0]), -len(kv[1]))):
        if any(len(longer) > len(gram) and kept >= sessions and _contains(longer, gram) for longer, kept in closed):
            continue
        closed.append((gram, sessions))

    patterns: list[WorkflowPattern] = []
    for gram, keys in closed:
        session_rows = [seqs[k].row for k in keys]
        pattern = WorkflowPattern(
            steps=gram,
            sessions=[
                {"session_key": r["session_key"], "source": r["source"], "session_id": r["session_id"],
                 "title": r["title"], "date": r["session_date"], "project": r["project"]}
                for r in sorted(session_rows, key=lambda r: r["ended_at"], reverse=True)
            ],
        )
        pattern.score = (
            len(keys) * (1 + 0.15 * (len(gram) - min_len))
            * (1.3 if len(pattern.sources) > 1 else 1.0)
            * (1.2 if len(pattern.projects) > 1 else 1.0)
        )
        _attach_evidence(pattern, [seqs[k] for k in keys])
        patterns.append(pattern)

    patterns.sort(key=lambda p: (-p.score, p.steps))
    kept: list[WorkflowPattern] = []
    for pattern in patterns:
        keys = {s["session_key"] for s in pattern.sessions}
        duplicate = False
        for other in kept:
            other_keys = {s["session_key"] for s in other.sessions}
            jaccard = len(keys & other_keys) / len(keys | other_keys)
            shared = len(set(pattern.steps) & set(other.steps)) / len(set(pattern.steps))
            if jaccard >= 0.8 and shared >= 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(pattern)
        if len(kept) >= limit:
            break
    return kept


def _attach_evidence(pattern: WorkflowPattern, seqs: list[_Seq]) -> None:
    n = len(pattern.steps)
    shell_sigs = {s.split(":", 1)[1] for s in pattern.steps if s.startswith("shell:")}
    for seq in seqs:
        for i in range(len(seq.tokens) - n + 1):
            if tuple(seq.tokens[i:i + n]) != pattern.steps:
                continue
            for offset in range(n):
                ev = seq.events[i + offset]
                if ev["command"]:
                    first_line = ev["command"].strip().splitlines()[0][:160]
                    pattern.examples.setdefault(offset, Counter())[first_line] += 1
                elif ev["paths"]:
                    paths = json.loads(ev["paths"])
                    if paths:
                        pattern.examples.setdefault(offset, Counter())[PurePosixPath(paths[0]).name] += 1
            break
        results = seq.row.get("results", [])
        for j, res in enumerate(results):
            if not res["is_error"] or res["cmd_sig"] not in shell_sigs:
                continue
            error = res["text"].strip().splitlines()[-1][:160] if (res["text"] or "").strip() else "failed"
            fix = next(
                (r["command"].strip().splitlines()[0][:160] for r in results[j + 1:]
                 if not r["is_error"] and r["cmd_sig"] == res["cmd_sig"]),
                "",
            )
            item = (res["cmd_sig"], error, fix)
            if item not in pattern.failures and len(pattern.failures) < 5:
                pattern.failures.append(item)
        intent = seq.row.get("first_ask") or ""
        if intent:
            pattern.intents.append(intent)


def intent_keywords(texts: list[str], min_share: float = 0.5, limit: int = 4) -> list[str]:
    """Words shared by at least min_share of the texts (titles / first asks)."""
    if not texts:
        return []
    counts: Counter = Counter()
    for text in texts:
        words = {
            w for w in re.findall(r"[a-z][a-z0-9+#.-]{2,}", text.lower())
            if w not in _GENERIC_WORDS and not any(ch.isdigit() for ch in w)
        }
        counts.update(words)
    needed = max(2, math.ceil(len(texts) * min_share)) if len(texts) > 1 else 1
    return [w for w, c in counts.most_common() if c >= needed][:limit]
