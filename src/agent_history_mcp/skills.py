"""
Suggest, draft, and track reusable skills from the local chat graph.

Candidates come from two deterministic signals: recurring topic clusters in
the graph, and repeated multi-step tool workflows (workflows.py). Each one is
checked against already-installed skills (create / update / skip), can be
drafted as an Agent Skills SKILL.md, and installed skills are scored for
usage, missed triggers and drift. Nothing here calls a model or writes files.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .graph import HistoryGraphIndex
from .sources import SOURCE_NAMES


@dataclass
class SkillEvidence:
    session_id: str
    session_title: str
    session_date: str
    source: str


@dataclass
class SkillCandidate:
    candidate_id: str
    name: str
    slug: str
    description: str
    triggers: list[str]
    recurring_terms: list[str]
    evidence: list[SkillEvidence]
    confidence: float
    why: str
    boundaries: str
    kind: str = "topic"                      # topic | workflow
    steps: list[str] = field(default_factory=list)
    scope: str = ""
    examples: dict[int, list[str]] = field(default_factory=dict)
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    action: str = "create"                   # create | update | skip
    overlap_with: str = ""
    overlap_reason: str = ""


_GENERIC_TERMS = {
    "agent", "agents", "answer", "answers", "build", "chat", "chats", "check",
    "claude", "code", "codex", "command", "commands", "context", "current",
    "data", "debug", "error", "feature", "file", "files", "fix", "fixed",
    "generate", "github", "history", "implement", "issue", "local", "message",
    "messages", "mcp", "need", "output", "plan", "project", "query",
    "read", "request", "response", "result", "results", "search", "session",
    "sessions", "source", "test", "tests", "tool", "tools", "update",
    "user", "using", "work", "workflow",
    "access", "can", "download", "downloads", "monis", "users", "windows",
    "you", "already", "background", "does", "failed", "failure", "not",
}
_NOISY_PREFIXES = ("term:topic:",)
_ACTIONABLE_KINDS = {"api", "command", "error", "path"}
_TECH_HINTS = {
    "api", "app", "appservice", "auth", "azure", "cache", "cli", "cloudflare",
    "cuda", "database", "deploy", "deployment", "docker", "fastapi", "fixture",
    "github", "graph", "hmac", "http", "index", "jira", "json", "kusto",
    "migration", "mcp", "npm", "oauth", "pytest", "python", "redis", "sqlite",
    "sql", "terraform", "token", "typescript", "webapp",
}
_NOISY_PATH_PARTS = (
    "/.agent-history-mcp/",
    "/program files/",
    "/github.com/",
    "/mozilla/",
    "/users/",
    "/windows/",
    "codexsandboxoffline",
    "gh_",
    "powershell.exe",
)
_SOURCE_EXTENSIONS = (".c", ".cc", ".cpp", ".cs", ".cu", ".cuh", ".go", ".java", ".js", ".jsx", ".kt", ".py", ".rs", ".ts", ".tsx")


def suggest_skill_candidates(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    max_candidates: int = 5,
    min_sessions: int = 2,
    days_back: int | None = None,
    include_workflows: bool = False,
    project: str | None = None,
    installed: list | None = None,
) -> list[SkillCandidate]:
    candidates = _topic_candidates(index, sources, max_candidates, min_sessions, days_back)
    if include_workflows:
        workflows = workflow_candidates(index, sources, min_sessions, days_back, project, limit=max_candidates)
        candidates = [c for c in candidates if not _covered_by_workflow(c, workflows)]
        merged = {c.candidate_id: c for c in [*workflows, *candidates]}
        candidates = sorted(merged.values(), key=lambda c: (-c.confidence, c.kind != "workflow", c.slug))
    unique_slugs: dict[str, SkillCandidate] = {}
    for candidate in candidates:
        unique_slugs.setdefault(candidate.slug, candidate)
    candidates = list(unique_slugs.values())
    if installed is not None:
        annotate_overlap(candidates, installed)
        candidates.sort(key=lambda c: (c.action == "skip", -c.confidence))
    candidates = candidates[:max_candidates]
    _LAST_CANDIDATES.update({c.candidate_id: c for c in candidates})
    return candidates


def _covered_by_workflow(candidate: SkillCandidate, workflows: list[SkillCandidate]) -> bool:
    """A single-command topic is redundant when a workflow over the same sessions contains it."""
    from .model import command_signature

    commands = [t.split(":", 1)[1] for t in candidate.recurring_terms if t.startswith("command:")]
    if not commands:
        return False
    sessions = {(e.source, e.session_id) for e in candidate.evidence}
    for workflow in workflows:
        step_sigs = {s.split(":", 1)[1] for s in workflow.steps if s.startswith("shell:")}
        wf_sessions = {(e.source, e.session_id) for e in workflow.evidence}
        overlap = len(sessions & wf_sessions) / len(sessions) if sessions else 0
        if overlap >= 0.8 and all(command_signature(c) in step_sigs for c in commands):
            return True
    return False


def _topic_candidates(
    index: HistoryGraphIndex,
    sources: list[str] | None,
    max_candidates: int,
    min_sessions: int,
    days_back: int | None,
) -> list[SkillCandidate]:
    source_set = sorted({s for s in (sources or SOURCE_NAMES) if s in SOURCE_NAMES})
    if not source_set:
        return []
    min_sessions = max(1, min_sessions)
    max_candidates = max(1, max_candidates)

    rows = _candidate_term_rows(index, source_set, min_sessions, days_back)
    candidates = [
        _build_candidate(index, row, source_set, days_back)
        for row in rows
    ]
    candidates = [c for c in candidates if c is not None and len(c.evidence) >= min_sessions]
    candidates.sort(key=lambda c: (-c.confidence, -len(c.evidence), c.slug))
    deduped: list[SkillCandidate] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            continue
        seen_ids.add(candidate.candidate_id)
        deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break
    return deduped


def format_skill_candidates(candidates: list[SkillCandidate]) -> str:
    if not candidates:
        return (
            "No reusable skill candidates found yet. Try again after more related "
            "Codex or Claude sessions, or lower min_sessions."
        )

    lines = [f"Found {len(candidates)} skill candidate(s):\n"]
    for idx, candidate in enumerate(candidates, 1):
        sources = sorted({item.source for item in candidate.evidence})
        evidence = ", ".join(
            f"{item.source}:{item.session_id[:12]} ({item.session_date or 'unknown'})"
            for item in candidate.evidence[:5]
        )
        lines.extend([
            f"{idx}. {candidate.name}",
            f"   ID: {candidate.candidate_id}",
            f"   Slug: {candidate.slug}",
            f"   Kind: {candidate.kind}" + (f" · scope: {candidate.scope}" if candidate.scope else ""),
            f"   Confidence: {candidate.confidence:.2f}",
            f"   Description: {candidate.description}",
        ])
        if candidate.steps:
            lines.append(f"   Steps: {' -> '.join(candidate.steps)}")
        lines.extend([
            f"   When to use: {', '.join(candidate.triggers[:6])}",
            f"   Recurring terms: {', '.join(candidate.recurring_terms[:8])}",
            f"   Evidence: {len(candidate.evidence)} sessions, {len(sources)} source(s)",
            f"   Sessions: {evidence}",
            f"   Why: {candidate.why}",
            f"   Boundaries: {candidate.boundaries}",
        ])
        if candidate.overlap_reason:
            lines.append(f"   Action: {candidate.action}" + (f" `{candidate.overlap_with}`" if candidate.overlap_with else "") + f" ({candidate.overlap_reason})")
        lines.append("")
    if any(c.action != "skip" for c in candidates):
        lines.append("Next: draft_skill(candidate_id) builds a SKILL.md you can review and install.")
    return "\n".join(lines).rstrip()


def _candidate_term_rows(
    index: HistoryGraphIndex,
    source_set: set[str],
    min_sessions: int,
    days_back: int | None,
) -> list:
    date_filter, params = _date_filter(days_back)
    source_placeholders = ",".join("?" for _ in source_set)
    rows = index.conn.execute(
        f"""
        SELECT
            n.node_key AS term_key,
            n.kind AS kind,
            n.value AS value,
            COUNT(DISTINCT s.session_key) AS session_count,
            COUNT(DISTINCT s.source) AS source_count,
            SUM(st.weight) AS total_weight,
            MAX(st.weight) AS max_weight
        FROM session_terms st
        JOIN nodes n ON n.node_key = st.term_key
        JOIN sessions s ON s.session_key = st.session_key
        WHERE s.source IN ({source_placeholders})
          AND n.kind NOT IN ('session', 'message')
          {date_filter}
        GROUP BY n.node_key, n.kind, n.value
        HAVING COUNT(DISTINCT s.session_key) >= ?
        ORDER BY
            CASE
                WHEN n.kind IN ('command', 'error', 'api') THEN 0
                WHEN n.kind = 'path' THEN 1
                ELSE 2
            END,
            COUNT(DISTINCT s.session_key) DESC,
            COUNT(DISTINCT s.source) DESC,
            SUM(st.weight) DESC
        LIMIT 80
        """,
        (*source_set, *params, min_sessions),
    ).fetchall()
    return [row for row in rows if _is_candidate_seed(row["kind"], row["value"], row["term_key"])]


def _build_candidate(
    index: HistoryGraphIndex,
    seed_row,
    source_set: set[str],
    days_back: int | None,
) -> SkillCandidate | None:
    seed_key = seed_row["term_key"]
    related_terms = _related_terms(index, seed_key)
    cluster_keys = [seed_key, *[term["term_key"] for term in related_terms[:8]]]
    evidence = _evidence_sessions(index, cluster_keys, source_set, days_back)
    if not evidence:
        return None

    recurring_terms = _recurring_terms(index, cluster_keys, source_set, days_back)
    if len(recurring_terms) < 2:
        recurring_terms = [_label(seed_row["kind"], seed_row["value"])]

    name = _candidate_name(seed_row["kind"], seed_row["value"], recurring_terms)
    slug = _slugify(name)
    candidate_id = _candidate_id(cluster_keys)
    triggers = _trigger_phrases(seed_row["kind"], seed_row["value"], recurring_terms)
    confidence = _confidence(seed_row, evidence, recurring_terms)
    description = _description(name, recurring_terms)
    why = _why(seed_row, evidence, recurring_terms)
    boundaries = _boundaries(seed_row["kind"], seed_row["value"])

    return SkillCandidate(
        candidate_id=candidate_id,
        name=name,
        slug=slug,
        description=description,
        triggers=triggers,
        recurring_terms=recurring_terms,
        evidence=evidence,
        confidence=confidence,
        why=why,
        boundaries=boundaries,
    )


def _related_terms(index: HistoryGraphIndex, seed_key: str) -> list:
    rows = index.conn.execute(
        """
        SELECT
            n.node_key AS term_key,
            n.kind AS kind,
            n.value AS value,
            MAX(e.weight) AS weight
        FROM edges e
        JOIN nodes n ON n.node_key = CASE WHEN e.src_key = ? THEN e.dst_key ELSE e.src_key END
        WHERE e.relation = 'co_occurs_with'
          AND (e.src_key = ? OR e.dst_key = ?)
          AND n.kind NOT IN ('session', 'message')
        GROUP BY n.node_key, n.kind, n.value
        ORDER BY weight DESC, n.value ASC
        LIMIT 20
        """,
        (seed_key, seed_key, seed_key),
    ).fetchall()
    return [row for row in rows if _is_candidate_seed(row["kind"], row["value"], row["term_key"])]


def _evidence_sessions(
    index: HistoryGraphIndex,
    term_keys: list[str],
    source_set: set[str],
    days_back: int | None,
    limit: int = 8,
) -> list[SkillEvidence]:
    date_filter, params = _date_filter(days_back)
    term_placeholders = ",".join("?" for _ in term_keys)
    source_placeholders = ",".join("?" for _ in source_set)
    rows = index.conn.execute(
        f"""
        SELECT
            s.session_id,
            s.title,
            s.session_date,
            s.source,
            SUM(st.weight) AS score
        FROM session_terms st
        JOIN sessions s ON s.session_key = st.session_key
        WHERE st.term_key IN ({term_placeholders})
          AND s.source IN ({source_placeholders})
          {date_filter}
        GROUP BY s.session_key, s.session_id, s.title, s.session_date, s.source
        ORDER BY score DESC, s.session_date DESC
        LIMIT ?
        """,
        (*term_keys, *source_set, *params, limit),
    ).fetchall()
    return [
        SkillEvidence(
            session_id=row["session_id"],
            session_title=row["title"],
            session_date=row["session_date"],
            source=row["source"],
        )
        for row in rows
    ]


def _recurring_terms(
    index: HistoryGraphIndex,
    term_keys: list[str],
    source_set: set[str],
    days_back: int | None,
) -> list[str]:
    date_filter, params = _date_filter(days_back)
    term_placeholders = ",".join("?" for _ in term_keys)
    source_placeholders = ",".join("?" for _ in source_set)
    rows = index.conn.execute(
        f"""
        SELECT n.kind, n.value, COUNT(DISTINCT s.session_key) AS sessions, SUM(st.weight) AS weight
        FROM session_terms st
        JOIN nodes n ON n.node_key = st.term_key
        JOIN sessions s ON s.session_key = st.session_key
        WHERE st.term_key IN ({term_placeholders})
          AND s.source IN ({source_placeholders})
          {date_filter}
        GROUP BY n.node_key, n.kind, n.value
        ORDER BY sessions DESC, weight DESC, n.value ASC
        LIMIT 12
        """,
        (*term_keys, *source_set, *params),
    ).fetchall()
    return [_label(row["kind"], row["value"]) for row in rows if _is_candidate_seed(row["kind"], row["value"], "")]


def _date_filter(days_back: int | None) -> tuple[str, list[str]]:
    if days_back is None or days_back <= 0:
        return "", []
    cutoff = date.today() - timedelta(days=days_back)
    return "AND s.session_date GLOB '????-??-??' AND s.session_date >= ?", [cutoff.isoformat()]


def _is_candidate_seed(kind: str, value: str, term_key: str) -> bool:
    norm = _normalize(value)
    if not norm or len(norm) < 4:
        return False
    if norm in _GENERIC_TERMS:
        return False
    if kind == "api" and _looks_like_domain(norm):
        return False
    if kind == "api" and re.search(r"\.(?:cmd|csv|css|html|lock|npy|py|ps1|js|ts|json|md|toml|yaml|yml|sqlite|exe|zip)$", norm):
        return False
    if kind == "api" and norm in {"date.now", "math.max", "math.min"}:
        return False
    if kind == "error" and _looks_like_noisy_error(norm):
        return False
    if kind == "path":
        if _looks_like_environment_path(norm):
            return False
        if not norm.endswith(_SOURCE_EXTENSIONS):
            return False
    if kind == "topic":
        words = norm.split()
        if len(words) == 1 and words[0] in _GENERIC_TERMS:
            return False
        if len(words) > 4:
            return False
        generic_count = sum(1 for word in words if word in _GENERIC_TERMS)
        if generic_count >= max(1, len(words) - 1):
            return False
        if term_key.startswith(_NOISY_PREFIXES) and len(words) == 1 and len(norm) < 5:
            return False
        if not _is_technical_topic(value, words):
            return False
    return True


def _looks_like_domain(norm: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9.-]+\.(?:com|org|net|io|dev|ai|app)", norm))


def _looks_like_environment_path(norm: str) -> bool:
    norm = norm.replace("\\", "/")
    if any(part in norm for part in _NOISY_PATH_PARTS):
        return True
    if norm.endswith((".exe", ".sqlite", ".zip")):
        return True
    if re.fullmatch(r"[a-z0-9-]+\.[a-z]", norm):
        return True
    return False


def _looks_like_noisy_error(norm: str) -> bool:
    if norm in {"failed", "failure", "permission denied", "access denied"}:
        return True
    if "sandbox" in norm or "codexsandboxoffline" in norm:
        return True
    if len(norm) > 140:
        return True
    return False


def _is_technical_topic(value: str, words: list[str]) -> bool:
    if any(word in _TECH_HINTS for word in words):
        return True
    if any(any(ch.isdigit() for ch in word) or any(ch in word for ch in ("_", "-", "+", "#", ".")) for word in words):
        return True
    if any(token.isupper() and len(token) > 1 for token in value.split()):
        return True
    return False


def _label(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("\\", "/")).strip(".,:;!?)]}\"'")


def _candidate_name(kind: str, value: str, recurring_terms: list[str]) -> str:
    clean = _humanize(value)
    if kind == "error":
        return f"{clean} Troubleshooting"
    if kind == "command":
        command = clean.split()[0] if clean.split() else clean
        return f"{command} Workflow"
    if kind == "path":
        return f"{clean} Workflow"
    if kind == "api":
        return f"{clean} Integration"
    if any(term.startswith("error:") for term in recurring_terms):
        return f"{clean} Troubleshooting"
    return f"{clean} Workflow"


def _humanize(value: str) -> str:
    value = re.sub(r"[_./\\-]+", " ", value).strip()
    words = [word for word in value.split() if word.lower() not in _GENERIC_TERMS]
    words = words[:5] or value.split()[:5]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or "chat-derived-skill"


def _candidate_id(term_keys: list[str]) -> str:
    stable = "\n".join(sorted(set(term_keys)))
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]
    return f"skill-{digest}"


def _trigger_phrases(kind: str, value: str, recurring_terms: list[str]) -> list[str]:
    triggers = [_humanize(value).lower()]
    for term in recurring_terms:
        _kind, _, raw = term.partition(":")
        phrase = _humanize(raw).lower()
        if phrase and phrase not in triggers:
            triggers.append(phrase)
    if kind in _ACTIONABLE_KINDS:
        triggers.append(f"{kind} related workflow")
    return triggers[:8]


def _confidence(seed_row, evidence: list[SkillEvidence], recurring_terms: list[str]) -> float:
    session_count = len(evidence)
    source_count = len({item.source for item in evidence})
    actionable = sum(1 for term in recurring_terms if term.split(":", 1)[0] in _ACTIONABLE_KINDS)
    seed_kind_bonus = 0.12 if seed_row["kind"] in _ACTIONABLE_KINDS else 0.0
    score = (
        0.25
        + min(0.35, math.log1p(session_count) / 5)
        + min(0.18, actionable * 0.045)
        + (0.10 if source_count > 1 else 0.0)
        + seed_kind_bonus
    )
    return min(0.95, round(score, 2))


def _description(name: str, recurring_terms: list[str]) -> str:
    terms = ", ".join(recurring_terms[:4])
    return f"Reusable workflow guidance for {name.lower()} based on repeated chat evidence around {terms}."


def _why(seed_row, evidence: list[SkillEvidence], recurring_terms: list[str]) -> str:
    source_count = len({item.source for item in evidence})
    actionable = [term for term in recurring_terms if term.split(":", 1)[0] in _ACTIONABLE_KINDS]
    if actionable:
        return (
            f"Appears in {len(evidence)} sessions across {source_count} source(s) "
            f"with repeated actionable signals: {', '.join(actionable[:4])}."
        )
    return (
        f"Appears in {len(evidence)} sessions across {source_count} source(s) "
        f"with recurring topic cluster rooted at {_label(seed_row['kind'], seed_row['value'])}."
    )


def _boundaries(kind: str, value: str) -> str:
    if kind == "error":
        return "Keep the skill focused on diagnosing and resolving this error family; avoid general debugging advice."
    if kind == "command":
        return "Keep the skill focused on this command workflow and its common flags/failures."
    if kind == "api":
        return "Keep the skill focused on this API or package integration, not the whole application domain."
    if kind == "path":
        return "Keep the skill focused on workflows around this file or module area."
    return "Keep the skill narrow: include repeated workflow steps and exclude one-off project details."


# Workflow candidates

_LAST_CANDIDATES: dict[str, SkillCandidate] = {}


def workflow_candidates(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    min_sessions: int = 2,
    days_back: int | None = None,
    project: str | None = None,
    limit: int = 10,
) -> list[SkillCandidate]:
    from .workflows import describe_step, intent_keywords, mine_workflows

    patterns = mine_workflows(index, sources, project, days_back, min_sessions=min_sessions, limit=limit)
    candidates = []
    for pattern in patterns:
        keywords = intent_keywords(pattern.intents + [s["title"] for s in pattern.sessions])
        sigs = [step.split(":", 1)[1] for step in pattern.steps if step.startswith("shell:")]
        if keywords:
            base = " ".join(keywords[:3])
        else:
            base = " ".join(dict.fromkeys(sig.split()[0] for sig in sigs)) or pattern.steps[0].split(":")[-1]
        name = _humanize(base) + " Workflow"
        slug = _slugify(name)
        session_count = len(pattern.sessions)
        source_count = len(pattern.sources)
        confidence = min(0.95, round(
            0.3 + min(0.35, math.log1p(session_count) / 5) + (0.1 if source_count > 1 else 0.0)
            + (0.1 if len(pattern.projects) > 1 else 0.0) + min(0.1, 0.02 * len(pattern.steps)), 2,
        ))
        pretty = [describe_step(step) for step in pattern.steps]
        triggers = [k for k in keywords] + [sig for sig in sigs if sig not in keywords]
        candidates.append(SkillCandidate(
            candidate_id=pattern.candidate_id,
            name=name,
            slug=slug,
            description=f"Repeatable workflow: {', then '.join(pretty)}.",
            triggers=triggers[:8] or [slug.replace('-', ' ')],
            recurring_terms=[f"step:{s}" for s in pattern.steps],
            evidence=[SkillEvidence(s["session_id"], s["title"], s["date"], s["source"]) for s in pattern.sessions[:8]],
            confidence=confidence,
            why=(
                f"The same {len(pattern.steps)}-step tool sequence ran in {session_count} sessions "
                f"across {source_count} agent(s) and {max(1, len(pattern.projects))} project(s)."
            ),
            boundaries="Keep the skill to this sequence and its known failure modes; leave project-specific values as parameters.",
            kind="workflow",
            steps=list(pattern.steps),
            scope=pattern.scope,
            examples={i: [cmd for cmd, _n in counter.most_common(2)] for i, counter in pattern.examples.items()},
            failures=list(pattern.failures),
        ))
    return candidates


# Overlap with installed skills

_WORD_RE = re.compile(r"[a-z][a-z0-9+#-]{2,}")


def _key_terms(candidate: SkillCandidate) -> set[str]:
    words: set[str] = set()
    for text in [candidate.name, *candidate.triggers[:6], *candidate.steps, *candidate.recurring_terms[:6]]:
        for word in _WORD_RE.findall(text.lower().replace(":", " ")):
            if word not in _GENERIC_TERMS and word not in ("workflow", "troubleshooting", "integration", "shell", "edit", "step"):
                words.add(word)
    return words


def annotate_overlap(candidates: list[SkillCandidate], installed: list) -> None:
    try:
        from rapidfuzz.fuzz import token_set_ratio
    except ImportError:  # pragma: no cover
        token_set_ratio = lambda a, b: 100.0 if a == b else 0.0  # noqa: E731

    for candidate in candidates:
        terms = _key_terms(candidate)
        best = (0.0, None, "")
        for skill in installed:
            name_score = token_set_ratio(candidate.slug.replace("-", " "), skill.name.replace("-", " "))
            haystack = set(_WORD_RE.findall(f"{skill.name} {skill.description} {skill.body[:4000]}".lower()))
            coverage = (len(terms & haystack) / len(terms) * 100) if terms else 0.0
            score = max(name_score if name_score >= 85 else 0.0, coverage)
            if score > best[0]:
                reason = "same name" if name_score >= 85 and name_score >= coverage else f"{coverage:.0f}% of its key terms already covered"
                best = (score, skill, reason)
        score, skill, reason = best
        if skill is None or score < 50:
            candidate.action, candidate.overlap_with = "create", ""
            candidate.overlap_reason = "no installed skill covers this"
        elif score >= 80:
            candidate.action, candidate.overlap_with = "skip", skill.name
            candidate.overlap_reason = f"already covered by installed skill ({reason})"
        else:
            candidate.action, candidate.overlap_with = "update", skill.name
            candidate.overlap_reason = f"partly covered ({reason}); extend that skill instead of adding a new one"


def find_candidate(index: HistoryGraphIndex, candidate_id: str, sources: list[str] | None = None) -> SkillCandidate | None:
    if candidate_id in _LAST_CANDIDATES:
        return _LAST_CANDIDATES[candidate_id]
    for min_sessions in (2, 1):
        for candidate in suggest_skill_candidates(index, sources, max_candidates=100, min_sessions=min_sessions, include_workflows=True):
            if candidate.candidate_id == candidate_id or candidate.slug == candidate_id:
                return candidate
    return None


# Drafting

def draft_skill_md(candidate: SkillCandidate) -> str:
    from .workflows import describe_step

    slug = candidate.slug[:64].strip("-") or "chat-derived-skill"
    agents = sorted({e.source for e in candidate.evidence})
    when = ", ".join(candidate.triggers[:5])
    description = f"{candidate.description.rstrip('.')}. Use when working on {when}." if when else candidate.description
    description = description.replace("\n", " ")[:1000]

    lines = ["---", f"name: {slug}", f"description: {description}", "---", "", f"# {candidate.name}", ""]
    lines.append(f"{candidate.description}")
    lines.extend(["", "## When to use", ""])
    lines.extend(f"- {t}" for t in candidate.triggers[:6])

    lines.extend(["", "## Steps", ""])
    if candidate.steps:
        for i, step in enumerate(candidate.steps):
            examples = candidate.examples.get(i, [])
            example = f" — e.g. `{examples[0]}`" if examples else ""
            lines.append(f"{i + 1}. {describe_step(step)[:1].upper() + describe_step(step)[1:]}{example}")
    else:
        commands = [t.split(":", 1)[1] for t in candidate.recurring_terms if t.startswith("command:")]
        apis = [t.split(":", 1)[1] for t in candidate.recurring_terms if t.startswith(("api:", "path:"))]
        lines.append("1. Confirm the goal and gather the current state (logs, config, failing output).")
        if commands:
            lines.append("2. Use the commands that worked before: " + ", ".join(f"`{c}`" for c in commands[:4]) + ".")
        if apis:
            lines.append(f"{3 if commands else 2}. Check the usual touch points: " + ", ".join(f"`{a}`" for a in apis[:4]) + ".")
        lines.append(f"{2 + bool(commands) + bool(apis)}. Apply the fix, then re-run the failing check to verify.")

    failures = list(candidate.failures)
    errors = [t.split(":", 1)[1] for t in candidate.recurring_terms if t.startswith("error:")]
    if failures or errors:
        lines.extend(["", "## Known failure modes", ""])
        for sig, error, fix in failures[:5]:
            lines.append(f"- `{sig}` failed with \"{error}\"" + (f"; it later succeeded as `{fix}`" if fix else ""))
        for error in errors[:3]:
            lines.append(f"- {error}")

    verify = next((ex[0] for i, ex in sorted(candidate.examples.items(), reverse=True)
                   if ex and i < len(candidate.steps) and candidate.steps[i].startswith("shell:")), "")
    lines.extend(["", "## Verify", "", f"- Re-run `{verify}` and confirm it succeeds." if verify else "- Re-run the check that originally failed and confirm it passes."])
    lines.extend(["", "## Boundaries", "", f"- {candidate.boundaries}"])
    lines.extend([
        "", "## Evidence", "",
        f"Derived by agent-history-mcp from {len(candidate.evidence)} session(s) in {', '.join(agents)}:",
    ])
    lines.extend(f"- {e.source} {e.session_date or ''}: {e.session_title[:80]}" for e in candidate.evidence[:5])
    lines.extend(["", "Review and edit before installing; remove project-specific values."])
    return "\n".join(lines) + "\n"


# Skill health

@dataclass
class SkillHealth:
    name: str
    path: str
    agents: list[str]
    uses: list[dict]
    by_source: Counter
    last_used: str
    missed: list[dict]
    stale_commands: list[str]
    outcome: str
    recommendation: str


def _sessions_matching_steps(index: HistoryGraphIndex, steps: list[str], days_back: int | None) -> list[dict]:
    from .workflows import load_sequences

    target = tuple(steps)
    n = len(target)
    matches = []
    for seq in load_sequences(index, None, None, days_back).values():
        if any(tuple(seq.tokens[i:i + n]) == target for i in range(len(seq.tokens) - n + 1)):
            matches.append({"session_key": seq.row["session_key"], "source": seq.row["source"],
                            "session_id": seq.row["session_id"], "date": seq.row["session_date"]})
    return matches


def _sessions_matching_name(index: HistoryGraphIndex, name: str, days_back: int | None) -> list[dict]:
    words = [w for w in re.split(r"[-_\s]+", name.lower()) if len(w) >= 3 and w not in _GENERIC_TERMS and w not in ("skill", "workflow", "helper")]
    if not words or (len(words) == 1 and len(words[0]) < 5):
        return []
    date_filter, params = _date_filter(days_back)
    likes = " AND ".join("LOWER(e.text) LIKE ?" for _ in words)
    rows = index.conn.execute(
        f"""
        SELECT DISTINCT s.session_key, s.source, s.session_id, s.session_date AS date
        FROM events e JOIN sessions s ON s.session_key = e.session_key
        WHERE e.kind = 'user' AND {likes} {date_filter}
        """,
        (*[f"%{w}%" for w in words], *params),
    ).fetchall()
    return [dict(row) for row in rows]


def _error_rate(index: HistoryGraphIndex, session_keys: list[str]) -> float | None:
    if not session_keys:
        return None
    row = index.conn.execute(
        f"SELECT SUM(n_errors) AS e, SUM(n_tool_calls) AS t FROM sessions WHERE session_key IN ({','.join('?' for _ in session_keys)})",
        session_keys,
    ).fetchone()
    return (row["e"] or 0) / row["t"] if row and row["t"] else None


def skill_health(
    index: HistoryGraphIndex,
    installed: list,
    days_back: int | None = 60,
    project: str | None = None,
) -> list[SkillHealth]:
    from .graph import extract_graph_terms
    from .model import command_signature
    from .skill_store import installed_records, skill_invocations

    usage = skill_invocations(index, days_back, project)
    records = installed_records(index)
    date_filter, params = _date_filter(days_back)
    seen_sigs = {
        row["cmd_sig"] for row in index.conn.execute(
            f"SELECT DISTINCT e.cmd_sig FROM events e JOIN sessions s ON s.session_key = e.session_key "
            f"WHERE e.cmd_sig != '' {date_filter}", params,
        ).fetchall()
    }
    report: list[SkillHealth] = []
    for skill in installed:
        uses = usage.get(skill.name.lower(), [])
        if skill.scope == "plugin" and not uses:
            continue  # plugin bundles ship many skills; only report the ones you use
        used_keys = {u["session_key"] for u in uses}
        record = records.get(skill.name)
        steps = json.loads(record["steps"]) if record and record.get("steps") else None
        candidates = _sessions_matching_steps(index, steps, days_back) if steps else _sessions_matching_name(index, skill.name, days_back)
        missed = [m for m in candidates if m["session_key"] not in used_keys]

        referenced = {
            command_signature(term.value) for term in extract_graph_terms(skill.body, max_terms=60) if term.kind == "command"
        }
        stale = sorted(sig for sig in referenced if sig and sig not in seen_sigs) if seen_sigs else []

        used_rate = _error_rate(index, list(used_keys))
        missed_rate = _error_rate(index, [m["session_key"] for m in missed])
        if used_rate is not None and missed_rate is not None:
            outcome = f"error rate with skill {used_rate:.0%} vs without {missed_rate:.0%} (directional)"
        else:
            outcome = "not enough comparable sessions"

        if not uses and missed:
            recommendation = "revise description: matching work happened but the skill never triggered"
        elif not uses:
            recommendation = "consider retiring: not used and no matching work in this window"
        elif used_rate is not None and missed_rate is not None and used_rate > missed_rate + 0.1:
            recommendation = "update: sessions using it fail more often than similar sessions without it"
        elif len(missed) >= len(uses):
            recommendation = f"revise description: it fired in {len(uses)} of {len(uses) + len(missed)} matching sessions"
        elif stale:
            recommendation = "update: it references commands that no longer appear in your history"
        else:
            recommendation = "keep"
        report.append(SkillHealth(
            name=skill.name, path=str(skill.path), agents=skill.agents, uses=uses,
            by_source=Counter(u["source"] for u in uses), last_used=uses[-1]["date"] if uses else "",
            missed=missed, stale_commands=stale, outcome=outcome, recommendation=recommendation,
        ))
    order = {"keep": 3}
    report.sort(key=lambda h: (order.get(h.recommendation, 0), -len(h.missed), h.name))
    return report


def format_skill_health(report: list[SkillHealth], days_back: int | None) -> str:
    if not report:
        return "No installed skills found in ~/.claude/skills, ~/.agents/skills, ~/.codex/skills or the project."
    window = f"last {days_back} days" if days_back else "all time"
    lines = [f"Skill health for {len(report)} installed skill(s), {window}:", ""]
    for h in report:
        used = ", ".join(f"{src} {n}" for src, n in h.by_source.most_common()) or "never"
        lines.append(f"- {h.name} ({', '.join(h.agents)})")
        lines.append(f"  used: {used}" + (f" · last {h.last_used}" if h.last_used else ""))
        if h.missed:
            examples = ", ".join(f"{m['source']}:{m['session_id'][:10]}" for m in h.missed[:3])
            lines.append(f"  missed triggers: {len(h.missed)} matching session(s) without it ({examples})")
        if h.stale_commands:
            lines.append(f"  drift: references {', '.join(f'`{c}`' for c in h.stale_commands[:4])} not seen recently")
        lines.append(f"  outcome: {h.outcome}")
        lines.append(f"  -> {h.recommendation}")
    return "\n".join(lines)
