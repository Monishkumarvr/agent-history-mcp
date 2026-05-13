"""
Suggest reusable Codex/Claude skills from the local chat graph.

The suggestions are deterministic and evidence-backed. They do not write skill
files and they do not call a model.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .graph import HistoryGraphIndex


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
) -> list[SkillCandidate]:
    source_set = sorted({s for s in (sources or ["codex", "claude"]) if s in ("codex", "claude")})
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
            f"   Confidence: {candidate.confidence:.2f}",
            f"   Description: {candidate.description}",
            f"   When to use: {', '.join(candidate.triggers[:6])}",
            f"   Recurring terms: {', '.join(candidate.recurring_terms[:8])}",
            f"   Evidence: {len(candidate.evidence)} sessions, {len(sources)} source(s)",
            f"   Sessions: {evidence}",
            f"   Why: {candidate.why}",
            f"   Boundaries: {candidate.boundaries}",
            "",
        ])
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
