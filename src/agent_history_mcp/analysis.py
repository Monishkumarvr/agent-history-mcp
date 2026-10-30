"""
Deterministic per-session scorecards across every indexed agent.

This module turns indexed sessions into local scorecards and evidence-backed
recommendations. It does not call models and it does not write instruction or
skill files; it only reports what the chat logs make observable.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .graph import HistoryGraphIndex, event_from_row, session_filters
from .model import Event

_EVENT_TYPES = {"user": "user_prompt", "assistant": "assistant_response"}


@dataclass
class FrictionFinding:
    label: str
    severity: str
    details: str
    evidence: list[str]
    remedy: str
    impact: float


@dataclass
class SessionScorecard:
    session_id: str
    source: str
    title: str
    session_date: str
    overall_score: float
    prompt_clarity: float
    context_efficiency: float
    rework_rate: float
    ai_consistency: float
    task_completion: float
    metrics: dict[str, int | float]
    findings: list[FrictionFinding] = field(default_factory=list)

    @property
    def session_key(self) -> str:
        return f"session:{self.source}:{self.session_id}"


@dataclass
class InstructionSuggestion:
    suggestion_id: str
    target: str
    rule: str
    rationale: str
    confidence: float
    evidence: list[str]
    boundaries: str


_PATH_RE = re.compile(
    r"(?<![\w.-])("
    r"[A-Za-z]:[\\/][^\s`'\"<>|]{3,}|"
    r"(?:\.{1,2}[\\/]|[/\\])?[A-Za-z0-9_.@()-]+(?:[\\/][A-Za-z0-9_.@() -]+)+\.[A-Za-z0-9]{1,8}|"
    r"[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|rs|go|java|cs|cpp|cxx|c|h|hpp|sql|sh|ps1)"
    r")(?![A-Za-z0-9])"
)
_COMMAND_PREFIX_RE = re.compile(
    r"(?im)^\s*((?:az|cargo|docker|dotnet|git|gh|go|kubectl|make|mvn|node|npm|pnpm|powershell|pwsh|pytest|python|python3|ruff|uv|yarn)\b[^\n]{0,180})"
)
_CORRECTION_RE = re.compile(
    r"\b(actually|again|already said|does not work|didn't ask|incorrect|missed|not what|still|that is wrong|wrong)\b",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(always|do not|don't|must|never|prefer|remember|should not|use .* instead)\b[^\n.]{0,180}",
    re.IGNORECASE,
)
_ACCEPTANCE_RE = re.compile(r"\b(test|verify|acceptance|criteria|done when|pass|expected|must)\b", re.IGNORECASE)
_BROAD_VERB_RE = re.compile(r"\b(build|check|create|debug|do|fix|help|implement|make|review|update)\b", re.IGNORECASE)
_FAILURE_RE = re.compile(r"\b(error|exception|failed|failure|not found|permission denied|timed out|timeout|traceback)\b", re.IGNORECASE)
_PERMISSION_RE = re.compile(r"\b(access denied|blocked|not allowed|permission denied|sandbox)\b", re.IGNORECASE)
_COMPLETION_RE = re.compile(r"\b(done|fixed|implemented|passes|verified|complete|completed)\b", re.IGNORECASE)


def analyze_session(session: dict) -> SessionScorecard:
    events = _events_for_session(session)
    messages = session.get("messages", [])
    findings: list[FrictionFinding] = []

    file_reads = Counter(_norm_path(event.get("path", "")) for event in events if _is_file_read(event))
    file_reads = Counter({path: count for path, count in file_reads.items() if path})
    repeated_reads = {path: count for path, count in file_reads.items() if count >= 3}
    if repeated_reads:
        worst = sorted(repeated_reads.items(), key=lambda item: (-item[1], item[0]))[:5]
        findings.append(FrictionFinding(
            label="repeated_file_reads",
            severity="medium",
            details=f"{len(repeated_reads)} file(s) were read repeatedly.",
            evidence=[f"{path} read {count} times" for path, count in worst],
            remedy="Cache findings in the working notes or ask the agent to summarize file state before rereading.",
            impact=min(20.0, sum(repeated_reads.values()) * 2.0),
        ))

    failed_commands = Counter(_command_key(event.get("command", "")) for event in events if _is_failed_command(event))
    failed_commands = Counter({cmd: count for cmd, count in failed_commands.items() if cmd})
    repeated_failures = {cmd: count for cmd, count in failed_commands.items() if count >= 2}
    if repeated_failures:
        worst = sorted(repeated_failures.items(), key=lambda item: (-item[1], item[0]))[:5]
        findings.append(FrictionFinding(
            label="repeated_failed_commands",
            severity="high",
            details=f"{len(repeated_failures)} command family/families failed repeatedly.",
            evidence=[f"{cmd} failed {count} times" for cmd, count in worst],
            remedy="Add a preflight check or narrow troubleshooting rule before retrying the same command family.",
            impact=min(24.0, sum(repeated_failures.values()) * 4.0),
        ))

    permission_failures = [
        event for event in events
        if _is_failed_command(event) and _PERMISSION_RE.search(event.get("text", ""))
    ]
    if permission_failures:
        findings.append(FrictionFinding(
            label="permission_or_tool_blocked",
            severity="medium",
            details=f"{len(permission_failures)} command/tool failure(s) looked permission-related.",
            evidence=[_tool_label(event) for event in permission_failures[:5]],
            remedy="Document safe required commands, permissions, or approval expectations in project instructions.",
            impact=min(14.0, len(permission_failures) * 4.0),
        ))

    correction_turns = _correction_turns(messages)
    if correction_turns:
        findings.append(FrictionFinding(
            label="correction_loop",
            severity="high" if len(correction_turns) >= 3 else "medium",
            details=f"{len(correction_turns)} user correction turn(s) detected.",
            evidence=correction_turns[:5],
            remedy="Move repeated constraints into CLAUDE.md or AGENTS.md and ask for a short plan before execution.",
            impact=min(28.0, len(correction_turns) * 7.0),
        ))

    first_prompt = _first_user_prompt(messages)
    vague_opener = _is_vague_opener(first_prompt)
    tool_calls = [event for event in events if event.get("event_type") == "tool_call"]
    if vague_opener and len(tool_calls) >= 6:
        findings.append(FrictionFinding(
            label="vague_opener_excessive_exploration",
            severity="medium",
            details="The opening prompt was broad and was followed by heavy tool exploration.",
            evidence=[first_prompt[:220], f"{len(tool_calls)} tool call(s) recorded"],
            remedy="Start with file paths, desired outcome, constraints, and explicit acceptance checks.",
            impact=18.0,
        ))

    context_heavy = _context_heavy_events(events, messages)
    if context_heavy:
        findings.append(FrictionFinding(
            label="context_heavy_turns",
            severity="medium",
            details=f"{len(context_heavy)} context-heavy turn(s) detected.",
            evidence=context_heavy[:5],
            remedy="Ask for summarized tool output and avoid pasting full logs unless needed.",
            impact=min(20.0, len(context_heavy) * 5.0),
        ))

    if len(tool_calls) >= max(12, len(messages) * 2):
        findings.append(FrictionFinding(
            label="tool_call_explosion",
            severity="medium",
            details=f"{len(tool_calls)} tool call(s) across {max(1, len(messages))} message(s).",
            evidence=[_tool_label(event) for event in tool_calls[:6]],
            remedy="Require a short exploration budget or ask the agent to state what it needs before broad searching.",
            impact=min(22.0, len(tool_calls) * 1.2),
        ))

    constraints = _repeated_constraints(messages)
    if constraints:
        findings.append(FrictionFinding(
            label="repeated_instruction_candidates",
            severity="low",
            details=f"{len(constraints)} repeated constraint pattern(s) could become standing instructions.",
            evidence=constraints[:5],
            remedy="Promote stable preferences into CLAUDE.md and AGENTS.md instead of restating them in chat.",
            impact=min(12.0, len(constraints) * 3.0),
        ))

    abandoned = _looks_abandoned(messages, events)
    if abandoned:
        findings.append(FrictionFinding(
            label="non_converged_session",
            severity="medium",
            details="The session appears to end without a clear completion signal.",
            evidence=[_last_message_text(messages)[:220]],
            remedy="End long sessions with an explicit verification checklist or next-step summary.",
            impact=12.0,
        ))

    metrics = {
        "messages": len(messages),
        "events": len(events),
        "tool_calls": len(tool_calls),
        "observed_cost": round(sum(float(event.get("cost") or 0.0) for event in events), 6),
        "file_reads": sum(file_reads.values()),
        "repeated_file_reads": sum(repeated_reads.values()),
        "failed_commands": sum(failed_commands.values()),
        "repeated_failed_commands": sum(repeated_failures.values()),
        "permission_failures": len(permission_failures),
        "correction_turns": len(correction_turns),
        "context_heavy_turns": len(context_heavy),
        "repeated_constraints": len(constraints),
        "vague_opener": 1 if vague_opener else 0,
        "non_converged": 1 if abandoned else 0,
    }

    prompt_clarity = _clamp(92 - (22 if vague_opener else 0) - (10 if not _ACCEPTANCE_RE.search(first_prompt) and len(tool_calls) >= 6 else 0))
    context_efficiency = _clamp(92 - _impact(findings, {"repeated_file_reads", "context_heavy_turns", "tool_call_explosion"}))
    rework_rate = _clamp(94 - _impact(findings, {"correction_loop", "repeated_failed_commands", "repeated_instruction_candidates"}))
    ai_consistency = _clamp(90 - _impact(findings, {"repeated_failed_commands", "tool_call_explosion", "non_converged_session"}) * 0.7)
    task_completion = _clamp(92 - (18 if abandoned else 0) - min(16, metrics["failed_commands"] * 2) - min(18, metrics["correction_turns"] * 4))
    overall = round(
        prompt_clarity * 0.20
        + context_efficiency * 0.25
        + rework_rate * 0.20
        + ai_consistency * 0.15
        + task_completion * 0.20,
        1,
    )

    return SessionScorecard(
        session_id=session["session_id"],
        source=session["source"],
        title=session["session_title"],
        session_date=session.get("session_date", ""),
        overall_score=overall,
        prompt_clarity=round(prompt_clarity, 1),
        context_efficiency=round(context_efficiency, 1),
        rework_rate=round(rework_rate, 1),
        ai_consistency=round(ai_consistency, 1),
        task_completion=round(task_completion, 1),
        metrics=metrics,
        findings=findings,
    )


def analyze_sessions(
    sessions: list[dict],
    days_back: int | None = 30,
    sources: list[str] | None = None,
) -> list[SessionScorecard]:
    source_set = set(sources or [])
    cutoff = date.today() - timedelta(days=days_back) if days_back and days_back > 0 else None
    scorecards = []
    for session in sessions:
        if source_set and session.get("source") not in source_set:
            continue
        if cutoff and not _date_in_range(session.get("session_date", ""), cutoff):
            continue
        scorecards.append(analyze_session(session))
    scorecards.sort(key=lambda card: (card.overall_score, card.session_date, card.title))
    return scorecards


def ensure_analysis_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session_metrics (
            session_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            session_id TEXT NOT NULL,
            score_overall REAL NOT NULL,
            prompt_clarity REAL NOT NULL,
            context_efficiency REAL NOT NULL,
            rework_rate REAL NOT NULL,
            ai_consistency REAL NOT NULL,
            task_completion REAL NOT NULL,
            metrics_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS friction_labels (
            session_key TEXT NOT NULL,
            label TEXT NOT NULL,
            severity TEXT NOT NULL,
            details TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            remedy TEXT NOT NULL,
            impact REAL NOT NULL,
            PRIMARY KEY(session_key, label, details)
        );

        CREATE TABLE IF NOT EXISTS tool_call_counts (
            session_key TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(session_key, tool_name)
        );

        CREATE TABLE IF NOT EXISTS command_counts (
            session_key TEXT NOT NULL,
            command TEXT NOT NULL,
            success_count INTEGER NOT NULL,
            failure_count INTEGER NOT NULL,
            PRIMARY KEY(session_key, command)
        );

        CREATE TABLE IF NOT EXISTS recommendation_evidence (
            recommendation_id TEXT NOT NULL,
            target TEXT NOT NULL,
            session_key TEXT NOT NULL,
            label TEXT NOT NULL,
            evidence TEXT NOT NULL,
            confidence REAL NOT NULL,
            PRIMARY KEY(recommendation_id, target, session_key, label)
        );

        CREATE INDEX IF NOT EXISTS idx_session_metrics_source ON session_metrics(source, score_overall);
        CREATE INDEX IF NOT EXISTS idx_friction_labels_label ON friction_labels(label, severity);
        """
    )
    conn.commit()


def persist_analysis(conn: sqlite3.Connection, sessions: list[dict], scorecards: list[SessionScorecard]) -> None:
    ensure_analysis_tables(conn)
    session_by_key = {f"session:{session['source']}:{session['session_id']}": session for session in sessions}
    for card in scorecards:
        conn.execute(
            """
            INSERT OR REPLACE INTO session_metrics(
                session_key, source, session_id, score_overall, prompt_clarity,
                context_efficiency, rework_rate, ai_consistency, task_completion,
                metrics_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.session_key,
                card.source,
                card.session_id,
                card.overall_score,
                card.prompt_clarity,
                card.context_efficiency,
                card.rework_rate,
                card.ai_consistency,
                card.task_completion,
                json.dumps(card.metrics, sort_keys=True),
                time.time(),
            ),
        )
        conn.execute("DELETE FROM friction_labels WHERE session_key = ?", (card.session_key,))
        for finding in card.findings:
            conn.execute(
                """
                INSERT OR REPLACE INTO friction_labels(
                    session_key, label, severity, details, evidence_json, remedy, impact
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card.session_key,
                    finding.label,
                    finding.severity,
                    finding.details,
                    json.dumps(finding.evidence, ensure_ascii=False),
                    finding.remedy,
                    finding.impact,
                ),
            )

        session = session_by_key.get(card.session_key, {})
        events = _events_for_session(session)
        conn.execute("DELETE FROM tool_call_counts WHERE session_key = ?", (card.session_key,))
        for tool_name, count in Counter(_tool_name(event) for event in events if event.get("event_type") == "tool_call").items():
            conn.execute(
                "INSERT OR REPLACE INTO tool_call_counts(session_key, tool_name, count) VALUES (?, ?, ?)",
                (card.session_key, tool_name, count),
            )

        command_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for event in events:
            command = _command_key(event.get("command", ""))
            if not command:
                continue
            if event.get("success") is False:
                command_counts[command][1] += 1
            else:
                command_counts[command][0] += 1
        conn.execute("DELETE FROM command_counts WHERE session_key = ?", (card.session_key,))
        for command, (success_count, failure_count) in command_counts.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO command_counts(session_key, command, success_count, failure_count)
                VALUES (?, ?, ?, ?)
                """,
                (card.session_key, command, success_count, failure_count),
            )
    conn.commit()


def suggest_instruction_updates(
    scorecards: list[SessionScorecard],
    targets: list[str] | None = None,
    max_suggestions: int = 8,
) -> list[InstructionSuggestion]:
    target_set = [target for target in (targets or ["CLAUDE.md", "AGENTS.md"]) if target in ("CLAUDE.md", "AGENTS.md")]
    if not target_set:
        return []

    label_to_cards: dict[str, list[SessionScorecard]] = defaultdict(list)
    for card in scorecards:
        for finding in card.findings:
            label_to_cards[finding.label].append(card)

    suggestions: list[InstructionSuggestion] = []
    for label, cards in sorted(label_to_cards.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(cards) < 2 and label != "repeated_instruction_candidates":
            continue
        rule, rationale, boundaries = _rule_for_label(label)
        if not rule:
            continue
        evidence = [
            f"{card.source}:{card.session_id[:12]} ({card.session_date or 'unknown'}) score {card.overall_score:.1f}"
            for card in cards[:5]
        ]
        confidence = min(0.92, 0.48 + len(cards) * 0.08)
        for target in target_set:
            if target == "CLAUDE.md" and not any(card.source == "claude" for card in cards):
                continue
            if target == "AGENTS.md" and not any(card.source != "claude" for card in cards):
                continue
            suggestions.append(InstructionSuggestion(
                suggestion_id=_suggestion_id(target, label, rule),
                target=target,
                rule=rule,
                rationale=rationale,
                confidence=round(confidence, 2),
                evidence=evidence,
                boundaries=boundaries,
            ))

    suggestions.sort(key=lambda item: (-item.confidence, item.target, item.rule))
    return suggestions[:max_suggestions]


def persist_instruction_suggestions(conn: sqlite3.Connection, suggestions: list[InstructionSuggestion]) -> None:
    ensure_analysis_tables(conn)
    for suggestion in suggestions:
        for evidence in suggestion.evidence:
            session_key = _session_key_from_evidence(evidence)
            conn.execute(
                """
                INSERT OR REPLACE INTO recommendation_evidence(
                    recommendation_id, target, session_key, label, evidence, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion.suggestion_id,
                    suggestion.target,
                    session_key,
                    suggestion.rule[:80],
                    evidence,
                    suggestion.confidence,
                ),
            )
    conn.commit()


def format_scorecard(card: SessionScorecard) -> str:
    lines = [
        f"Session scorecard: {card.title}",
        f"Source : {card.source.upper()}",
        f"Date   : {card.session_date or 'unknown'}",
        f"ID     : {card.session_id}",
        "",
        f"Overall score      : {card.overall_score:.1f}/100",
        f"Prompt clarity     : {card.prompt_clarity:.1f}/100",
        f"Context efficiency : {card.context_efficiency:.1f}/100",
        f"Rework rate        : {card.rework_rate:.1f}/100",
        f"AI consistency     : {card.ai_consistency:.1f}/100",
        f"Task completion    : {card.task_completion:.1f}/100",
        "",
        "Key metrics:",
    ]
    for key in ("messages", "tool_calls", "file_reads", "failed_commands", "correction_turns", "context_heavy_turns"):
        lines.append(f"- {key}: {card.metrics.get(key, 0)}")
    if card.findings:
        lines.append("")
        lines.append("Findings:")
        for finding in card.findings:
            lines.append(f"- {finding.label} ({finding.severity}): {finding.details}")
            if finding.evidence:
                lines.append(f"  Evidence: {finding.evidence[0]}")
            lines.append(f"  Remedy: {finding.remedy}")
    else:
        lines.append("")
        lines.append("No major deterministic friction patterns detected.")
    return "\n".join(lines)


def format_aggregate_report(scorecards: list[SessionScorecard], days_back: int | None, sources: list[str]) -> str:
    if not scorecards:
        return "No sessions found for the requested source/date filters."

    avg = sum(card.overall_score for card in scorecards) / len(scorecards)
    source_counts = Counter(card.source for card in scorecards)
    labels = Counter(finding.label for card in scorecards for finding in card.findings)
    worst = sorted(scorecards, key=lambda card: card.overall_score)[:5]
    best = sorted(scorecards, key=lambda card: card.overall_score, reverse=True)[:3]

    lines = [
        "Cross-agent session insights",
        f"Scope: {', '.join(sources)}" + (f", last {days_back} day(s)" if days_back else ", all dates"),
        f"Sessions analyzed: {len(scorecards)}",
        f"Average score: {avg:.1f}/100",
        "Source mix: " + ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items())),
        "",
        "Top friction patterns:",
    ]
    if labels:
        for label, count in labels.most_common(8):
            lines.append(f"- {label}: {count} session(s)")
    else:
        lines.append("- No repeated friction patterns detected.")

    lines.append("")
    lines.append("Lowest scoring sessions:")
    for card in worst:
        labels_text = ", ".join(f.label for f in card.findings[:3]) or "no major findings"
        lines.append(f"- {card.overall_score:.1f} {card.source}:{card.session_id[:12]} {card.session_date or 'unknown'} - {labels_text}")

    lines.append("")
    lines.append("Highest scoring sessions:")
    for card in best:
        lines.append(f"- {card.overall_score:.1f} {card.source}:{card.session_id[:12]} {card.session_date or 'unknown'} - {card.title[:70]}")

    return "\n".join(lines)


def format_instruction_suggestions(suggestions: list[InstructionSuggestion]) -> str:
    if not suggestions:
        return "No evidence-backed instruction updates found for the requested filters."
    lines = [f"Found {len(suggestions)} instruction suggestion(s):\n"]
    for idx, suggestion in enumerate(suggestions, 1):
        lines.extend([
            f"{idx}. {suggestion.target}",
            f"   ID: {suggestion.suggestion_id}",
            f"   Confidence: {suggestion.confidence:.2f}",
            f"   Suggested rule: {suggestion.rule}",
            f"   Why: {suggestion.rationale}",
            f"   Evidence: {'; '.join(suggestion.evidence[:4])}",
            f"   Boundaries: {suggestion.boundaries}",
            "",
        ])
    return "\n".join(lines).rstrip()


def format_friction_patterns(scorecards: list[SessionScorecard], min_sessions: int = 2) -> str:
    labels: dict[str, list[SessionScorecard]] = defaultdict(list)
    examples: dict[str, list[str]] = defaultdict(list)
    for card in scorecards:
        for finding in card.findings:
            labels[finding.label].append(card)
            examples[finding.label].extend(finding.evidence[:1])

    rows = [
        (label, cards)
        for label, cards in labels.items()
        if len({card.session_key for card in cards}) >= min_sessions
    ]
    rows.sort(key=lambda item: (-len(item[1]), item[0]))
    if not rows:
        return "No friction pattern crossed the requested min_sessions threshold."

    lines = ["Recurring friction patterns:\n"]
    for label, cards in rows:
        source_count = len({card.source for card in cards})
        avg_score = sum(card.overall_score for card in cards) / len(cards)
        lines.append(f"- {label}: {len(cards)} session(s), {source_count} source(s), avg score {avg_score:.1f}")
        if examples[label]:
            lines.append(f"  Example: {examples[label][0]}")
    return "\n".join(lines)


def format_comparison(before: SessionScorecard, after: SessionScorecard) -> str:
    delta = after.overall_score - before.overall_score
    direction = "improved" if delta > 0 else "declined" if delta < 0 else "stayed flat"
    rows = [
        ("Overall", before.overall_score, after.overall_score),
        ("Prompt clarity", before.prompt_clarity, after.prompt_clarity),
        ("Context efficiency", before.context_efficiency, after.context_efficiency),
        ("Rework rate", before.rework_rate, after.rework_rate),
        ("AI consistency", before.ai_consistency, after.ai_consistency),
        ("Task completion", before.task_completion, after.task_completion),
    ]
    lines = [
        "Session comparison",
        f"Before: {before.source}:{before.session_id[:12]} {before.session_date or 'unknown'}",
        f"After : {after.source}:{after.session_id[:12]} {after.session_date or 'unknown'}",
        f"Result: {direction} by {delta:+.1f} point(s)",
        "",
    ]
    for label, old, new in rows:
        lines.append(f"- {label}: {old:.1f} -> {new:.1f} ({new - old:+.1f})")
    before_labels = Counter(f.label for f in before.findings)
    after_labels = Counter(f.label for f in after.findings)
    resolved = sorted(set(before_labels) - set(after_labels))
    introduced = sorted(set(after_labels) - set(before_labels))
    if resolved:
        lines.append("")
        lines.append("Resolved patterns: " + ", ".join(resolved))
    if introduced:
        lines.append("New patterns: " + ", ".join(introduced))
    return "\n".join(lines)


def find_scorecard(scorecards: list[SessionScorecard], session_id: str, source: str) -> SessionScorecard | None:
    for card in scorecards:
        if card.source == source and (card.session_id == session_id or card.session_id.startswith(session_id)):
            return card
    return None


def load_sessions(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    project: str | None = None,
    days_back: int | None = None,
) -> list[dict]:
    """Sessions with messages and events, read from the graph index (no JSONL re-parse)."""
    clauses, params = session_filters(sources, project, days_back)
    where = " AND ".join(clauses) or "1=1"
    sessions: dict[str, dict] = {}
    for row in index.conn.execute(f"SELECT * FROM sessions s WHERE {where}", params):
        sessions[row["session_key"]] = {
            "session_id": row["session_id"],
            "session_title": row["title"],
            "session_date": row["session_date"],
            "source": row["source"],
            "messages": [],
            "events": [],
        }
    if not sessions:
        return []
    for msg in index.conn.execute(
        f"SELECT m.session_key, m.role, m.text, m.ts FROM messages m JOIN sessions s ON s.session_key = m.session_key "
        f"WHERE {where} ORDER BY m.session_key, m.msg_idx",
        params,
    ):
        sessions[msg["session_key"]]["messages"].append({"role": msg["role"], "text": msg["text"], "ts": msg["ts"] or ""})
    for row in index.conn.execute(
        f"SELECT e.* FROM events e JOIN sessions s ON s.session_key = e.session_key "
        f"WHERE {where} ORDER BY e.session_key, e.idx",
        params,
    ):
        session = sessions[row["session_key"]]
        session["events"].append(_event_dict(event_from_row(row), session))
    return list(sessions.values())


def _event_dict(event: Event, session: dict) -> dict:
    """Map a unified model.Event onto the flat event shape the scorecard rules read."""
    if event.kind == "tool_result":
        success = not event.is_error
    else:
        success = None
    return {
        "event_type": _EVENT_TYPES.get(event.kind, event.kind),
        "ts": event.ts,
        "source": session["source"],
        "session_id": session["session_id"],
        "role": "tool" if event.kind.startswith("tool") else event.kind,
        "text": event.text,
        "tool_name": event.tool,
        "canon_tool": event.canon_tool,
        "command": event.command,
        "path": event.paths[0] if event.paths else "",
        "success": success,
        "tokens": {},
        "cost": None,
    }


def _events_for_session(session: dict) -> list[dict]:
    events = [event for event in session.get("events", []) if isinstance(event, dict)]
    if events:
        return events

    out = []
    for msg in session.get("messages", []):
        text = msg.get("text", "")
        event_type = "user_prompt" if msg.get("role") == "user" else "assistant_response"
        out.append({
            "event_type": event_type,
            "ts": msg.get("ts", ""),
            "source": session.get("source", ""),
            "session_id": session.get("session_id", ""),
            "role": msg.get("role", ""),
            "text": text,
            "tool_name": "",
            "command": "",
            "path": "",
            "success": None,
            "tokens": {},
            "cost": None,
        })
        for command in _COMMAND_PREFIX_RE.findall(text):
            out.append({
                "event_type": "tool_call",
                "ts": msg.get("ts", ""),
                "source": session.get("source", ""),
                "session_id": session.get("session_id", ""),
                "role": "tool",
                "text": command,
                "tool_name": "shell",
                "command": command,
                "path": "",
                "success": None,
                "tokens": {},
                "cost": None,
            })
    return out


def _is_file_read(event: dict) -> bool:
    if event.get("canon_tool"):
        return event.get("event_type") == "tool_call" and event["canon_tool"] == "read" and bool(event.get("path"))
    tool = _tool_name(event)
    path = event.get("path") or ""
    text = (event.get("text") or "").lower()
    return bool(path) and (
        event.get("event_type") == "tool_call"
        and any(marker in tool or marker in text for marker in ("read", "grep", "search", "open", "get-content", "select-string", "cat"))
    )


def _is_failed_command(event: dict) -> bool:
    command = event.get("command") or ""
    if not command:
        return False
    if event.get("canon_tool"):
        return event.get("event_type") == "tool_result" and event.get("success") is False
    if event.get("success") is False:
        return True
    return bool(_FAILURE_RE.search(event.get("text", "")))


def _tool_name(event: dict) -> str:
    return str(event.get("tool_name") or "unknown").lower()


def _tool_label(event: dict) -> str:
    tool = event.get("tool_name") or "tool"
    command = event.get("command") or event.get("path") or event.get("text", "")
    return f"{tool}: {str(command)[:100]}"


def _command_key(command: str) -> str:
    command = re.sub(r"\s+", " ", command.strip())
    if not command:
        return ""
    parts = command.split()
    if len(parts) >= 2 and parts[0].lower() in {"python", "python3", "pytest", "npm", "pnpm", "yarn", "git", "gh", "az", "docker"}:
        return " ".join(parts[:2]).lower()
    return parts[0].lower()


def _norm_path(path: str) -> str:
    return path.strip().replace("\\", "/").lower()


def _correction_turns(messages: list[dict]) -> list[str]:
    corrections = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user" or idx == 0:
            continue
        text = msg.get("text", "")
        if _CORRECTION_RE.search(text):
            corrections.append(text[:220])
    return corrections


def _first_user_prompt(messages: list[dict]) -> str:
    return next((msg.get("text", "") for msg in messages if msg.get("role") == "user"), "")


def _last_message_text(messages: list[dict]) -> str:
    return messages[-1].get("text", "") if messages else ""


def _is_vague_opener(text: str) -> bool:
    if not text:
        return False
    has_path = bool(_PATH_RE.search(text))
    has_command = bool(_COMMAND_PREFIX_RE.search(text))
    has_specific_token = bool(re.search(r"\b[A-Z][A-Za-z0-9_]+(?:Error|Exception)\b|`[^`]+`|\w+\.\w+", text))
    broad = bool(_BROAD_VERB_RE.search(text))
    return broad and not (has_path or has_command or has_specific_token) and len(text) < 240


def _context_heavy_events(events: list[dict], messages: list[dict]) -> list[str]:
    heavy = []
    for event in events:
        tokens = event.get("tokens") or {}
        total_tokens = tokens.get("total") or tokens.get("input", 0) + tokens.get("output", 0)
        if total_tokens and total_tokens >= 12000:
            heavy.append(f"{event.get('event_type', 'event')} used {total_tokens} token(s)")
        elif len(event.get("text", "")) >= 16000:
            heavy.append(f"{event.get('event_type', 'event')} had {len(event.get('text', ''))} characters")
    for msg in messages:
        if len(msg.get("text", "")) >= 16000:
            heavy.append(f"{msg.get('role', 'message')} message had {len(msg.get('text', ''))} characters")
    return heavy


def _repeated_constraints(messages: list[dict]) -> list[str]:
    counts: Counter[str] = Counter()
    original: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for match in _CONSTRAINT_RE.finditer(msg.get("text", "")):
            raw = re.sub(r"\s+", " ", match.group(0)).strip()
            norm = raw.lower()
            if len(norm) < 18:
                continue
            counts[norm] += 1
            original.setdefault(norm, raw)
    return [f"{original[norm]} ({count}x)" for norm, count in counts.items() if count >= 2]


def _looks_abandoned(messages: list[dict], events: list[dict]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") == "user":
        return True
    text = last.get("text", "")
    if _COMPLETION_RE.search(text):
        return False
    if any(event.get("success") is False for event in events[-4:]):
        return True
    return bool(text.rstrip().endswith("?") and len(messages) > 4)


def _impact(findings: list[FrictionFinding], labels: set[str]) -> float:
    return sum(finding.impact for finding in findings if finding.label in labels)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _date_in_range(session_date: str, cutoff: date) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", session_date or ""):
        return True
    return date.fromisoformat(session_date) >= cutoff


def _rule_for_label(label: str) -> tuple[str, str, str]:
    rules = {
        "repeated_file_reads": (
            "Before rereading the same files repeatedly, summarize what is already known and state the missing fact being checked.",
            "Repeated reads suggest the agent is using the filesystem as short-term memory.",
            "Keep this as a workflow rule; do not forbid necessary rereads during debugging.",
        ),
        "repeated_failed_commands": (
            "After a command fails twice, stop retrying and diagnose prerequisites, environment, permissions, and exact error output first.",
            "Repeated command failures create avoidable context and time waste.",
            "Allow retries when the command or environment has materially changed.",
        ),
        "correction_loop": (
            "When corrected by the user, restate the corrected requirement in one sentence before making more changes.",
            "Correction loops show that constraints are not being retained across turns.",
            "Do not turn this into a long planning ritual for trivial fixes.",
        ),
        "vague_opener_excessive_exploration": (
            "For broad implementation requests, ask for or infer target files, constraints, and verification steps before wide exploration.",
            "Broad openers followed by many tool calls indicate preventable exploration.",
            "Use this for open-ended coding tasks, not direct questions.",
        ),
        "context_heavy_turns": (
            "Prefer summarized logs and focused excerpts over full tool output unless the full output is required.",
            "Large turns inflate context and make later reasoning more expensive.",
            "Do not hide important stack traces; trim to the relevant failing frame/output.",
        ),
        "tool_call_explosion": (
            "Set an exploration budget for large searches and report the best next target before continuing.",
            "High tool-call density often means the agent lacks a clear search strategy.",
            "Do not block necessary exploration in unfamiliar repositories.",
        ),
        "repeated_instruction_candidates": (
            "Promote stable repeated constraints from chat into this instruction file.",
            "Repeated constraints are durable preferences that should not be typed every session.",
            "Only promote constraints that remain true across projects.",
        ),
        "non_converged_session": (
            "Close substantial tasks with a verification summary, remaining risks, and explicit next steps.",
            "Non-converged sessions are harder to resume and search later.",
            "Skip this for quick Q&A sessions.",
        ),
        "permission_or_tool_blocked": (
            "Document safe command permissions, required tools, and approval expectations before workflows that commonly hit access blocks.",
            "Permission-related failures are often preventable when the agent knows the expected approval path.",
            "Only document permissions that are safe and project-specific; do not encourage bypassing security controls.",
        ),
    }
    return rules.get(label, ("", "", ""))


def _suggestion_id(target: str, label: str, rule: str) -> str:
    digest = hashlib.sha1(f"{target}\n{label}\n{rule}".encode("utf-8")).hexdigest()[:12]
    return f"instruction-{digest}"


def _session_key_from_evidence(evidence: str) -> str:
    match = re.match(r"([a-z]+):([^\s]+)", evidence)
    if not match:
        return ""
    return f"session:{match.group(1)}:{match.group(2)}"


def write_markdown_report(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
