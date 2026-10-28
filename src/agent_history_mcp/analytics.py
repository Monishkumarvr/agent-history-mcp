"""
Deterministic analytics over indexed sessions from every agent.

Computes activity, cost (tokens), tool mix, command failure hotspots, file
churn, friction signals (retry loops, corrections, interrupts, rejected tool
calls, context compactions) and a directional cross-agent comparison for
topics that more than one agent worked on.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

from .graph import HistoryGraphIndex, session_filters
from .model import parse_ts

ACTIVE_GAP_SECONDS = 300
RETRY_LOOP_MIN = 3


@dataclass
class SessionStats:
    session_key: str
    source: str
    session_id: str
    title: str
    project: str
    date: str
    model: str
    input_tokens: int
    output_tokens: int
    n_user: int
    active_minutes: float = 0.0
    tool_calls: int = 0
    tool_results: int = 0
    errors: int = 0
    edits: int = 0
    retry_loops: list[str] = field(default_factory=list)
    signals: Counter = field(default_factory=Counter)
    tools: Counter = field(default_factory=Counter)

    @property
    def friction(self) -> int:
        s = self.signals
        return len(self.retry_loops) * 3 + s["correction"] * 2 + s["interrupt"] + s["rejected"] + s["compact"]


@dataclass
class Analysis:
    scope: str
    sessions: list[SessionStats]
    commands: Counter
    command_failures: Counter
    failure_examples: dict[str, str]
    files: Counter
    comparison: list[dict]


def _active_minutes(stamps: list[str]) -> float:
    times = sorted(t for t in (parse_ts(s) for s in stamps if s) if t)
    total = 0.0
    for a, b in zip(times, times[1:]):
        total += min((b - a).total_seconds(), ACTIVE_GAP_SECONDS)
    return total / 60


def analyze(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    project: str | None = None,
    days_back: int | None = 30,
) -> Analysis:
    clauses, params = session_filters(sources, project, days_back)
    where = " AND ".join(clauses) or "1=1"
    rows = index.conn.execute(f"SELECT * FROM sessions s WHERE {where} ORDER BY ended_at DESC", params).fetchall()
    stats: dict[str, SessionStats] = {}
    for row in rows:
        stats[row["session_key"]] = SessionStats(
            session_key=row["session_key"], source=row["source"], session_id=row["session_id"],
            title=row["title"], project=row["project"], date=row["session_date"], model=row["model"],
            input_tokens=row["input_tokens"], output_tokens=row["output_tokens"], n_user=row["n_user"],
        )

    commands: Counter = Counter()
    failures: Counter = Counter()
    failure_examples: dict[str, str] = {}
    files: Counter = Counter()
    stamps: dict[str, list[str]] = defaultdict(list)
    streak: dict[str, tuple[str, int]] = {}

    events = index.conn.execute(
        f"""
        SELECT e.session_key, e.idx, e.ts, e.kind, e.canon_tool, e.cmd_sig, e.paths, e.is_error, e.meta, e.text
        FROM events e JOIN sessions s ON s.session_key = e.session_key
        WHERE {where}
        ORDER BY e.session_key, e.idx
        """,
        params,
    )
    for ev in events:
        st = stats.get(ev["session_key"])
        if st is None:
            continue
        if ev["ts"]:
            stamps[ev["session_key"]].append(ev["ts"])
        meta = json.loads(ev["meta"]) if ev["meta"] else {}
        if meta.get("signal"):
            st.signals[meta["signal"]] += 1
        kind = ev["kind"]
        if kind == "tool_call":
            st.tool_calls += 1
            st.tools[ev["canon_tool"] or "other"] += 1
            if ev["canon_tool"] in ("edit", "write"):
                st.edits += 1
                for path in json.loads(ev["paths"]) if ev["paths"] else []:
                    files[path] += 1
        elif kind == "tool_result":
            st.tool_results += 1
            if ev["is_error"]:
                st.errors += 1
            if ev["canon_tool"] == "shell" and ev["cmd_sig"]:
                sig = ev["cmd_sig"]
                commands[sig] += 1
                if ev["is_error"]:
                    failures[sig] += 1
                    if sig not in failure_examples and ev["text"]:
                        failure_examples[sig] = ev["text"].strip().splitlines()[-1][:160] if ev["text"].strip() else ""
                    prev_sig, count = streak.get(st.session_key, ("", 0))
                    count = count + 1 if prev_sig == sig else 1
                    streak[st.session_key] = (sig, count)
                    if count == RETRY_LOOP_MIN:
                        st.retry_loops.append(sig)
                else:
                    streak[st.session_key] = ("", 0)

    for key, st in stats.items():
        st.active_minutes = _active_minutes(stamps.get(key, []))

    scope = project or "all projects"
    window = f"last {days_back} days" if days_back else "all time"
    return Analysis(
        scope=f"{scope} · {window}",
        sessions=list(stats.values()),
        commands=commands,
        command_failures=failures,
        failure_examples=failure_examples,
        files=files,
        comparison=_cross_agent_comparison(index, stats),
    )


def _cross_agent_comparison(index: HistoryGraphIndex, stats: dict[str, SessionStats], limit: int = 5) -> list[dict]:
    from .skills import _is_candidate_seed  # shared noise filter for extracted terms

    if len({s.source for s in stats.values()}) < 2:
        return []
    keys = list(stats)
    by_term: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for chunk_start in range(0, len(keys), 500):
        chunk = keys[chunk_start:chunk_start + 500]
        rows = index.conn.execute(
            f"""
            SELECT st.session_key, n.node_key, n.kind, n.value
            FROM session_terms st JOIN nodes n ON n.node_key = st.term_key
            WHERE st.session_key IN ({','.join('?' for _ in chunk)}) AND n.kind IN ('command', 'api', 'error', 'topic')
            """,
            chunk,
        ).fetchall()
        for row in rows:
            by_term[(row["node_key"], row["kind"], row["value"])].add(row["session_key"])

    results = []
    for (term_key, kind, value), session_keys in by_term.items():
        sources = {stats[k].source for k in session_keys}
        if len(sources) < 2 or len(session_keys) < 3 or not _is_candidate_seed(kind, value, term_key):
            continue
        per_source = {}
        for source in sorted(sources):
            group = [stats[k] for k in session_keys if stats[k].source == source]
            per_source[source] = {
                "sessions": len(group),
                "turns": statistics.median(s.n_user for s in group),
                "tool_calls": statistics.median(s.tool_calls for s in group),
                "errors": statistics.median(s.errors for s in group),
                "tokens": statistics.median(s.input_tokens + s.output_tokens for s in group),
            }
        results.append({"term": f"{kind}:{value}", "sessions": len(session_keys), "per_source": per_source})
    results.sort(key=lambda r: (-r["sessions"], r["term"]))
    return results[:limit]


# Text output

def _fmt_tokens(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def _group_key(st: SessionStats, group_by: str) -> str:
    if group_by == "source":
        return st.source
    if group_by == "day":
        return st.date or "unknown"
    if group_by == "model":
        return st.model or "unknown"
    return st.project or "(unknown project)"


def summarize(analysis: Analysis) -> dict:
    sessions = analysis.sessions
    results = sum(s.tool_results for s in sessions)
    return {
        "sessions": len(sessions),
        "active_hours": sum(s.active_minutes for s in sessions) / 60,
        "input_tokens": sum(s.input_tokens for s in sessions),
        "output_tokens": sum(s.output_tokens for s in sessions),
        "tool_calls": sum(s.tool_calls for s in sessions),
        "errors": sum(s.errors for s in sessions),
        "error_rate": (sum(s.errors for s in sessions) / results) if results else 0.0,
        "edits": sum(s.edits for s in sessions),
        "signals": sum((s.signals for s in sessions), Counter()),
        "retry_loops": sum(len(s.retry_loops) for s in sessions),
        "tools": sum((s.tools for s in sessions), Counter()),
    }


def format_analysis(analysis: Analysis, group_by: str = "project", top: int = 8) -> str:
    sessions = analysis.sessions
    if not sessions:
        return f"No sessions found for {analysis.scope}."
    total = summarize(analysis)
    lines = [
        f"# Session analytics: {analysis.scope}",
        f"{total['sessions']} sessions · {total['active_hours']:.1f} active hours · "
        f"tokens in {_fmt_tokens(total['input_tokens'])} / out {_fmt_tokens(total['output_tokens'])} · "
        f"{total['tool_calls']} tool calls · error rate {total['error_rate']:.0%}",
        "",
        f"## By {group_by}",
        f"{'GROUP':<40} {'SESS':>5} {'HOURS':>6} {'TOKENS':>8} {'TOOLS':>6} {'ERR%':>5}",
    ]
    groups: dict[str, list[SessionStats]] = defaultdict(list)
    for st in sessions:
        groups[_group_key(st, group_by)].append(st)
    for name, group in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top * 2]:
        results = sum(s.tool_results for s in group)
        err = sum(s.errors for s in group) / results if results else 0
        tokens = sum(s.input_tokens + s.output_tokens for s in group)
        hours = sum(s.active_minutes for s in group) / 60
        label = name if len(name) <= 40 else "…" + name[-39:]
        lines.append(f"{label:<40} {len(group):>5} {hours:>6.1f} {_fmt_tokens(tokens):>8} {sum(s.tool_calls for s in group):>6} {err:>5.0%}")

    if total["tools"]:
        lines.extend(["", "## Tool mix"])
        lines.append(", ".join(f"{tool} {n}" for tool, n in total["tools"].most_common(top + 2)))

    if analysis.commands:
        lines.extend(["", "## Top commands (runs / failures)"])
        for sig, n in analysis.commands.most_common(top):
            lines.append(f"- `{sig}` {n} / {analysis.command_failures.get(sig, 0)}")

    hotspots = [(sig, f) for sig, f in analysis.command_failures.most_common() if f >= 2][:top]
    if hotspots:
        lines.extend(["", "## Error hotspots"])
        for sig, f in hotspots:
            rate = f / analysis.commands[sig] if analysis.commands[sig] else 0
            example = analysis.failure_examples.get(sig, "")
            lines.append(f"- `{sig}` failed {f}× ({rate:.0%})" + (f" — e.g. {example}" if example else ""))

    if analysis.files:
        lines.extend(["", "## Most edited files"])
        lines.extend(f"- {path} ({n})" for path, n in analysis.files.most_common(top))

    signals = total["signals"]
    lines.extend([
        "", "## Friction",
        f"retry loops {total['retry_loops']} · corrections {signals['correction']} · interrupts {signals['interrupt']} · "
        f"rejected tool calls {signals['rejected']} · compactions {signals['compact']}",
    ])
    worst = sorted((s for s in sessions if s.friction), key=lambda s: -s.friction)[:5]
    for st in worst:
        parts = []
        if st.retry_loops:
            parts.append("retry loop on " + ", ".join(f"`{c}`" for c in st.retry_loops[:3]))
        parts.extend(f"{n} {name}" for name, n in st.signals.items() if name in ("correction", "interrupt", "rejected", "compact"))
        lines.append(f"- {st.source} {st.date} \"{st.title[:50]}\": {'; '.join(parts)}")
    exploring = [s for s in sessions if s.tool_calls >= 20 and s.edits == 0]
    if exploring:
        lines.append(f"{len(exploring)} long session(s) with 20+ tool calls and no edits (exploration-heavy).")

    if analysis.comparison:
        lines.extend(["", "## Cross-agent comparison (directional: medians per session, not controlled)"])
        for item in analysis.comparison:
            per = " | ".join(
                f"{src}: {d['sessions']} sess, {d['turns']:.0f} turns, {d['tool_calls']:.0f} tools, {d['errors']:.0f} err, {_fmt_tokens(d['tokens'])} tok"
                for src, d in item["per_source"].items()
            )
            lines.append(f"- {item['term']}: {per}")
    return "\n".join(lines)


# HTML report

_CSS = """
:root{--bg:#fafaf9;--panel:#ffffff;--text:#1c1917;--muted:#78716c;--line:#e7e5e4;--accent:#2563eb;--bad:#dc2626;--bar:#93c5fd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c0a09;--panel:#1c1917;--text:#f5f5f4;--muted:#a8a29e;--line:#292524;--accent:#60a5fa;--bad:#f87171;--bar:#1d4ed8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.wrap{overflow-x:auto}
.bar{height:8px;border-radius:4px;background:var(--bar);min-width:2px}code{font-size:13px;word-break:break-all}
.bad{color:var(--bad)}ul{padding-left:20px}li{margin:4px 0}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _bar_rows(items: list[tuple[str, float, str]]) -> str:
    peak = max((v for _l, v, _x in items), default=1) or 1
    rows = "".join(
        f"<tr><td><code>{_esc(label)}</code></td><td style='width:45%'><div class='bar' style='width:{100 * v / peak:.1f}%'></div></td>"
        f"<td class='num'>{_esc(extra)}</td></tr>"
        for label, v, extra in items
    )
    return f"<div class='wrap'><table>{rows}</table></div>"


def render_html(analysis: Analysis, sections: list[tuple[str, list[tuple[str, str]]]] | None = None) -> str:
    total = summarize(analysis)
    sessions = analysis.sessions
    by_source: dict[str, list[SessionStats]] = defaultdict(list)
    for st in sessions:
        by_source[st.source].append(st)

    cards = [
        ("Sessions", total["sessions"]),
        ("Active hours", f"{total['active_hours']:.1f}"),
        ("Tokens in / out", f"{_fmt_tokens(total['input_tokens'])} / {_fmt_tokens(total['output_tokens'])}"),
        ("Tool calls", total["tool_calls"]),
        ("Error rate", f"{total['error_rate']:.0%}"),
        ("Retry loops", total["retry_loops"]),
    ]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Agent History Report</title><style>{_CSS}</style></head><body><main>",
        "<h1>Agent history report</h1>",
        f"<div class='muted'>{_esc(analysis.scope)} · generated {date.today().isoformat()} · local, deterministic, no model calls</div>",
        "<div class='cards'>" + "".join(f"<div class='card'><span class='muted'>{_esc(k)}</span><b>{_esc(v)}</b></div>" for k, v in cards) + "</div>",
    ]

    if by_source:
        rows = []
        for source, group in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
            results = sum(s.tool_results for s in group)
            err = sum(s.errors for s in group) / results if results else 0
            sig = sum((s.signals for s in group), Counter())
            rows.append(
                f"<tr><td>{_esc(source)}</td><td class='num'>{len(group)}</td>"
                f"<td class='num'>{sum(s.active_minutes for s in group) / 60:.1f}</td>"
                f"<td class='num'>{_fmt_tokens(sum(s.input_tokens + s.output_tokens for s in group))}</td>"
                f"<td class='num'>{sum(s.tool_calls for s in group)}</td><td class='num'>{err:.0%}</td>"
                f"<td class='num'>{sig['correction']}</td><td class='num'>{sig['interrupt']}</td></tr>"
            )
        parts.append("<h2>By agent</h2><div class='wrap'><table><tr><th>Agent</th><th class='num'>Sessions</th><th class='num'>Hours</th>"
                     "<th class='num'>Tokens</th><th class='num'>Tool calls</th><th class='num'>Errors</th>"
                     "<th class='num'>Corrections</th><th class='num'>Interrupts</th></tr>" + "".join(rows) + "</table></div>")

    projects: dict[str, list[SessionStats]] = defaultdict(list)
    for st in sessions:
        projects[st.project or "(unknown)"].append(st)
    if projects:
        top_projects = sorted(projects.items(), key=lambda kv: -sum(s.active_minutes for s in kv[1]))[:10]
        parts.append("<h2>Where time went</h2>")
        parts.append(_bar_rows([
            (name, sum(s.active_minutes for s in group), f"{sum(s.active_minutes for s in group) / 60:.1f} h · {len(group)} sess")
            for name, group in top_projects
        ]))

    if total["tools"]:
        parts.append("<h2>Tool mix</h2>")
        parts.append(_bar_rows([(tool, n, str(n)) for tool, n in total["tools"].most_common(10)]))

    hotspots = [(sig, f) for sig, f in analysis.command_failures.most_common() if f >= 2][:10]
    if hotspots:
        parts.append("<h2>Error hotspots</h2>")
        parts.append(_bar_rows([
            (sig, f, f"{f}/{analysis.commands[sig]} failed" + (f" · {analysis.failure_examples.get(sig, '')[:60]}" if analysis.failure_examples.get(sig) else ""))
            for sig, f in hotspots
        ]))

    worst = sorted((s for s in sessions if s.friction), key=lambda s: -s.friction)[:8]
    if worst:
        items = []
        for st in worst:
            bits = []
            if st.retry_loops:
                bits.append("retry loop: " + ", ".join(st.retry_loops[:3]))
            bits.extend(f"{n} {name}" for name, n in st.signals.items() if name in ("correction", "interrupt", "rejected", "compact"))
            items.append(f"<li><b>{_esc(st.title[:70])}</b> <span class='muted'>{_esc(st.source)} · {_esc(st.date)}</span><br>"
                         f"<span class='bad'>{_esc('; '.join(bits))}</span></li>")
        parts.append("<h2>Highest-friction sessions</h2><ul>" + "".join(items) + "</ul>")

    if analysis.comparison:
        rows = []
        for item in analysis.comparison:
            for source, d in item["per_source"].items():
                rows.append(f"<tr><td><code>{_esc(item['term'])}</code></td><td>{_esc(source)}</td><td class='num'>{d['sessions']}</td>"
                            f"<td class='num'>{d['turns']:.0f}</td><td class='num'>{d['tool_calls']:.0f}</td>"
                            f"<td class='num'>{d['errors']:.0f}</td><td class='num'>{_fmt_tokens(d['tokens'])}</td></tr>")
        parts.append("<h2>Same topic, different agents</h2><p class='muted'>Median per session. Directional only: tasks are not controlled.</p>"
                     "<div class='wrap'><table><tr><th>Topic</th><th>Agent</th><th class='num'>Sessions</th><th class='num'>Turns</th>"
                     "<th class='num'>Tool calls</th><th class='num'>Errors</th><th class='num'>Tokens</th></tr>" + "".join(rows) + "</table></div>")

    for title, items in sections or []:
        if not items:
            continue
        parts.append(f"<h2>{_esc(title)}</h2><ul>" + "".join(
            f"<li><b>{_esc(head)}</b>" + (f"<br><span class='muted'>{_esc(detail)}</span>" if detail else "") + "</li>"
            for head, detail in items
        ) + "</ul>")

    parts.append("</main></body></html>")
    return "".join(parts)
