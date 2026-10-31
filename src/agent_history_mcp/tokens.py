"""
Where the tokens went, from the per-call usage in the api_calls table.

Numbers only (no transcript text): cache hit rate by source and model, cache
writes after idle gaps (the cache expired), what each compaction cost, and an
estimate of how many tokens big tool outputs cost by being re-sent on every
later call until the next compaction.
"""

from __future__ import annotations

import json
import statistics
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import PurePath

from .analytics import _fmt_tokens
from .graph import HistoryGraphIndex, session_filters
from .model import parse_ts

CACHE_TTL_SECONDS = 300      # Claude's default cache lifetime; a longer gap means a re-write
CHARS_PER_TOKEN = 4          # ponytail: rough English/code average, no tokenizer


def _prompt(call: dict) -> int:
    return call["input"] + call["cache_read"] + call["cache_write_5m"] + call["cache_write_1h"]


def _pct(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def token_report(
    index: HistoryGraphIndex,
    sources: list[str] | None = None,
    project: str | None = None,
    days_back: int | None = None,
    top: int = 10,
) -> str:
    clauses, params = session_filters(sources, project, days_back)
    where = " AND ".join(clauses) or "1=1"
    conn = index.conn
    sessions = {
        r["session_key"]: r for r in conn.execute(
            f"SELECT session_key, session_id, source, project, cwd FROM sessions s WHERE {where}", params)
    }
    calls: dict[str, list[dict]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT a.* FROM api_calls a JOIN sessions s USING(session_key) WHERE {where} ORDER BY a.session_key, a.idx", params
    ):
        calls[r["session_key"]].append(dict(r))
    if not calls:
        return "No per-call token usage indexed for this scope (run `agent-history refresh`)."
    # Tool outputs with a known full size, and compaction markers.
    marks: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for r in conn.execute(
        f"SELECT e.session_key, e.ts, e.tool, e.meta FROM events e JOIN sessions s USING(session_key) "
        f"WHERE {where} AND (e.meta LIKE '%out_chars%' OR e.meta LIKE '%\"compact\"%') ORDER BY e.session_key, e.idx", params
    ):
        marks[r["session_key"]].append((r["ts"] or "", r["tool"] or "?", json.loads(r["meta"])))

    by_source: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)
    idle = defaultdict(lambda: [0, 0])          # source -> [gaps, tokens re-written after them]
    compactions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    resent = defaultdict(lambda: [0, 0, 0])     # tool -> [outputs, chars, est. tokens re-sent]
    per_session = []

    for key, rows in calls.items():
        source = sessions[key]["source"]
        by_source[source].extend(rows)
        for c in rows:
            by_model[c["model"] or "?"].append(c)
        main = [c for c in rows if not c["sidechain"]]
        for prev, cur in zip(main, main[1:]):
            a, b = parse_ts(prev["ts"]), parse_ts(cur["ts"])
            if a and b and (b - a).total_seconds() > CACHE_TTL_SECONDS:
                idle[source][0] += 1
                idle[source][1] += cur["cache_write_5m"] + cur["cache_write_1h"]
        call_ts = [c["ts"] or "" for c in main]
        compact_ts = []
        for ts, _tool, meta in marks.get(key, ()):
            if meta.get("signal") == "compact":
                compact_ts.append(ts)
                if meta.get("pre_tokens"):
                    compactions[source].append((meta["pre_tokens"], meta.get("post_tokens") or 0))
        for ts, tool, meta in marks.get(key, ()):
            if "out_chars" not in meta:
                continue
            # The output rides along in every later call until a compaction drops it.
            nxt = compact_ts[bisect_right(compact_ts, ts)] if bisect_right(compact_ts, ts) < len(compact_ts) else None
            later = (bisect_left(call_ts, nxt) if nxt else len(call_ts)) - bisect_right(call_ts, ts)
            stat = resent[tool]
            stat[0] += 1
            stat[1] += meta["out_chars"]
            stat[2] += meta["out_chars"] // CHARS_PER_TOKEN * max(0, later)
        per_session.append((sum(_prompt(c) for c in rows), key))

    def table(title: str, groups: dict[str, list[dict]], limit: int) -> list[str]:
        lines = ["", f"## {title}",
                 f"{'':<28} {'CALLS':>7} {'PROMPT':>8} {'HIT%':>5} {'WRITE':>8} {'UNCACHED':>9} {'OUT':>8} {'REASON':>8}"]
        ranked = sorted(groups.items(), key=lambda kv: -sum(_prompt(c) for c in kv[1]))[:limit]
        for name, rows in ranked:
            prompt = sum(_prompt(c) for c in rows)
            write = sum(c["cache_write_5m"] + c["cache_write_1h"] for c in rows)
            label = name if len(name) <= 28 else name[:27] + "…"
            lines.append(
                f"{label:<28} {len(rows):>7} {_fmt_tokens(prompt):>8} {_pct(sum(c['cache_read'] for c in rows), prompt):>5} "
                f"{_fmt_tokens(write):>8} {_fmt_tokens(sum(c['input'] for c in rows)):>9} "
                f"{_fmt_tokens(sum(c['output'] for c in rows)):>8} {_fmt_tokens(sum(c['reasoning'] for c in rows)):>8}"
            )
        return lines

    all_calls = [c for rows in calls.values() for c in rows]
    total_prompt = sum(_prompt(c) for c in all_calls)
    lines = [
        "# Token report",
        f"{len(all_calls)} model calls in {len(calls)} sessions · prompt {_fmt_tokens(total_prompt)} "
        f"({_pct(sum(c['cache_read'] for c in all_calls), total_prompt)} read from cache) · "
        f"output {_fmt_tokens(sum(c['output'] for c in all_calls))}",
        "PROMPT = everything sent in (uncached + cache read + cache write). HIT% = share read from cache.",
    ]
    lines += table("By source", by_source, len(by_source))
    lines += table("By model", by_model, top)

    if any(gaps for gaps, _ in idle.values()):
        lines += ["", f"## Idle gaps over {CACHE_TTL_SECONDS // 60} min (cache likely expired, prompt re-written)"]
        for source, (gaps, written) in sorted(idle.items()):
            after = f"{_fmt_tokens(written)} tokens written to cache on the call after" if written else "cache writes not reported"
            lines.append(f"- {source}: {gaps} gaps, {after}")

    if compactions:
        lines += ["", "## Compactions (context size before → after)"]
        for source, pairs in sorted(compactions.items()):
            pre = [p for p, _ in pairs]
            post = [q for _, q in pairs if q]
            after = f" → median {_fmt_tokens(statistics.median(post))}" if post else ""
            lines.append(f"- {source}: {len(pairs)} compactions, median {_fmt_tokens(statistics.median(pre))}{after}")

    if resent:
        lines += ["", "## Large tool outputs, re-sent until the next compaction (estimate)",
                  f"{'TOOL':<28} {'OUTPUTS':>8} {'SIZE':>8} {'RE-SENT':>9}"]
        for tool, (n, chars, tokens) in sorted(resent.items(), key=lambda kv: -kv[1][2])[:top]:
            lines.append(f"{tool[:28]:<28} {n:>8} {_fmt_tokens(chars / CHARS_PER_TOKEN):>8} {_fmt_tokens(tokens):>9}")
        lines.append("Size is in tokens (chars/4). Mostly cache reads, so cheap per token; Codex trims output "
                     "before the model sees it, so its rows overstate.")

    lines += ["", f"## Biggest sessions by prompt tokens",
              f"{'SESSION':<10} {'SOURCE':<8} {'PROJECT':<24} {'CALLS':>6} {'PROMPT':>8} {'HIT%':>5}"]
    for prompt, key in sorted(per_session, reverse=True)[:top]:
        row, rows = sessions[key], calls[key]
        name = PurePath(row["project"] or row["cwd"] or "?").name[:24]
        lines.append(f"{row['session_id'][:8]:<10} {row['source']:<8} {name:<24} {len(rows):>6} "
                     f"{_fmt_tokens(prompt):>8} {_pct(sum(c['cache_read'] for c in rows), prompt):>5}")
    return "\n".join(lines)
